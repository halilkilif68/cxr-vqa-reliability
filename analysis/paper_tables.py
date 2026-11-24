"""Tables used in the paper that are computed from the per-pair test predictions.

Input : results/{blip,vilt,clip}_test_predictions.csv.gz  (written by notebooks 01-03;
        these files contain labels and are therefore not part of the public repository)
Output: results/threshold_metrics.csv   (Table 4: accuracy, balanced accuracy, sensitivity,
                                         specificity, precision, F1 at threshold 0.5)
        results/signed_dP_ci.csv        (Table 7: label-signed probability shift with
                                         cluster-bootstrap 95% CI and Cohen's dz)
        results/per_disease_merged.csv  (Table 5 / Figure 5: per-disease AUC of the three models)

Usage : python analysis/paper_tables.py [results_dir] [n_boot] [seed]
"""
import sys
import numpy as np
import pandas as pd

R = (sys.argv[1] if len(sys.argv) > 1 else "results").rstrip("/") + "/"
N_BOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 500
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 42
MODELS = ["blip", "vilt", "clip"]
P = {m: pd.read_csv(R + f"{m}_test_predictions.csv.gz") for m in MODELS}


def threshold_metrics(p, thr=0.5):
    out = []
    for src, d in [("all", p)] + [(s, g) for s, g in p.groupby("source")]:
        y = d.label.values.astype(int)
        pr = (d.prob_orig.values >= thr).astype(int)
        tp = int(((y == 1) & (pr == 1)).sum()); tn = int(((y == 0) & (pr == 0)).sum())
        fp = int(((y == 0) & (pr == 1)).sum()); fn = int(((y == 1) & (pr == 0)).sum())
        sens = tp / (tp + fn); spec = tn / (tn + fp); prec = tp / (tp + fp)
        out.append(dict(source=src, acc=(tp + tn) / len(d), bal_acc=(sens + spec) / 2,
                        sens=sens, spec=spec, prec=prec, f1=2 * prec * sens / (prec + sens),
                        tp=tp, tn=tn, fp=fp, fn=fn))
    return pd.DataFrame(out)


def signed_shift(p, alt):
    """(P_orig - P_alt) * (2y - 1): positive when removing the image moves the
    prediction away from the correct answer.  Clusters: subject (MIMIC-CXR) or
    image (VinDr-CXR); the `subject` column already encodes both."""
    d = (p.prob_orig.values - p[alt].values) * (2 * p.label.values - 1)
    cl = pd.factorize(p.subject)[0]
    n_cl = cl.max() + 1
    s = np.bincount(cl, weights=d, minlength=n_cl)
    c = np.bincount(cl, minlength=n_cl).astype(float)
    rng = np.random.default_rng(SEED)
    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.integers(0, n_cl, n_cl)
        boots[b] = s[i].sum() / c[i].sum()
    return dict(mean=d.mean(), lo=np.percentile(boots, 2.5), hi=np.percentile(boots, 97.5),
                dz=d.mean() / d.std(ddof=1), clusters=n_cl)


tm = pd.concat([threshold_metrics(P[m]).assign(model=m) for m in MODELS])
tm = tm[["model"] + [c for c in tm.columns if c != "model"]]
tm.to_csv(R + "threshold_metrics.csv", index=False)

rows = []
for m in MODELS:
    for alt in ["prob_black", "prob_noise"]:
        rows.append(dict(model=m, alt=alt, **signed_shift(P[m], alt)))
pd.DataFrame(rows).to_csv(R + "signed_dP_ci.csv", index=False)

pdz = None
for m in MODELS:
    t = pd.read_csv(R + f"{m}_per_disease_full.csv")[["disease", "n_pos", "reliable", "auc"]].rename(columns={"auc": m})
    pdz = t if pdz is None else pdz.merge(t[["disease", m]], on="disease")
pdz[["disease", "n_pos", "reliable"] + MODELS].to_csv(R + "per_disease_merged.csv", index=False)
print("written: threshold_metrics.csv, signed_dP_ci.csv, per_disease_merged.csv")
