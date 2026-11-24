"""Reference baselines for the pointing hit (Table 10) and the binomial tests of Section 6.7.

For every disease and every test image that enters the IoU analysis, the expert boxes of all three
radiologists are normalised by the original image size and rasterised on a grid.
  random point : probability that a uniformly random point falls inside the union of the boxes,
                 averaged over images
  central point: share of images whose centre (0.5, 0.5) lies inside the union of the boxes
The observed pointing hits of each model (results/<model>_iou_detail.csv, threshold 0.5) are then tested
against the random-point rate with a one-sided exact binomial test in both directions; the Bonferroni
level is 0.05 / 9 (three diseases x three models).

Needs VinDr-CXR files that are not part of this repository (credentialed data, run locally):
  annotations_train.csv   image_id, rad_id, class_name, x_min, y_min, x_max, y_max
  train_meta.csv          image_id, dim0 (height), dim1 (width)       [or --dicom_dir with *.dicom files]

Usage:
  python analysis/pointing_baselines.py --ann vindr/annotations_train.csv --meta vindr/train_meta.csv \
         --results results [--dicom_dir path/to/train/] [--grid 512]
"""
import argparse, os
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ap = argparse.ArgumentParser()
ap.add_argument("--ann", required=True)
ap.add_argument("--meta", default=None)
ap.add_argument("--dicom_dir", default=None)
ap.add_argument("--results", default="results")
ap.add_argument("--grid", type=int, default=512)
a = ap.parse_args()

DISEASES = ["Aortic enlargement", "Cardiomegaly", "Pleural thickening"]
MODELS = ["blip", "vilt", "clip"]
G = a.grid

ann = pd.read_csv(a.ann).dropna(subset=["x_min", "y_min", "x_max", "y_max"])
meta = {}
if a.meta:
    m = pd.read_csv(a.meta)
    meta = {r.image_id: (int(r.dim1), int(r.dim0)) for r in m.itertuples()}      # (W, H)


def size(img_id):
    if a.dicom_dir:
        p = os.path.join(a.dicom_dir, f"{img_id}.dicom")
        if os.path.exists(p):
            import pydicom
            ds = pydicom.dcmread(p, stop_before_pixels=True)
            return int(ds.Columns), int(ds.Rows)
    return meta[img_id]


def union_mask(boxes):
    m = np.zeros((G, G), bool)
    for x0, y0, x1, y1 in boxes:
        i0, i1 = int(np.floor(max(y0, 0) * G)), int(np.ceil(min(y1, 1) * G))
        j0, j1 = int(np.floor(max(x0, 0) * G)), int(np.ceil(min(x1, 1) * G))
        m[i0:i1, j0:j1] = True
    return m


detail = {m: pd.read_csv(os.path.join(a.results, f"{m}_iou_detail.csv")) for m in MODELS}
rows = []
base = {}
for dis in DISEASES:
    ids = detail["blip"].query("disease == @dis").image_id.unique()
    rnd, ctr = [], []
    for img in ids:
        W, H = size(img)
        b = ann[(ann.image_id == img) & (ann.class_name == dis)]
        boxes = [(r.x_min / W, r.y_min / H, r.x_max / W, r.y_max / H) for r in b.itertuples()]
        mk = union_mask(boxes)
        rnd.append(mk.mean()); ctr.append(bool(mk[G // 2, G // 2]))
    base[dis] = (len(ids), float(np.mean(rnd)), float(np.mean(ctr)))
    for mdl in MODELS:
        d = detail[mdl].query("disease == @dis and threshold == 0.5")
        k, n = int(d.hit.sum()), len(d)
        p = base[dis][1]
        rows.append(dict(model=mdl, disease=dis, n=n, hits=k, hit_rate=k / n, random_point=p,
                         expected_hits=n * p, central_point=base[dis][2],
                         p_greater=binomtest(k, n, p, alternative="greater").pvalue,
                         p_less=binomtest(k, n, p, alternative="less").pvalue))
out = pd.DataFrame(rows)
alpha = 0.05 / 9
out["above_random_bonferroni"] = out.p_greater < alpha
out["below_random_bonferroni"] = out.p_less < alpha
out.to_csv(os.path.join(a.results, "pointing_baselines.csv"), index=False)
pd.set_option("display.width", 220)
print(out.round(4).to_string(index=False))
