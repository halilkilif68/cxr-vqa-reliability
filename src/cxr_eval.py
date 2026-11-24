"""
cxr_eval.py  --  güvenilirlik testleri, istatistikler, model karşılaştırma ve IoU.

Eski analizlerdeki geçersizlikler burada düzeltilmiştir:
  * Aynı sabit ızgara, aynı sıra: tüm modeller ve tüm testler (orig/noise/black/shuffle) EŞLEŞİK.
  * "Blind" = sabit gerçek siyah görüntü. Sabit girdi -> çıktı yalnızca soruya bağlı ->
    hastalık-bazlı AUC=0.5 olması KAÇINILMAZDIR; bu yüzden "VRI = AUC_full - AUC_blind"
    hastalık düzeyinde raporlanmaz. Bunun yerine: eşleşik ΔP (Wilcoxon + Cohen dz),
    micro AUC (full vs black vs soru-önsel taban çizgisi) ve eşleşik bootstrap kullanılır.
  * Shuffle: her GÖRÜNTÜ için bağımsız permütasyon, sabit tohum.
  * Bootstrap hasta (subject) kümeleri üzerinden yapılır (aynı hastanın görüntüleri bağımsız değildir).
  * IoU: modeller eğitimdeki sınıfla + LoRA ile + strict=True yüklenir; yalnızca TEST bölmesindeki
    VinDr görüntüleri kullanılır; hata yutulmaz, atlanan görüntüler nedeniyle sayılır.
"""
from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score
try:
    from tqdm.auto import tqdm
except ImportError:                      # tqdm yoksa ilerleme çubuğu olmadan çalış
    def tqdm(x=None, **kw):
        return x

import cxr_data as D
import cxr_models as M

# --------------------------------------------------------------------------
# 1. Girdi bozma işlemleri
# --------------------------------------------------------------------------
def shuffle_patches(x: torch.Tensor, patch: int, gen: torch.Generator) -> torch.Tensor:
    """Her görüntü için BAĞIMSIZ rastgele yama permütasyonu. x: (B,C,H,W)."""
    B, C, H, W = x.shape
    assert H % patch == 0 and W % patch == 0, (H, W, patch)
    nh, nw = H // patch, W // patch
    t = x.unfold(2, patch, patch).unfold(3, patch, patch)            # B,C,nh,nw,p,p
    t = t.permute(0, 2, 3, 1, 4, 5).reshape(B, nh * nw, C, patch, patch)
    perm = torch.rand(B, nh * nw, generator=gen, device=x.device).argsort(dim=1)
    t = t[torch.arange(B, device=x.device)[:, None], perm]
    t = t.reshape(B, nh, nw, C, patch, patch).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
    return t.contiguous()


DEFAULT_MODES = (("orig", None), ("noise", None), ("black", None),
                 ("shuf", 32), ("shuf", 16), ("shuf", 8))


def mode_col(mode, arg) -> str:
    return f"prob_{mode}" + (f"{arg}" if arg else "")


@torch.no_grad()
def black_prob_by_disease(model, enc: M.Encoder, diseases, device, amp: bool = True) -> dict:
    """
    Sabit siyah girdi için her hastalık sorusunun olasılığı, TEK ileri geçişte hesaplanır.
    Böylece 'Black' sütunu bir hastalık içinde BİREBİR sabittir. (Aksi halde bf16 + farklı batch
    kompozisyonu ~1e-4'lük sayısal titreme üretir ve hastalık-bazlı AUC 0.5'ten rastgele sapar.)
    """
    ds_ = list(dict.fromkeys(diseases))
    qs = [enc.question(d) for d in ds_]
    b = {k: torch.stack([q[k] for q in qs]) for k in qs[0]}
    b["rep"] = torch.zeros(len(ds_), dtype=torch.long)
    S = enc.cfg.img_size
    img = enc.black_value(torch.zeros(1, 3, S, S)).contiguous().to(device)
    p = M.forward_rows(model, b, device, img, enc.name, amp)
    return dict(zip(ds_, p.astype(np.float32)))


@torch.no_grad()
def run_modes(model, grid: pd.DataFrame, enc: M.Encoder, device, modes=DEFAULT_MODES,
              seed: int = D.SEED, batch_images: int = 4, num_workers: int = 4,
              amp: bool = True, desc: str = "") -> pd.DataFrame:
    """
    Izgaranın her (görüntü, hastalık) çifti için, her mod altında olasılık.
    Görüntü bozma işlemleri GÖRÜNTÜ düzeyinde yapılır, sonra sorulara yayılır.
    Döndürür: grid (img_idx sıralı) + prob_<mod> sütunları.
    """
    ds = M.GridDataset(grid, enc)
    dl = M.make_loader(ds, batch_images, False, num_workers, M.grid_collate)
    out = ds.g.copy()
    cols = {mode_col(m, a): np.full(len(out), np.nan, np.float32) for m, a in modes}
    gens = {mode_col(m, a): torch.Generator(device=device).manual_seed(seed + 17 * k)
            for k, (m, a) in enumerate(modes)}
    model.eval()
    dis_arr = out.disease.values
    black_cache = (black_prob_by_disease(model, enc, dis_arr, device, amp)
                   if any(m == "black" for m, _ in modes) else None)
    for b in tqdm(dl, desc=desc or "inference"):
        pv = b["pixel_values"].to(device)
        rows = b["row_ids"].numpy()
        for (m, a) in modes:
            c = mode_col(m, a)
            if m == "black":
                cols[c][rows] = np.array([black_cache[d] for d in dis_arr[rows]], np.float32)
                continue
            if m == "orig":
                x = pv
            elif m == "noise":
                x = torch.randn(pv.shape, generator=gens[c], device=device, dtype=pv.dtype)
            elif m == "shuf":
                x = shuffle_patches(pv, a, gens[c])
            else:
                raise ValueError(m)
            cols[c][rows] = M.forward_rows(model, b, device, x, enc.name, amp)
    for c, v in cols.items():
        out[c] = v
    return out


# --------------------------------------------------------------------------
# 2. Metrikler
# --------------------------------------------------------------------------
def auc(y, p) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def per_disease_table(df: pd.DataFrame, prob_col: str, min_pos: int = 30, min_neg: int = 30) -> pd.DataFrame:
    rows = []
    for d, s in df.groupby("disease"):
        npos, nneg = int(s.label.sum()), int(len(s) - s.label.sum())
        rows.append(dict(disease=d, n=len(s), n_pos=npos, n_neg=nneg,
                         auc=auc(s.label, s[prob_col]) if (npos > 0 and nneg > 0) else np.nan,
                         reliable=(npos >= min_pos and nneg >= min_neg)))
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def macro_auc(df: pd.DataFrame, prob_col: str, min_pos=30, min_neg=30) -> float:
    t = per_disease_table(df, prob_col, min_pos, min_neg)
    return float(t.loc[t.reliable, "auc"].mean())


def headline_table(df: pd.DataFrame, prob_cols: dict[str, str]) -> pd.DataFrame:
    """prob_cols: {'Full':'prob_orig','Shuffle-32':'prob_shuf32',...} -> micro/macro/kaynak bazlı AUC."""
    rows = []
    for label, c in prob_cols.items():
        r = dict(condition=label, micro_auc=auc(df.label, df[c]), macro_auc=macro_auc(df, c))
        for src, s in df.groupby("source"):
            r[f"auc_{src}"] = auc(s.label, s[c])
        rows.append(r)
    return pd.DataFrame(rows)


def text_prior_baseline(grid: pd.DataFrame, train_df: pd.DataFrame, label_cols) -> np.ndarray:
    """
    Görüntüye hiç bakmayan taban çizgisi: (kaynak, hastalık) için EĞİTİM prevalansı.
    micro AUC'nin ne kadarının yalnızca 'hangi hastalık soruldu' bilgisinden geldiğini gösterir.
    """
    prev = {}
    for src, d in train_df.groupby("source"):
        for c in label_cols:
            v = d[c].dropna()
            if len(v):
                prev[(src, c)] = float(v.mean())
    return np.array([prev.get((s, d), np.nan) for s, d in zip(grid.source, grid.disease)])


# --------------------------------------------------------------------------
# 3. Eşleşik testler
# --------------------------------------------------------------------------
def fmt_p(p: float) -> str:
    return "<1e-300" if (p is None or p < 1e-300) else f"{p:.3g}"


def wilcoxon_signed_rank(d: np.ndarray) -> tuple[float, float]:
    """
    Eşleşik farklar d için işaretli-sıra testi (normal yaklaşım, bağ düzeltmeli, iki yönlü).
    scipy sürümünden bağımsız ve büyük n'de p-değeri alt-taşması yok. Döndürür: (z, log10 p).
    """
    d = np.asarray(d, float)
    d = d[(d != 0) & ~np.isnan(d)]
    n = len(d)
    if n == 0:
        return 0.0, 0.0
    ranks = stats.rankdata(np.abs(d))
    w_plus = ranks[d > 0].sum()
    mean = n * (n + 1) / 4.0
    _, counts = np.unique(np.abs(d), return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24.0 - ((counts ** 3 - counts).sum()) / 48.0
    z = (w_plus - mean) / np.sqrt(var) if var > 0 else 0.0
    log10_p = (np.log(2.0) + stats.norm.logsf(abs(z))) / np.log(10.0)
    return float(z), float(min(log10_p, 0.0))


def paired_tests(a: np.ndarray, b: np.ndarray) -> dict:
    """Eşleşik fark d=a-b için: ortalama, medyan, Cohen dz, Wilcoxon işaretli-sıra (z, p)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    z, lp = wilcoxon_signed_rank(d)
    sd = d.std(ddof=1)
    return dict(n=len(d), mean_diff=float(d.mean()), median_diff=float(np.median(d)),
                cohen_dz=float(d.mean() / sd) if sd > 0 else float("nan"),
                wilcoxon_z=z, wilcoxon_p=fmt_p(10 ** lp), frac_positive=float((d > 0).mean()))


def visual_dependence_table(df: pd.DataFrame, ref="prob_orig", alt=("prob_noise", "prob_black")) -> pd.DataFrame:
    """
    ΔP = P(orig) - P(alt). Tümü / yalnızca pozitifler / yalnızca negatifler, eşleşik testlerle.
    + "all (label-signed)": ΔP'nin etikete göre işaretli hali, (P(orig)-P(alt))*(2y-1).
      Pozitif değer = gerçek görüntü tahmini DOĞRU etikete doğru kaydırıyor (görsel kanıt kullanılıyor).
      Ham ΔP'nin işareti kalibrasyona bağlıdır (OOD girdide olasılık ~0.45 çıkabilir; model gerçek
      görüntülerde düşük prevalansı yansıttığından ham ΔP negatif olabilir) — bu yüzden yorum için
      işaretli sürüm ve AUC farkları kullanılmalıdır. frac_positive = doğru yöne kayan çiftlerin oranı.
    """
    rows = []
    y = df.label.values
    for c in alt:
        for name, m in (("all", np.ones(len(df), bool)), ("positives", y == 1), ("negatives", y == 0)):
            rows.append(dict(condition=c, subset=name, **paired_tests(df.loc[m, ref], df.loc[m, c])))
        d = (df[ref].values - df[c].values) * (2 * y - 1)
        rows.append(dict(condition=c, subset="all (label-signed)", **paired_tests(d, np.zeros_like(d))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4. Küme (hasta) bootstrap'ı
# --------------------------------------------------------------------------
def _cluster_index(groups: pd.Series):
    codes, uniq = pd.factorize(groups)
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    return order, starts, len(uniq)


def bootstrap_auc(df: pd.DataFrame, prob_cols: list[str], diffs: list[tuple[str, str]] = (),
                  n_boot: int = 500, seed: int = D.SEED, group: str = "subject",
                  stat: str = "micro", alpha: float = 0.05) -> pd.DataFrame:
    """
    Aynı yeniden örneklemeler üzerinde tüm sütunların AUC'sini ve verilen farkları
    (a - b) hesaplar. Yeniden örnekleme birimi = hasta (VinDr'de görüntü).
    stat: 'micro' veya 'macro' (güvenilir hastalıklar, n>=30 poz/neg).
    """
    y = df.label.values
    P = {c: df[c].values for c in set(prob_cols) | {x for pair in diffs for x in pair}}
    dis = pd.factorize(df.disease)[0]
    order, starts, G = _cluster_index(df[group])
    rng = np.random.RandomState(seed)

    def stat_fn(idx, c):
        if stat == "micro":
            return auc(y[idx], P[c][idx])
        vals = []
        yi, pi, di = y[idx], P[c][idx], dis[idx]
        for d in np.unique(di):
            m = di == d
            if yi[m].sum() >= 30 and (~(yi[m] == 1)).sum() >= 30:
                vals.append(auc(yi[m], pi[m]))
        return float(np.mean(vals)) if vals else float("nan")

    full = np.arange(len(df))
    point = {c: stat_fn(full, c) for c in P}
    boots = {c: [] for c in P}
    for _ in tqdm(range(n_boot), desc="bootstrap"):
        pick = rng.randint(0, G, G)
        idx = np.concatenate([order[starts[g]:starts[g + 1]] for g in pick])
        for c in P:
            boots[c].append(stat_fn(idx, c))
    rows = []
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    for c in prob_cols:
        lo, hi = np.nanpercentile(boots[c], q)
        rows.append(dict(item=c, estimate=point[c], ci_low=lo, ci_high=hi))
    for a, b in diffs:
        d = np.array(boots[a]) - np.array(boots[b])
        lo, hi = np.nanpercentile(d, q)
        # iki yönlü bootstrap p (yaklaşık): 0'ın hangi tarafında kalma oranı
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        rows.append(dict(item=f"{a} - {b}", estimate=point[a] - point[b], ci_low=lo, ci_high=hi,
                         boot_p=max(p, 1.0 / (n_boot + 1))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 5. Modeller arası karşılaştırma (eşleşik)
# --------------------------------------------------------------------------
def merge_models(preds: dict[str, pd.DataFrame], cols=("prob_orig",)) -> pd.DataFrame:
    """Aynı ızgara üzerinde üretilen tahminleri (image_path, disease) anahtarıyla yan yana koyar."""
    key = ["image_path", "disease"]
    base = None
    for name, df in preds.items():
        keep = key + ["source", "subject", "label"] + list(cols) if base is None else key + list(cols)
        t = df[keep].rename(columns={c: f"{c}__{name}" for c in cols})
        base = t if base is None else base.merge(t, on=key, how="inner", validate="one_to_one")
    return base


def compare_models(preds: dict[str, pd.DataFrame], col="prob_orig", n_boot=500) -> dict[str, pd.DataFrame]:
    """
    İkili karşılaştırmalar:
      (a) ΔAUC (micro) + hasta-küme bootstrap GA ve p
      (b) Eşleşik Wilcoxon: mutlak hata |p - y| üzerinde
    """
    m = merge_models(preds, cols=(col,))
    names = list(preds)
    pcols = [f"{col}__{n}" for n in names]
    pairs = [(f"{col}__{a}", f"{col}__{b}") for a, b in itertools.combinations(names, 2)]
    boot = bootstrap_auc(m, pcols, diffs=pairs, n_boot=n_boot)
    wil = []
    for a, b in itertools.combinations(names, 2):
        ea = np.abs(m[f"{col}__{a}"] - m.label)
        eb = np.abs(m[f"{col}__{b}"] - m.label)
        wil.append(dict(pair=f"{a} vs {b}", **paired_tests(ea, eb)))
    return {"auc_bootstrap": boot, "abs_error_wilcoxon": pd.DataFrame(wil)}


# --------------------------------------------------------------------------
# 6. Grad-CAM / dikkat haritası ve IoU
# --------------------------------------------------------------------------
def _cam_target(model, name):
    if name == "blip":
        return model.vision_model.post_layernorm            # çapraz-dikkate giren son görsel özellikler
    if name == "clip":
        return model.clip.get_base_model().vision_model.encoder.layers[-1].layer_norm1
    raise ValueError(name)


def gradcam_map(model, name: str, enc: M.Encoder, image_path: str, disease: str, device) -> np.ndarray:
    """
    BLIP/CLIP için Grad-CAM (ağırlık = token boyunca ortalama gradyan; ReLU). Skor = tek lojit.
    Girdi piksellerine requires_grad verilir, böylece donuk görsel kodlayıcıda da gradyan akar.
    NOT (CLIP): eski kod son bloğun layer_norm2 çıktısını kullanıyordu; yama jetonlarının oradaki
    çıktısı CLS'e hiç ulaşmadığı için gradyan sıfırdı. Burada layer_norm1 kullanılır.
    """
    model.eval()
    pv = enc.image(image_path).unsqueeze(0).to(device).requires_grad_(True)
    q = {k: v.unsqueeze(0).to(device) for k, v in enc.question(disease).items()}
    store = {}

    def hook(_, __, o):
        if o.dim() == 3 and o.shape[1] > 1 and "act" not in store:   # BLIP post_layernorm iki kez çağrılır
            store["act"] = o

    h = _cam_target(model, name).register_forward_hook(hook)
    try:
        with torch.enable_grad():
            logit = model(pixel_values=pv, **q)
            act = store["act"]
            g = torch.autograd.grad(logit.sum(), act)[0]
    finally:
        h.remove()
    w = g.mean(dim=1, keepdim=True)
    cam = torch.relu((w * act).sum(-1))[0, 1:]                     # CLS hariç
    side = int(round(cam.numel() ** 0.5))
    return cam.detach().float().cpu().numpy().reshape(side, side)


def vilt_attention_map(model, enc: M.Encoder, image_path: str, disease: str, device) -> np.ndarray:
    """ViLT: son katman, kafa ortalaması, metin [CLS] -> son 144 görüntü yaması (12x12)."""
    model.eval()
    pv = enc.image(image_path).unsqueeze(0).to(device)
    q = {k: v.unsqueeze(0).to(device) for k, v in enc.question(disease).items()}
    with torch.no_grad():
        _, atts = model(pixel_values=pv, output_attentions=True, **q)
    a = atts[-1].mean(dim=1)[0, 0, -144:]
    return a.float().cpu().numpy().reshape(12, 12)


def heatmap(model, name, enc, image_path, disease, device):
    return (vilt_attention_map(model, enc, image_path, disease, device) if name == "vilt"
            else gradcam_map(model, name, enc, image_path, disease, device))


def map_to_box(m: np.ndarray, thr: float, size: int = 512):
    """min-max normalize -> eşik (>= thr, tepeye göre göreli) -> en büyük bileşen -> normalize kutu."""
    import cv2
    r = cv2.resize(m.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
    lo, hi = float(r.min()), float(r.max())
    if hi - lo < 1e-12:
        return None
    r = (r - lo) / (hi - lo)
    mask = (r >= thr).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return [x / size, y / size, (x + w) / size, (y + h) / size]


def box_iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


_META_CACHE = {}


def _load_meta(meta_csv):
    """vindr/train_meta.csv: image_id, dim0, dim1  (dim0=yükseklik/satır, dim1=genişlik/sütun varsayımı;
    DICOM varsa run_iou içinde çapraz doğrulanır)."""
    if meta_csv and meta_csv not in _META_CACHE:
        m = pd.read_csv(meta_csv)
        _META_CACHE[meta_csv] = {r.image_id: (int(r.dim0), int(r.dim1)) for r in m.itertuples()}
    return _META_CACHE.get(meta_csv, {})


def _image_size(image_id: str, png_path: str, dicom_dir: str | None, meta_csv: str | None = None):
    """Kutular DICOM piksel uzayında. Öncelik: DICOM başlığı -> train_meta.csv -> (src='png', güvenilmez).
    Döner: (W, H, kaynak) ; kaynak in {'dicom','meta','png'}"""
    if dicom_dir:
        p = os.path.join(dicom_dir, f"{image_id}.dicom")
        if os.path.exists(p):
            import pydicom
            ds = pydicom.dcmread(p, stop_before_pixels=True)
            return int(ds.Columns), int(ds.Rows), "dicom"
    if meta_csv is None and os.path.exists("vindr/train_meta.csv"):
        meta_csv = "vindr/train_meta.csv"
    meta = _load_meta(meta_csv)
    if image_id in meta:
        d0, d1 = meta[image_id]
        return d1, d0, "meta"          # W = dim1, H = dim0 (run_iou 278 DICOM ile doğrular)
    from PIL import Image
    with Image.open(png_path) as im:
        return im.size[0], im.size[1], "png"


def run_iou(name, model, enc, test_df: pd.DataFrame, annotations_csv: str, dicom_dir: str | None,
            device, diseases=("Aortic enlargement", "Cardiomegaly", "Pleural thickening"),
            thresholds=(0.5, 0.7, 0.85), max_images_per_disease: int | None = None,
            allow_png_size: bool = False, meta_csv: str | None = "auto"):
    """
    Yalnızca TEST bölmesindeki VinDr görüntüleri. Her (görüntü, hastalık) için:
      iou_best : tahmin kutusunun, o hastalığa ait GT kutularından EN İYİ eşleşmesi
      iou_mean : tüm GT kutularına (radyolog başına) ortalama
      hit      : ısı haritası tepe noktası herhangi bir GT kutusunun içinde mi
      center_iou_best : sabit merkez kutusu taban çizgisi (şans düzeyi referansı)
    Döndürür: (ayrıntı df, özet df, atlama nedenleri sözlüğü)
    """
    ann = pd.read_csv(annotations_csv).dropna(subset=["x_min", "y_min", "x_max", "y_max"])
    vt = test_df[test_df.source == "vindr"].copy()
    vt["image_id"] = vt.image_path.map(lambda p: os.path.splitext(os.path.basename(p))[0])
    id2path = dict(zip(vt.image_id, vt.image_path))
    center = [0.25, 0.25, 0.75, 0.75]
    if meta_csv == "auto":
        c = os.path.join(os.path.dirname(annotations_csv), "train_meta.csv")
        meta_csv = c if os.path.exists(c) else None
    rows, skipped = [], {"no_png": 0, "no_map": 0, "no_dicom": 0, "box_out_of_range": 0,
                         "size_dicom": 0, "size_meta": 0, "meta_dicom_conflict": 0}
    for dis in diseases:
        a = ann[(ann.class_name == dis) & ann.image_id.isin(set(id2path))]
        ids = a.image_id.unique()
        if max_images_per_disease:
            ids = ids[:max_images_per_disease]
        for img_id in tqdm(ids, desc=f"IoU {name}/{dis}"):
            path = id2path[img_id]
            if not os.path.exists(path):
                skipped["no_png"] += 1
                continue
            W, H, src = _image_size(img_id, path, dicom_dir, meta_csv)
            skipped["size_dicom"] += int(src == "dicom"); skipped["size_meta"] += int(src == "meta")
            if src == "dicom" and meta_csv:                      # çapraz doğrulama (yükseklik/genişlik sırası)
                mm = _load_meta(meta_csv).get(img_id)
                if mm is not None and (mm[1], mm[0]) != (W, H):
                    skipped["meta_dicom_conflict"] += 1
            if src == "png" and not allow_png_size:
                # Kutular DICOM piksel uzayında: PNG boyutuyla bölmek YANLIŞ normalize eder -> atla
                skipped["no_dicom"] += 1
                continue
            gts = [[r.x_min / W, r.y_min / H, r.x_max / W, r.y_max / H] for r in a[a.image_id == img_id].itertuples()]
            if any(g[2] > 1.02 or g[3] > 1.02 or g[0] < -0.02 or g[1] < -0.02 for g in gts):
                skipped["box_out_of_range"] += 1
                continue
            m = heatmap(model, name, enc, path, dis, device)
            gy, gx = np.unravel_index(int(np.argmax(m)), m.shape)
            px, py = (gx + 0.5) / m.shape[1], (gy + 0.5) / m.shape[0]
            hit = any(g[0] <= px <= g[2] and g[1] <= py <= g[3] for g in gts)
            cb = max(box_iou(center, g) for g in gts)
            for t in thresholds:
                b = map_to_box(m, t)
                if b is None:
                    skipped["no_map"] += 1
                    ious = [0.0]
                else:
                    ious = [box_iou(b, g) for g in gts]
                rows.append(dict(model=name, disease=dis, image_id=img_id, threshold=t, n_boxes=len(gts),
                                 iou_best=max(ious), iou_mean=float(np.mean(ious)), hit=bool(hit),
                                 center_iou_best=cb))
    res = pd.DataFrame(rows)
    summ = (res.groupby(["model", "disease", "threshold"])
               .agg(n=("image_id", "nunique"), iou_best=("iou_best", "mean"), iou_mean=("iou_mean", "mean"),
                    pointing_hit=("hit", "mean"), center_baseline=("center_iou_best", "mean"))
               .reset_index())
    return res, summ, skipped


def plot_cam_examples(model, name, enc, items, device, path="cam_examples.png"):
    """items: [(image_path, disease, [gt_boxes_normalized])] -> kısa bir görsel kontrol."""
    import cv2
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(items), figsize=(4 * len(items), 4))
    axes = np.atleast_1d(axes)
    for ax, (p, dis, gts) in zip(axes, items):
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        m = heatmap(model, name, enc, p, dis, device)
        r = cv2.resize(m.astype(np.float32), (w, h))
        r = (r - r.min()) / (r.max() - r.min() + 1e-8)
        ax.imshow(img); ax.imshow(np.ma.masked_less(r, 0.2), cmap="jet", alpha=0.5, vmin=0, vmax=1)
        pb = map_to_box(m, 0.5)                                   # tahmin kutusu (eşik 0.5), kırmızı kesikli
        if pb:
            ax.add_patch(plt.Rectangle((pb[0] * w, pb[1] * h), (pb[2] - pb[0]) * w, (pb[3] - pb[1]) * h,
                                       fill=False, edgecolor="red", lw=2, ls="--"))
        for g in gts:
            ax.add_patch(plt.Rectangle((g[0] * w, g[1] * h), (g[2] - g[0]) * w, (g[3] - g[1]) * h,
                                       fill=False, edgecolor="lime", lw=2))
        ax.set_title(dis); ax.axis("off")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig
