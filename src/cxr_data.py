"""
cxr_data.py  --  veri hazırlama, subject-düzeyinde bölme, sabit soru ızgarası.

Bu modül torch gerektirmez (yalnızca pandas / numpy / sklearn).

Eski kodla farklar (hepsi README.md'de gerekçeleriyle listelenmiştir):
  * MIMIC-CXR hasta (subject_id) bazında bölünür -> hasta/çalışma sızıntısı yok.
  * Soru üretimi rastgele-on-the-fly değil: değerlendirme için SABİT ızgara
    (her görüntü x o kaynakta etiketi tanımlı her hastalık).
  * Etiketi tanımlı olmayan (MIMIC'te hiç etiketlenmemiş VinDr hastalıkları) ve
    belirsiz (CheXpert -1) çiftler sorulmaz; eski kod bunları "negatif" sayıyordu.
  * MIMIC'in 'No Finding' ve 'Pneumonia' etiketleri de eşlenir.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

# CheXpert sütunu -> VinDr sütun adı
MIMIC_TO_VINDR = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
    "Lung Opacity": "Lung Opacity",
    "Pleural Effusion": "Pleural effusion",
    "Pneumothorax": "Pneumothorax",
    "Pneumonia": "Pneumonia",
    "No Finding": "No finding",
}

# "Is there no finding?" mantıksal olarak bir hastalık sorusu değildir.
DEFAULT_DROP_FROM_QUESTIONS = ("No finding",)


# --------------------------------------------------------------------------
# 1. Ana tablo
# --------------------------------------------------------------------------
def question_text(disease: str) -> str:
    """Eski kodla aynı şablon."""
    return f"Is there {disease.lower()}?"


def _dominant_class(row, label_cols):
    """Yalnızca TABAKALAMA için (VinDr). Eski kodla aynı mantık."""
    if row.get("No finding", 0) == 1:
        return "No finding"
    for c in label_cols:
        if c != "No finding" and row[c] == 1:
            return c
    return "Normal"


def build_master(root: str = ".", vindr_agg: str = "max", frontal_only: bool = False,
                 check_files: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """
    Döndürür: (df_all, label_cols)

    df_all sütunları:
      image_path, source, subject, study, view,  <label_cols...>
    Etiket değerleri: 1.0 / 0.0 / NaN  (NaN = bu görüntü için bu soru sorulamaz)

    vindr_agg : 'max'  -> 3 radyolog birleşimi (eski kodla aynı)
                'majority' -> en az 2 radyolog pozitif
    frontal_only : True ise MIMIC'te yalnızca PA/AP görüntüler.
    """
    j = lambda *p: os.path.join(root, *p)

    # ---------------- VinDr ----------------
    print("[data] VinDr okunuyor...")
    raw = pd.read_csv(j("vindr", "image_labels_train.csv"))
    label_cols = [c for c in raw.columns if c not in ("image_id", "rad_id")]
    if vindr_agg == "max":
        v = raw.groupby("image_id")[label_cols].max().reset_index()
    elif vindr_agg == "majority":
        v = (raw.groupby("image_id")[label_cols].sum() >= 2).astype(float).reset_index()
    else:
        raise ValueError(vindr_agg)

    def vpath(i):
        p = j("vindr", "images", "train", f"{i}.png")
        return p if os.path.exists(p) else j("vindr", "images", "test", f"{i}.png")

    v["image_path"] = v["image_id"].map(vpath)
    v["source"] = "vindr"
    v["subject"] = "vindr_" + v["image_id"].astype(str)  # VinDr'de hasta kimliği YOK
    v["study"] = v["subject"]
    v["view"] = "NA"
    v[label_cols] = v[label_cols].astype(float)

    # ---------------- MIMIC ----------------
    print("[data] MIMIC birleştiriliyor...")
    meta = pd.read_csv(j("mimic", "mimic-cxr-2.0.0-metadata.csv"))
    chex = pd.read_csv(j("mimic", "mimic-cxr-2.0.0-chexpert.csv"))
    spl = pd.read_csv(j("mimic", "mimic-cxr-2.0.0-split.csv"))
    m = meta.merge(chex, on=["subject_id", "study_id"]).merge(
        spl, on=["dicom_id", "subject_id", "study_id"])
    if frontal_only and "ViewPosition" in m.columns:
        m = m[m["ViewPosition"].isin(["PA", "AP"])]
    m = m.reset_index(drop=True)

    m["image_path"] = [
        j("mimic", "official_data_iccv_final", "files", f"p{str(s)[:2]}", f"p{s}", f"s{st}", f"{d}.jpg")
        for s, st, d in zip(m["subject_id"], m["study_id"], m["dicom_id"])
    ]
    if check_files:
        keep = np.fromiter((os.path.exists(p) for p in m["image_path"]), bool, len(m))
        print(f"[data] MIMIC dosyası bulunan: {keep.sum()} / {len(m)}")
        m = m[keep].reset_index(drop=True)

    mm = pd.DataFrame(np.nan, index=m.index, columns=label_cols)  # varsayılan: TANIMSIZ
    for m_col, v_col in MIMIC_TO_VINDR.items():
        if v_col not in label_cols or m_col not in m.columns:
            continue
        s = m[m_col]
        out = pd.Series(0.0, index=m.index)          # boş = "bahsedilmemiş" -> negatif (CheXpert U-zeros/blank)
        out[s == 1] = 1.0
        out[s == -1] = np.nan                         # belirsiz -> soru sorulmaz
        mm[v_col] = out
    mm["image_path"] = m["image_path"]
    mm["source"] = "mimic"
    mm["subject"] = "p" + m["subject_id"].astype(str)
    mm["study"] = "s" + m["study_id"].astype(str)
    mm["view"] = m["ViewPosition"].fillna("NA") if "ViewPosition" in m.columns else "NA"
    mm["official_split"] = m["split"]

    df = pd.concat([v, mm], ignore_index=True)
    df = df[["image_path", "source", "subject", "study", "view"] + label_cols + ["official_split"]]
    print(f"[data] toplam görüntü: {len(df)}  (VinDr {int((df.source=='vindr').sum())}, "
          f"MIMIC {int((df.source=='mimic').sum())})  | etiket sütunu: {len(label_cols)}")
    return df, label_cols


# --------------------------------------------------------------------------
# 2. Bölme
# --------------------------------------------------------------------------
def _group_holdout(df: pd.DataFrame, group_col: str, n_target: int, seed: int):
    """
    Grup (hasta) bütünlüğünü koruyarak ~n_target görüntü ayırır.
    Hastalar rastgele sıralanır, toplam >= n_target olana dek eklenir.
    Döndürür: (kalan, ayrılan)
    """
    rng = np.random.RandomState(seed)
    sizes = df.groupby(group_col).size()
    order = rng.permutation(sizes.index.values)
    cum = np.cumsum(sizes.loc[order].values)
    k = int(np.searchsorted(cum, n_target, side="left")) + 1
    held = set(order[:k])
    mask = df[group_col].isin(held)
    return df[~mask], df[mask]


def _strat_or_plain_split(df: pd.DataFrame, test_size, seed: int, label_cols):
    """VinDr: hasta kimliği olmadığından görüntü düzeyinde (tabakalı, mümkünse)."""
    cls = df.apply(lambda r: _dominant_class(r, label_cols), axis=1)
    vc = cls.value_counts()
    strat = cls if vc.min() >= 2 else None
    # tabakalı bölmede test_size >= sınıf sayısı olmalı
    if strat is not None and isinstance(test_size, int) and test_size < vc.size:
        strat = None
    a, b = train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)
    return a, b


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    label_cols: list[str]
    meta: dict = field(default_factory=dict)


def make_splits(df_all: pd.DataFrame, label_cols: list[str], seed: int = SEED,
                mimic_test: int = 7500, vindr_test: int = 2500, val_frac: float = 0.10) -> Splits:
    """
    MIMIC : subject (hasta) bazlı bölme  (test, sonra val)
    VinDr : görüntü düzeyi (hasta kimliği yok -> sınırlılık)
    """
    mim = df_all[df_all.source == "mimic"]
    vin = df_all[df_all.source == "vindr"]

    m_pool, m_test = _group_holdout(mim, "subject", mimic_test, seed)
    m_val_target = int(round(val_frac * len(m_pool)))
    m_train, m_val = _group_holdout(m_pool, "subject", m_val_target, seed + 1)

    v_pool, v_test = _strat_or_plain_split(vin, vindr_test, seed, label_cols)
    v_train, v_val = _strat_or_plain_split(v_pool, val_frac, seed + 1, label_cols)

    sp = Splits(
        train=pd.concat([m_train, v_train], ignore_index=True),
        val=pd.concat([m_val, v_val], ignore_index=True),
        test=pd.concat([m_test, v_test], ignore_index=True),
        label_cols=label_cols,
        meta=dict(seed=seed, mimic_test_target=mimic_test, vindr_test_target=vindr_test,
                  val_frac=val_frac, split_unit_mimic="subject_id",
                  split_unit_vindr="image (no patient id available)"),
    )
    assert_no_leak(sp)
    return sp


def assert_no_leak(sp: Splits) -> None:
    """MIMIC için subject VE study kesişimi boş olmalı; her iki kaynakta görüntü kesişimi boş olmalı."""
    parts = {"train": sp.train, "val": sp.val, "test": sp.test}
    names = list(parts)
    for i in range(len(names)):
        for k in range(i + 1, len(names)):
            a, b = parts[names[i]], parts[names[k]]
            assert not (set(a.image_path) & set(b.image_path)), f"görüntü kesişimi: {names[i]}-{names[k]}"
            am, bm = a[a.source == "mimic"], b[b.source == "mimic"]
            assert not (set(am.subject) & set(bm.subject)), f"MIMIC subject sızıntısı: {names[i]}-{names[k]}"
            assert not (set(am.study) & set(bm.study)), f"MIMIC study sızıntısı: {names[i]}-{names[k]}"


def leakage_report(sp: Splits) -> pd.DataFrame:
    rows = []
    for name, d in (("train", sp.train), ("val", sp.val), ("test", sp.test)):
        rows.append(dict(split=name, n_images=len(d),
                         n_mimic=int((d.source == "mimic").sum()),
                         n_vindr=int((d.source == "vindr").sum()),
                         n_mimic_subjects=d[d.source == "mimic"].subject.nunique()))
    return pd.DataFrame(rows)


def save_splits(sp: Splits, out_dir: str = "splits") -> None:
    os.makedirs(out_dir, exist_ok=True)
    for n in ("train", "val", "test"):
        getattr(sp, n).to_csv(os.path.join(out_dir, f"{n}.csv"), index=False)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(dict(sp.meta, label_cols=sp.label_cols), f, indent=2)
    print(f"[data] bölmeler kaydedildi -> {out_dir}/")


def load_splits(out_dir: str = "splits") -> Splits:
    with open(os.path.join(out_dir, "meta.json")) as f:
        meta = json.load(f)
    label_cols = meta.pop("label_cols")
    sp = Splits(*(pd.read_csv(os.path.join(out_dir, f"{n}.csv"), low_memory=False) for n in ("train", "val", "test")),
                label_cols=label_cols, meta=meta)
    assert_no_leak(sp)
    return sp


# --------------------------------------------------------------------------
# 3. Sorulabilir etiketler ve sabit ızgara
# --------------------------------------------------------------------------
def askable_cols(df: pd.DataFrame, label_cols: list[str],
                 drop=DEFAULT_DROP_FROM_QUESTIONS) -> dict[str, list[str]]:
    """Her kaynakta (en az bir) etiketi tanımlı sütunlar."""
    out = {}
    for src, d in df.groupby("source"):
        out[src] = [c for c in label_cols if c not in drop and d[c].notna().any()]
    return out


def build_eval_grid(df: pd.DataFrame, label_cols: list[str], drop=DEFAULT_DROP_FROM_QUESTIONS,
                    questions_per_image: int | None = None, seed: int = SEED) -> pd.DataFrame:
    """
    SABİT (deterministik) değerlendirme ızgarası: her görüntü x etiketi tanımlı her hastalık.
    questions_per_image verilirse görüntü başına bu kadar soru (seed ile sabit) örneklenir
    (doğrulama için hız amaçlı).

    Döndürür: sütunlar [img_idx, image_path, source, subject, disease, label]
    img_idx = df içindeki konum (0..len(df)-1); DataLoader bunu kullanır.
    """
    df = df.reset_index(drop=True)
    rng = np.random.RandomState(seed)
    rows = []
    cols_by_src = askable_cols(df, label_cols, drop)
    L = df[label_cols].values  # (N, C) float, NaN = tanımsız
    cidx = {c: i for i, c in enumerate(label_cols)}
    for i, (path, src, subj) in enumerate(zip(df.image_path, df.source, df.subject)):
        cands = [c for c in cols_by_src[src] if not np.isnan(L[i, cidx[c]])]
        if questions_per_image is not None and len(cands) > questions_per_image:
            cands = [cands[k] for k in sorted(rng.choice(len(cands), questions_per_image, replace=False))]
        for c in cands:
            rows.append((i, path, src, subj, c, float(L[i, cidx[c]])))
    g = pd.DataFrame(rows, columns=["img_idx", "image_path", "source", "subject", "disease", "label"])
    return g


def grid_summary(grid: pd.DataFrame) -> pd.DataFrame:
    s = grid.groupby(["disease", "source"])["label"].agg(n="size", n_pos="sum").reset_index()
    s["n_neg"] = s["n"] - s["n_pos"]
    return s.sort_values(["disease", "source"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 4. Eğitim için soru örnekleyici (torch'suz; Dataset içinde kullanılır)
# --------------------------------------------------------------------------
def sample_training_question(label_row: np.ndarray, label_cols: list[str], askable: list[str],
                             rng, p_pos: float = 0.5):
    """
    Bir görüntü için (disease, label) örnekler.
      * p_pos olasılıkla: görüntünün pozitif etiketlerinden biri (varsa) -> label=1
      * aksi halde: sorulabilir etiketlerden rastgele biri -> label = etiket (0/1)
    NaN etiketli çiftler hiç seçilmez. Çelişkili "Is there no finding?" sorusu yoktur.
    """
    cidx = {c: i for i, c in enumerate(label_cols)}
    valid = [c for c in askable if not np.isnan(label_row[cidx[c]])]
    pos = [c for c in valid if label_row[cidx[c]] == 1]
    if pos and rng.random() < p_pos:
        d = pos[int(rng.random() * len(pos))]
    else:
        d = valid[int(rng.random() * len(valid))]
    return d, float(label_row[cidx[d]])
