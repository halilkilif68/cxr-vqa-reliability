"""
cxr_models.py  --  BLIP / ViLT / CLIP modelleri, veri kümeleri ve eğitim döngüsü.

Tek kaynak ilkesi: eğitimde ve her analizde modeller AYNI `build_model` ile kurulur,
ağırlıklar AYNI `load_checkpoint` ile (strict=True) yüklenir. Böylece "yanlış
mimariye sessizce yüklenen" (strict=False) hata sınıfı ortadan kalkar.

Mimari/LoRA ayarları eski notebook'larla aynıdır (r=16, alpha=32, dropout=0.05):
  BLIP : tüm BlipForQuestionAnswering sarılır, hedef ["query","value"]
         (NOT: BLIP görsel kodlayıcısı birleşik 'qkv' kullandığından bu isimler yalnızca
          metin kodlayıcı/dekoder dikkat katmanlarında eşleşir; param_report bunu gösterir)
  ViLT : ViltModel sarılır, hedef ["query","key","value"]; sınıflandırma başlığı
         LoRA dışındadır ve EĞİTİLİR (eski kodda başlık PEFT içinde donuk kalıyordu)
  CLIP : CLIPModel sarılır, hedef ["q_proj","v_proj"]; Linear(512,1) başlığı eğitilir
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

import cxr_data as D

MODEL_NAMES = ("blip", "vilt", "clip")

# --------------------------------------------------------------------------
# Yapılandırma
# --------------------------------------------------------------------------
@dataclass
class ModelCfg:
    hf_name: str
    img_size: int
    max_len: int
    batch_size: int
    lr: float = 5e-5
    weight_decay: float = 0.02
    epochs: int = 10
    head_lr_mult: float = 1.0      # sınıflandırma başlığı için öğrenme oranı çarpanı (1.0 = eski davranış)


CFG = {
    "blip": ModelCfg("Salesforce/blip-vqa-base", 384, 32, 16, weight_decay=0.02),
    "vilt": ModelCfg("dandelin/vilt-b32-finetuned-vqa", 384, 40, 16, weight_decay=0.01),
    "clip": ModelCfg("openai/clip-vit-base-patch32", 224, 32, 64, weight_decay=0.02,
                      head_lr_mult=100.0),   # başlık girdisi birim-normlu embedding çarpımı (çok küçük) -> başlık 100x lr
}
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05, bias="none")


def seed_everything(seed: int = D.SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# Modeller
# --------------------------------------------------------------------------
# BLIP LoRA kapsamı:
#   "vision_text" : görsel kodlayıcı (birleşik qkv) + metin kodlayıcı (query/value)   [önerilen]
#   "text_only"   : yalnızca metin kodlayıcı query/value
#   "legacy"      : ["query","value"] -> metin kodlayıcı + KULLANILMAYAN text_decoder (eski davranış)
BLIP_LORA_MODE = "vision_text"


def _blip_targets(mode: str):
    if mode == "legacy":
        return ["query", "value"]
    if mode == "text_only":
        return r".*text_encoder\..*\.(query|value)"
    if mode == "vision_text":
        return r".*(text_encoder\..*\.(query|value)|vision_model\..*\.qkv)"
    raise ValueError(mode)


class RubinBLIP(nn.Module):
    def __init__(self, hf_name: str):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import BlipForQuestionAnswering
        base = BlipForQuestionAnswering.from_pretrained(hf_name)
        self.blip = get_peft_model(base, LoraConfig(target_modules=_blip_targets(BLIP_LORA_MODE), **LORA))
        core = self.blip.get_base_model()
        self.vision_model = core.vision_model
        self.text_encoder = core.text_encoder
        self.classifier = nn.Linear(768, 1)

    def forward(self, pixel_values, input_ids, attention_mask, **_):
        img = self.vision_model(pixel_values=pixel_values).last_hidden_state
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask,
                                encoder_hidden_states=img, return_dict=True)
        return self.classifier(out.last_hidden_state[:, 0, :]).squeeze(-1)


class RubinViLT(nn.Module):
    def __init__(self, hf_name: str):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import ViltModel
        try:
            base = ViltModel.from_pretrained(hf_name, attn_implementation="eager")
        except TypeError:
            base = ViltModel.from_pretrained(hf_name)
        self.backbone = get_peft_model(base, LoraConfig(target_modules=["query", "key", "value"], **LORA))
        self.head = nn.Sequential(nn.Linear(768, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 1))

    def forward(self, pixel_values, input_ids, attention_mask, token_type_ids=None,
                output_attentions=False, **_):
        out = self.backbone(pixel_values=pixel_values, input_ids=input_ids,
                            attention_mask=attention_mask, token_type_ids=token_type_ids,
                            output_attentions=output_attentions, return_dict=True)
        logits = self.head(out.pooler_output).squeeze(-1)
        return (logits, out.attentions) if output_attentions else logits


class RubinCLIP(nn.Module):
    def __init__(self, hf_name: str):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import CLIPModel
        base = CLIPModel.from_pretrained(hf_name)
        self.clip = get_peft_model(base, LoraConfig(target_modules=["q_proj", "v_proj"], **LORA))
        self.classifier = nn.Linear(512, 1)

    def forward(self, pixel_values, input_ids, attention_mask, **_):
        o = self.clip(pixel_values=pixel_values, input_ids=input_ids,
                      attention_mask=attention_mask, return_dict=True)
        return self.classifier(o.image_embeds * o.text_embeds).squeeze(-1)


_CLASSES = {"blip": RubinBLIP, "vilt": RubinViLT, "clip": RubinCLIP}


def build_model(name: str, device=None):
    """(model, processor) döndürür. Eğitim ve tüm analizler bunu kullanır."""
    from transformers import BlipProcessor, CLIPProcessor, ViltProcessor
    cfg = CFG[name]
    model = _CLASSES[name](cfg.hf_name)
    proc = {"blip": BlipProcessor, "vilt": ViltProcessor, "clip": CLIPProcessor}[name].from_pretrained(cfg.hf_name)
    return model.to(device or get_device()), proc


def param_report(model: nn.Module) -> pd.DataFrame:
    """Eğitilen parametrelerin üst-modül bazında dökümü (makaledeki 'LoRA' ifadesini doğrulamak için)."""
    rows = {}
    for n, p in model.named_parameters():
        top = n.split(".")[0]
        sub = "lora" if "lora_" in n else "other"
        r = rows.setdefault((top, sub), dict(module=top, kind=sub, n_params=0, n_trainable=0))
        r["n_params"] += p.numel()
        r["n_trainable"] += p.numel() if p.requires_grad else 0
    df = pd.DataFrame(rows.values())
    tot = df.n_trainable.sum()
    print(f"[model] eğitilebilir parametre: {tot:,} / {df.n_params.sum():,} ({100*tot/df.n_params.sum():.2f}%)")
    return df


def save_checkpoint(model, path, **extra):
    torch.save({"model": model.state_dict(), **extra}, path)


def load_checkpoint(model, path, device=None) -> dict:
    """strict=True: eksik/fazla anahtar varsa HATA verir (sessiz yüklememe yok)."""
    ck = torch.load(path, map_location=device or get_device())
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    res = model.load_state_dict(sd, strict=True)
    print(f"[ckpt] yüklendi: {path}  (missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)})")
    return ck if isinstance(ck, dict) else {}


# --------------------------------------------------------------------------
# Girdi kodlama
# --------------------------------------------------------------------------
class Encoder:
    """Görüntü ve soru kodlaması. Görüntü HER ZAMAN modelin kare giriş boyutuna yeniden
    ölçeklenir (BLIP/ViLT 384, CLIP 224) -> Grad-CAM/IoU koordinatları tam görüntüye karşılık gelir
    (CLIP'in eski 'merkez kırpma'sı yoktur)."""

    def __init__(self, name: str, processor):
        self.name, self.proc, self.cfg = name, processor, CFG[name]
        self.tok = processor.tokenizer
        self.imgproc = processor.image_processor
        self._qcache: dict[str, dict] = {}
        self.mean = torch.tensor(self.imgproc.image_mean).view(1, 3, 1, 1)
        self.std = torch.tensor(self.imgproc.image_std).view(1, 3, 1, 1)

    def image(self, path: str) -> torch.Tensor:
        s = self.cfg.img_size
        im = Image.open(path).convert("RGB").resize((s, s), Image.BICUBIC)
        return self.imgproc(im, return_tensors="pt")["pixel_values"][0]

    def question(self, disease: str) -> dict:
        if disease not in self._qcache:
            t = self.tok(D.question_text(disease), padding="max_length", truncation=True,
                         max_length=self.cfg.max_len, return_tensors="pt")
            d = {"input_ids": t["input_ids"][0], "attention_mask": t["attention_mask"][0]}
            if self.name == "vilt":
                d["token_type_ids"] = t["token_type_ids"][0]
            self._qcache[disease] = d
        return self._qcache[disease]

    def black_value(self, like: torch.Tensor) -> torch.Tensor:
        """Gerçek siyah görüntü (ham piksel 0) -> normalize uzayında (0-mean)/std."""
        b = (0.0 - self.mean) / self.std
        return b.to(like.device, like.dtype).expand_as(like)


# --------------------------------------------------------------------------
# Veri kümeleri
# --------------------------------------------------------------------------
class TrainDataset(Dataset):
    """Her erişimde (seed, epoch, idx)'e bağlı DETERMINISTIK bir soru örnekler."""

    def __init__(self, df: pd.DataFrame, label_cols, encoder: Encoder, seed=D.SEED, p_pos=0.5):
        df = df.reset_index(drop=True)
        self.label_cols, self.enc, self.seed, self.p_pos = list(label_cols), encoder, seed, p_pos
        self.askable = D.askable_cols(df, self.label_cols)
        # sorulabilir hiçbir etiketi olmayan görüntüleri (ör. tüm etiketleri belirsiz) çıkar
        L_all = df[self.label_cols].values.astype(float)
        cidx = {c: i for i, c in enumerate(self.label_cols)}
        ok = np.zeros(len(df), bool)
        for src, cols in self.askable.items():
            m = (df.source.values == src)
            ok[m] = ~np.isnan(L_all[m][:, [cidx[c] for c in cols]]).all(axis=1)
        if (~ok).any():
            print(f"[data] soru üretilemeyen {int((~ok).sum())} görüntü eğitimden çıkarıldı")
        self.df = df[ok].reset_index(drop=True)
        self.L = self.df[self.label_cols].values.astype(float)
        self.src = self.df.source.values
        self.paths = self.df.image_path.values
        self.epoch = 0

    def set_epoch(self, e: int):
        self.epoch = e

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        rng = random.Random(self.seed * 1_000_003 + self.epoch * 10_007 + int(i))
        disease, label = D.sample_training_question(self.L[i], self.label_cols,
                                                    self.askable[self.src[i]], rng, self.p_pos)
        item = {"pixel_values": self.enc.image(self.paths[i]), **self.enc.question(disease),
                "label": torch.tensor(label, dtype=torch.float32)}
        return item


def _collate(batch):
    keys = [k for k in batch[0]]
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


class GridDataset(Dataset):
    """
    Görüntü başına bir öğe: görüntü tensörü + o görüntünün tüm (sabit ızgara) soruları.
    Böylece her görüntü diskten YALNIZCA BİR KEZ okunur (ızgara ~28 soru/görüntü).
    """

    def __init__(self, grid: pd.DataFrame, encoder: Encoder):
        self.enc = encoder
        self.g = grid.sort_values(["img_idx"], kind="stable").reset_index(drop=True)
        self.img_ids = self.g.img_idx.unique()
        self.rows = {k: np.asarray(v) for k, v in self.g.groupby("img_idx").indices.items()}
        self.paths = self.g.groupby("img_idx").image_path.first()

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, k):
        img_idx = self.img_ids[k]
        rows = self.rows[img_idx]
        sub = self.g.iloc[rows]
        qs = [self.enc.question(d) for d in sub.disease]
        out = {"pixel_values": self.enc.image(self.paths.loc[img_idx]),
               "row_ids": torch.as_tensor(rows, dtype=torch.long)}
        for key in qs[0]:
            out[key] = torch.stack([q[key] for q in qs])
        return out


def grid_collate(batch):
    """Görüntüleri yığınlar, soruları düzleştirir; rep[j] = j. sorunun ait olduğu görüntünün batch indeksi."""
    pv = torch.stack([b["pixel_values"] for b in batch])
    counts = [len(b["row_ids"]) for b in batch]
    rep = torch.repeat_interleave(torch.arange(len(batch)), torch.tensor(counts))
    out = {"pixel_values": pv, "rep": rep, "row_ids": torch.cat([b["row_ids"] for b in batch])}
    for key in batch[0]:
        if key in ("pixel_values", "row_ids"):
            continue
        out[key] = torch.cat([b[key] for b in batch])
    return out


def make_loader(ds, batch_size, shuffle, num_workers=4, collate=None):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate or _collate, pin_memory=torch.cuda.is_available(),
                      persistent_workers=False)


def model_inputs(batch, device, pixel_values=None):
    """Modelin forward'ının beklediği argümanlar."""
    kw = {k: batch[k].to(device) for k in ("input_ids", "attention_mask", "token_type_ids") if k in batch}
    kw["pixel_values"] = (batch["pixel_values"] if pixel_values is None else pixel_values).to(device)
    return kw


# --------------------------------------------------------------------------
# Değerlendirme yardımcıları (eğitim içi doğrulama için)
# --------------------------------------------------------------------------
ROW_CHUNK = {"blip": 32, "vilt": 64, "clip": 128}   # tek ileri geçişteki (görüntü,soru) satırı; GPU belleğini sınırlar


@torch.no_grad()
def forward_rows(model, b, device, x_img, name: str, amp=True, chunk=None) -> np.ndarray:
    """
    Bir batch'teki tüm (görüntü,soru) satırları için olasılık. x_img: (B_img,C,H,W) görüntü tensörü;
    satırlar b['rep'] ile görüntülere eşlenir. Bellek için satırlar `chunk` parçalarında işlenir
    (eski sürüm 8 görüntü x ~28 soru = ~224 kopyayı tek seferde veriyordu -> BLIP'te CUDA OOM).
    """
    chunk = chunk or ROW_CHUNK[name]
    rep = b["rep"].to(device)
    out = []
    for s in range(0, len(rep), chunk):
        sl = slice(s, s + chunk)
        kw = {k: b[k][sl].to(device) for k in ("input_ids", "attention_mask", "token_type_ids") if k in b}
        kw["pixel_values"] = x_img[rep[sl]]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            logit = model(**kw)
        out.append(torch.sigmoid(logit.float()).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def predict_grid(model, grid: pd.DataFrame, enc: Encoder, device, batch_images=4,
                 num_workers=4, amp=True) -> pd.DataFrame:
    """Izgaradaki her (görüntü, hastalık) çifti için olasılık; grid satır sırasında (sıralı) döner."""
    ds = GridDataset(grid, enc)
    dl = make_loader(ds, batch_images, False, num_workers, grid_collate)
    probs = np.full(len(ds.g), np.nan, dtype=np.float32)
    model.eval()
    for b in dl:
        probs[b["row_ids"].numpy()] = forward_rows(model, b, device, b["pixel_values"].to(device), enc.name, amp)
    out = ds.g.copy()
    out["prob"] = probs
    return out


def _safe_auc(y, p):
    y = np.asarray(y)
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan


def summarize_predictions(df: pd.DataFrame, prob_col="prob", min_pos=10, min_neg=10) -> dict:
    micro = _safe_auc(df.label, df[prob_col])
    per = []
    for d, s in df.groupby("disease"):
        npos, nneg = int(s.label.sum()), int(len(s) - s.label.sum())
        if npos >= min_pos and nneg >= min_neg:
            per.append(_safe_auc(s.label, s[prob_col]))
    out = dict(micro_auc=micro, macro_auc=float(np.nanmean(per)) if per else np.nan, n_diseases=len(per))
    for src, s in df.groupby("source"):
        out[f"auc_{src}"] = _safe_auc(s.label, s[prob_col])
    return out


# --------------------------------------------------------------------------
# Eğitim
# --------------------------------------------------------------------------
def train_model(name: str, splits: D.Splits, out_dir="checkpoints", device=None, amp=True,
                val_questions_per_image=4, num_workers=4, max_train_images=None,
                max_val_images=None, epochs=None, resume_best=False):
    """
    Eğitir; her epoch sonunda SABİT doğrulama ızgarasında macro AUC ölçer;
    `<name>_best.pt` (en iyi doğrulama macro AUC) ve `<name>_last.pt` kaydeder.
    Test verisi bu fonksiyonda HİÇ kullanılmaz.
    """
    cfg = CFG[name]
    epochs = epochs or cfg.epochs
    device = device or get_device()
    os.makedirs(out_dir, exist_ok=True)
    seed_everything()

    model, proc = build_model(name, device)
    print(param_report(model).to_string(index=False))
    enc = Encoder(name, proc)

    tr_df = splits.train if max_train_images is None else splits.train.sample(max_train_images, random_state=D.SEED)
    va_df = splits.val if max_val_images is None else splits.val.sample(max_val_images, random_state=D.SEED)
    train_ds = TrainDataset(tr_df, splits.label_cols, enc)
    val_grid = D.build_eval_grid(va_df, splits.label_cols, questions_per_image=val_questions_per_image)
    print(f"[train] train görüntü: {len(train_ds)} | doğrulama çifti: {len(val_grid)}")

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    head = [p for n, p in named if n.startswith(("classifier.", "head."))]
    rest = [p for n, p in named if not n.startswith(("classifier.", "head."))]
    groups = [{"params": rest, "lr": cfg.lr}]
    if head:
        groups.append({"params": head, "lr": cfg.lr * cfg.head_lr_mult})
    print(f"[train] lr={cfg.lr} | başlık lr çarpanı={cfg.head_lr_mult} | LoRA/diğer param tensörü={len(rest)}, başlık={len(head)}")
    opt = torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.BCEWithLogitsLoss()
    log, best = [], -1.0

    for ep in range(epochs):
        train_ds.set_epoch(ep)
        dl = make_loader(train_ds, cfg.batch_size, True, num_workers)
        model.train()
        tot, nb, t0 = 0.0, 0, time.time()
        for b in dl:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                logit = model(**model_inputs(b, device))
            loss = crit(logit.float(), b["label"].to(device))
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
            if nb % 200 == 0:
                print(f"  ep{ep+1} it{nb}/{len(dl)} loss={tot/nb:.4f}")
        sched.step()

        vp = predict_grid(model, val_grid, enc, device, num_workers=num_workers, amp=amp)
        vl = float(nn.functional.binary_cross_entropy(torch.tensor(vp.prob.values).clamp(1e-6, 1 - 1e-6),
                                                      torch.tensor(vp.label.values, dtype=torch.float32)))
        m = summarize_predictions(vp)
        rec = dict(epoch=ep + 1, train_loss=tot / max(nb, 1), val_loss=vl, minutes=(time.time() - t0) / 60, **m)
        log.append(rec)
        print("[epoch]", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()})
        pd.DataFrame(log).to_csv(os.path.join(out_dir, f"{name}_train_log.csv"), index=False)
        save_checkpoint(model, os.path.join(out_dir, f"{name}_last.pt"), epoch=ep + 1, metrics=rec)
        score = m["macro_auc"] if not np.isnan(m["macro_auc"]) else m["micro_auc"]
        if np.isnan(score):
            score = 0.0                      # (yalnızca duman testinde olabilir) yine de bir 'best' üret
        if score > best:
            best = score
            save_checkpoint(model, os.path.join(out_dir, f"{name}_best.pt"), epoch=ep + 1, metrics=rec)
            print(f"  ★ yeni en iyi (val macro AUC={best:.4f}) -> {name}_best.pt")
    return pd.DataFrame(log)
