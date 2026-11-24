# Beyond Accuracy: A Reliability Evaluation Framework for Template-Based Chest X-ray Question Answering

Code, split identifiers and result tables for the paper *Beyond Accuracy: A Reliability Evaluation Framework
for Template-Based Chest X-ray Question Answering* (BLIP, ViLT and CLIP fine-tuned with LoRA on MIMIC-CXR and VinDr-CXR).

The repository contains no images, no labels and no trained weights. Both datasets are credentialed
(PhysioNet) and cannot be redistributed; see *Data* below.

## Contents

```
src/cxr_data.py        label mapping, subject-level split, fixed evaluation grid, training question sampler
src/cxr_models.py      BLIP / ViLT / CLIP wrappers, LoRA setup, datasets, training, strict checkpoint loading
src/cxr_eval.py        image perturbations (noise, black, patch shuffle), AUC, cluster bootstrap, Grad-CAM / attention, IoU
00_data_split.ipynb    builds the split (MIMIC-CXR by subject_id, VinDr-CXR by image)
01_blip.ipynb          train + test + analyses for BLIP
02_vilt.ipynb          same for ViLT
03_clip.ipynb          same for CLIP
04_compare_models.ipynb  paired comparison of the three models
analysis/paper_tables.py   Table 4 (threshold metrics), Table 7 (label-signed shift), per-disease AUC table
analysis/make_figures.py   Figures 3-5
splits/                image identifiers per partition (no labels), meta.json
results/               summary tables and figures produced by the notebooks
```

Comments inside the notebooks and source files are in Turkish.

## Data

1. MIMIC-CXR-JPG 2.0.0 (credentialed access): https://physionet.org/content/mimic-cxr-jpg/ and the report-derived
   metadata, CheXpert labels and official split files.
2. VinDr-CXR 1.0.0 (credentialed access): https://physionet.org/content/vindr-cxr/ (training-set release,
   15,000 images, annotated independently by three radiologists each).

Place them next to the code as `mimic/` and `vindr/` (the notebooks use `DATA_ROOT = "."`). `cxr_data.py` expects
`mimic/mimic-cxr-2.0.0-{metadata,chexpert,split}.csv` and the image files under
`mimic/official_data_iccv_final/files/`, and `vindr/image_labels_train.csv`, `vindr/annotations_train.csv` and the
PNG images under `vindr/images/`. Set `DICOM_DIR` in notebooks 01-03 to the folder with the original VinDr-CXR DICOM
files (needed only for the IoU analysis, which uses the original image size).

The identifiers of the split used in the paper are in `splits/*_ids.csv` (82,357 / 9,188 / 10,014 images).
Running `00_data_split.ipynb` with the same seed (42) regenerates the labelled split files.

## Running

1. `00_data_split.ipynb` -> `splits/{train,val,test}.csv` (labelled; stays local) and a leakage check
   (zero subject and study overlap for MIMIC-CXR).
2. `01_blip.ipynb`, `02_vilt.ipynb`, `03_clip.ipynb`. Set `SMOKE = True` first for a quick end-to-end test.
   Training uses AdamW, cosine annealing over 10 epochs, bf16 autocast and seed 42; the checkpoint with the
   highest validation macro AUC is kept. The notebooks write `checkpoints/` (not tracked) and `results/`.
3. `04_compare_models.ipynb`.
4. `python analysis/paper_tables.py` then `python analysis/make_figures.py` (these read the per-pair test
   predictions that notebooks 01-03 write to `results/*_test_predictions.csv.gz`).

Software used for the paper: PyTorch 2.9.1, Transformers 5.23.0, PEFT 0.21.3 (see `requirements.txt`).
Bootstrap intervals use 500 cluster resamples; the last digit of an interval can differ between seeds.

## Where the paper's numbers come from

| Paper | File |
|---|---|
| Table 2 (split) | `splits/meta.json`, `results/label_counts_by_split.csv`, `splits/*_ids.csv` |
| Table 4 (performance) | micro/macro AUC: `results/compare_headline.csv`, `results/compare_auc_bootstrap.csv`; threshold metrics: `results/threshold_metrics.csv` |
| Table 5, Figure 5 (per disease) | `results/*_per_disease_full.csv`, `results/per_disease_merged.csv` |
| Table 6 (by source) | `results/*_headline.csv` (`auc_mimic`, `auc_vindr`, Text prior row) |
| Table 7 (label-signed shift) | `results/signed_dP_ci.csv`, `results/*_visual_dependence.csv` (Wilcoxon, dz) |
| Table 8 (patch shuffling) | `results/compare_shuffle_bootstrap.csv` |
| Table 9, Figure 4 (black image) | `results/black_image_summary.csv` (mean P(black), range across diseases, AUC with CI), `results/black_image_prob_by_disease.csv`, `results/*_bootstrap_micro_auc.csv`; figure: `analysis/make_figures.py` |
| Table 10 (IoU, pointing hit) | `results/*_iou_summary.csv`, `results/*_iou_detail.csv` |
| Table 11 (Full / Black / Prior) | `results/compare_auc_bootstrap.csv`, `results/*_bootstrap_micro_auc.csv` |
| Table 12 (Wilcoxon) | `results/compare_abs_error_wilcoxon.csv` |
| Figure 3 (confusion matrices) | `analysis/make_figures.py` |
| Figures 6-8 (maps) | `results/{blip,vilt,clip}_cam_examples.png` |

## Not included

Trained weights (`checkpoints/`) and the per-pair test predictions are not tracked: the predictions carry the
reference labels of the test images, which the data use agreements do not allow us to redistribute. Both can be
regenerated with the notebooks.

## License

MIT, see `LICENSE`. The datasets keep their own licenses.
