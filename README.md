# Final Project — GastroVision GI Endoscopy Classification

**Course:** AIN501 · Artificial Intelligence (Deep Learning) · MSE FSB
**Team:** 3 members · **Goal:** beat a published baseline on GastroVision

---

## 1. What we're building

Multi-class classification of **gastrointestinal (GI) endoscopy images** on the
**GastroVision** dataset (~8,000 images, 27 classes / **22 used in experiments**).
Classes span anatomical landmarks, pathological findings (polyps, ulcers, cancer,
esophagitis...), normal findings, and therapeutic interventions.

The core challenge is a **heavy long-tail distribution**: some diseases have fewer
than 10 images while common classes have hundreds. This is why the published
**macro-F1 baseline is only 0.6504** — and why most of the improvement comes from the
**data** (imbalance handling), not just the model.

### Goal
Raise **macro-F1 from 0.6504 → ≥ 0.75** on the **original stratified 60:20:20 split**,
reported as **mean ± std over ≥ 3 seeds**, with a **per-class F1 table** and confusion matrix.

### Published baseline (paper Table 2 — arXiv 2307.08140)

> ✅ Verified against the paper (Table 2). Still **reproduce DenseNet-121 yourself** on the same
> split so the comparison is fair and honest (and to rule out an under-trained baseline).

| Model | Macro-F1 | Micro-F1 | MCC |
|---|---|---|---|
| **DenseNet-121 (pretrained)** | **0.6504** | 0.8203 | 0.7987 |
| ResNet-50 (pretrained) | 0.6176 | 0.8146 | 0.7921 |
| DenseNet-169 | 0.4883 | 0.7055 | 0.6685 |
| EfficientNet-B0 | 0.4519 | 0.6759 | 0.6351 |
| ResNet-50 (from scratch) | 0.4330 | 0.6816 | 0.6416 |

**Note:** the paper baseline has **6 CNNs and zero Transformers** — adding Swin-T / ViT-S / a
hybrid (CoAtNet) into this exact table gives a clean **CNN vs Transformer vs Hybrid** comparison,
almost for free since the CNN rows already exist.

### Links
- Dataset: [GitHub](https://github.com/DebeshJha/GastroVision) · [OSF](https://osf.io/84e7f/) · paper arXiv 2307.08140
- Original split: **stratified 60:20:20** — reproduce with a **fixed seed** (no ad-hoc re-splitting → avoids leakage).

### Baseline sources & how to reproduce each (proof the baselines are valid)

Be explicit about **two provenance categories** when presenting — it's what makes the comparison defensible:

- **Reference baseline (has a published number to match):** taken from the GastroVision paper, Table 2.
  Reproducing it on our split *proves our pipeline is correct* before we compare anything.
- **New baselines (no published number — our contribution):** the Transformer and Hybrid are **not** in
  the paper. Here "reproduce" means *train under the identical protocol*, not match a target number.

| Baseline | Family | Architecture source | Pretrained weights | Published macro-F1 to match |
|---|---|---|---|---|
| **DenseNet-121** | CNN | Huang et al., CVPR 2017 (arXiv:1608.06993) — the model used by the GastroVision paper | torchvision `DenseNet121_Weights.IMAGENET1K_V1` | **0.6504** (paper Table 2) |
| **Swin-T** | Transformer | Liu et al., ICCV 2021 (arXiv:2103.14030) | `timm` · `swin_tiny_patch4_window7_224` (ImageNet-1k) | — (new; no paper number) |
| **ViT-S** *(alt.)* | Transformer | Dosovitskiy et al., ICLR 2021 (arXiv:2010.11929) | `timm` · `vit_small_patch16_224` (ImageNet-21k→1k) | — (new) |
| **CoAtNet-0** | Hybrid | Dai et al., NeurIPS 2021 (arXiv:2106.04803) | `timm` · `coatnet_0_rw_224` (ImageNet-1k) | — (new) |

Weights come from [`timm`](https://github.com/huggingface/pytorch-image-models) (Wightman, *PyTorch Image Models*) and torchvision — pretrained on ImageNet, then we replace the classifier head with a fresh `NUM_CLASSES` linear layer.

**Identical reproduce protocol for all four** (this equal footing is what makes the comparison valid):
- **Same data:** 22 classes (paper rule *">25 samples"*), stratified **60:20:20**, `SPLIT_SEED=42` — fixed for everyone.
- **Input** 224×224, ImageNet mean/std normalization; pretrained backbone + fresh linear head.
- **AdamW** (lr `1e-4`, weight-decay `1e-4`), **30 epochs**, batch 32, AMP on GPU, keep the **best-val-macro-F1** checkpoint.
- **≥ 3 seeds → report mean ± std**; primary metric **macro-F1** (+ per-class F1 + confusion matrix).
- One driver for all architectures: `run_seeds(build_fn, tag=...)` in the shared notebook.

> **Validity check already passed:** our reproduced **DenseNet-121 test macro-F1 = 0.676** (1 seed) ≥ paper
> **0.6504** — confirming the split + eval harness are correct. (Run all 3 seeds for the final `mean ± std`.)

---

## 2. What the report should cover

| Part | What to do |
|---|---|
| Problem & baseline | Quote the paper table; state split / metric / protocol; report your reproduced baseline |
| Data analysis (EDA) | Class distribution, image quality, artifacts, suspicious labels, **leakage check** (with figures) |
| **Data processing** | Normalize, domain-justified augmentation, class-balancing — **each step backed by an ablation** |
| Labels & validation | Long-tail handling, confusion matrix, per-class F1, typical error examples |
| Architecture | CNN vs Transformer vs Hybrid, same protocol / same seed; argue why local vs global matters here |
| Transfer learning | linear probe / progressive unfreezing / layer-wise LR decay / full fine-tune — comparison table |
| Deployment | Export ONNX, measure latency + model size, Gradio/Streamlit demo |

### Method rules (keep results honest & comparable)
1. **Use the author's original split.** If you must re-split, publish the seed and explain.
2. Medical data → split at the level the authors defined; state that leakage was checked.
3. Imbalanced data → always report **macro-F1**, never accuracy or weighted-F1 alone.
4. **≥ 3 seeds, report mean ± std.** A single number beating baseline by 0.5 pt is meaningless.
5. **Reproduce the strongest baseline before claiming you beat it.** If you can't, say so plainly.

---

## 3. Team split (3 people)

The **Data** work is done **together** (one shared recipe so all numbers are consistent); the
**Model** work is where the **one-baseline-per-person** split lives.

Everyone builds on the **shared baseline notebook** (`notebooks/00_baseline_gastrovision.ipynb`)
so all measurements use the same split, seeds, and eval harness.

### Shared by the whole team — Data
Agree on **one** data recipe so every architecture is compared on identical data:
- EDA (class distribution, image quality, artifacts, suspicious labels, **leakage check**)
- GI-domain augmentation pipeline (justified per step, with an **ablation table**)
- Class-imbalance handling: class weights / LDAM / Balanced-Softmax / decoupled (cRT) / repeat-factor sampling

### One baseline per person — Model
Each member **owns one architecture family** (this gives the CNN vs Transformer vs Hybrid
comparison). Each: reproduce/train their baseline on the fixed split via `run_seeds(build_fn, ...)`,
apply the shared data recipe, report macro-F1 mean ± std.

| Member | Baseline (architecture) | Model to build |
|---|---|---|
| **Member A** | **CNN** | DenseNet-121 (the published baseline, macro-F1 0.6504 — reproduce first) |
| **Member B** | **Transformer** | Swin-T or ViT-S (`timm`) |
| **Member C** | **Hybrid** | CoAtNet (`timm`) |

### Shared responsibilities (split however you like)
- **Eval harness** (macro-F1, per-class F1, confusion matrix, multi-seed runner) — already in the notebook.
- **Transfer-learning study** (linear probe / progressive unfreezing / layer-wise LR / full fine-tune).
- **Deployment**: ONNX export + latency/size + Gradio demo.

**Final step:** combine → pick the best `(architecture × data recipe)` as the proposed model and
compare it against the three reproduced baselines.

---

## 4. Getting started (Google Colab)

The base notebook `notebooks/00_baseline_gastrovision.ipynb` is written **Colab-first (T4)**.
Compute per run is light: **~15–25 min @ 224px**.

**Each session:**
1. Colab → **Runtime → Change runtime type → T4 GPU**.
2. Open the notebook and **Run all**. On the first run it auto-downloads `Gastrovision.zip`
   (~1.8 GB) via `gdown`, unzips it, and discovers the class folders by a recursive walk.
3. The integrity cell (2.2) confirms all ~8,000 images are present; the split cell is shared —
   don't change it without telling the team.
4. Reproduce the DenseNet-121 baseline → confirm you land near macro-F1 0.65.
5. Each member adds their architecture (section 7) and reuses the shared eval harness.

> **Local run** works too — the notebook auto-detects the environment. `pip install -r requirements.txt`,
> place the dataset under `data/gastrovision/`, and it reads from there / writes checkpoints to `checkpoints/`.

### Auto-detection (baked into the notebook)
- **Compute device** (GPU/CPU) detected automatically; AMP + `pin_memory` enable only on GPU.
- **Environment** (Colab vs local) detected → data/checkpoint/output paths switch automatically.
- **Checkpoints** save after each seed → to Drive on Colab, to `checkpoints/` locally (survive disconnects).
- `timm` is pip-installed in the first cell (not preinstalled on Colab).
- **Dataset auto-downloads** on Colab: pulls the **official single zip** `Gastrovision.zip` (~1.8 GB)
  by file id, unzips, and discovers class folders by a **recursive walk** (`scan_class_folders`) — handles
  GastroVision's nested Upper-GI / Lower-GI category folders, no manual paths.
- **Integrity check (cell 2.2)** prints total images + per-class counts + a corrupt-file spot-check,
  then **hard-fails if < ~8,000 images are present**. Alternative source if the Drive link breaks: [OSF](https://osf.io/84e7f/).
- Classes are filtered by the **paper's rule** — *"classes with more than 25 samples"* (arXiv 2307.08140) → 22 classes — with an `assert NUM_CLASSES == 22` safety check.

---

## 5. Repo layout

```
final-project/
├── README.md                          # this file
├── .gitignore                         # excludes data/, checkpoints/, outputs/
├── requirements.txt
├── notebooks/
│   └── 00_baseline_gastrovision.ipynb # shared base: setup → data → split → baseline → eval harness
├── data/          # (gitignored) dataset downloads
├── checkpoints/   # (gitignored) model weights
└── outputs/       # (gitignored) figures, logs, ONNX exports
```

> This folder is a **standalone repo** — it is intentionally git-ignored by the parent
> `machine-learning` repo and pushed to its own remote. Do not commit `data/`,
> `checkpoints/`, or `outputs/`.
