# Final Project — GastroVision GI Endoscopy Classification

**Course:** AIN501 · Artificial Intelligence (Deep Learning) · MSE FSB
**Team:** 3 members · **Theme:** Beat a published baseline (Data 70% / Model 30%)

---

## 1. What we're building

Multi-class classification of **gastrointestinal (GI) endoscopy images** on the
**GastroVision** dataset (~8,000 images, 27 classes / **22 used in experiments**).
Classes span anatomical landmarks, pathological findings (polyps, ulcers, cancer,
esophagitis...), normal findings, and therapeutic interventions.

The core challenge is a **heavy long-tail distribution**: some diseases have fewer
than 10 images while common classes have hundreds. This is why the published
**macro-F1 baseline is only 0.6504** — and why the improvement work lives in the
**data** (imbalance handling), not just the model.

### Goal
Raise **macro-F1 from 0.6504 → ≥ 0.75** on the **original stratified 60:20:20 split**,
reported as **mean ± std over ≥ 3 seeds**, with a **per-class F1 table** and confusion matrix.

### Published baseline (paper Table 2 — arXiv 2307.08140)

> ✅ **Verified against the paper (arXiv 2307.08140, Table 2).** Still **reproduce DenseNet-121
> yourself** on the same split — reporting your own reproduced number is what earns the
> "scientific honesty" marks and preempts the "the baseline was just undertrained" critique.

| Model | Macro-F1 | Micro-F1 | MCC |
|---|---|---|---|
| **DenseNet-121 (pretrained)** | **0.6504** | 0.8203 | 0.7987 |
| ResNet-50 (pretrained) | 0.6176 | 0.8146 | 0.7921 |
| DenseNet-169 | 0.4883 | 0.7055 | 0.6685 |
| EfficientNet-B0 | 0.4519 | 0.6759 | 0.6351 |
| ResNet-50 (from scratch) | 0.4330 | 0.6816 | 0.6416 |

**Note:** the paper baseline has **6 CNNs and zero Transformers** — adding
Swin-T / ViT-S / a hybrid (CoAtNet) into this exact table is the "CNN vs Transformer
vs Hybrid" comparison the rubric asks for, and it's almost free because the CNN rows already exist.

### Links
- Dataset: [GitHub](https://github.com/DebeshJha/GastroVision) (official download / request form) · paper arXiv 2307.08140
- Original split: **Stratified 60:20:20** — reproduce with a **fixed, published seed** (do not re-split ad-hoc → avoids data leakage). If the authors ship a split file, use it verbatim.

---

## 2. Grading rubric we must hit (Data 70 / Model 30)

The instructor weights the report as follows. Every section below maps to points.

| # | Section | Weight | What it means |
|---|---|--:|---|
| 1 | Problem & published baseline | 5% | Quote paper table, state split/metric/protocol, **report how much you reproduced** |
| 2 | Data analysis (EDA) | 20% | Class distribution, image quality, artifacts, suspicious labels, **leakage check**. With figures |
| 3 | **Data processing** | 30% | Denoise, normalize, domain-justified augmentation, class-balancing. **Every step needs an ablation proving it helps** |
| 4 | Labels & validation | 20% | Long-tail handling, confusion matrix, per-class F1, typical error examples |
| 5 | Architecture: CNN vs Transformer vs Hybrid | 15% | 3 branches, same protocol, same seed. Argue why local/global matters here |
| 6 | Transfer learning: freeze vs trainable | 10% | Linear probe / progressive unfreezing / layer-wise LR decay / full fine-tune. Comparison table |
| 7 | Deployment | — | Export ONNX, measure latency + model size, Gradio/Streamlit demo ("Completeness of the Product") |

### Five non-negotiable rules
1. **Use the author's original split.** If you must re-split, publish the seed and explain.
2. Medical data → **split is patient/image-level as the authors defined**; state in the report that leakage was checked.
3. Imbalanced data → always report **macro-F1**, never accuracy or weighted-F1 alone.
4. **≥ 3 seeds, report mean ± std.** A single number beating baseline by 0.5 pt is meaningless.
5. **Reproduce the strongest baseline before claiming you beat it.** If you can't reproduce it, say so — that's honesty, not a deduction.

---

## 3. Team split (3 people)

Everyone builds on the **shared baseline notebook** (`notebooks/00_baseline_gastrovision.ipynb`)
so all measurements use the same split, seeds, and eval harness.

| Member | Owns | Deliverable |
|---|---|---|
| **Member A — Data (70%)** | EDA + data processing + imbalance | Class-distribution analysis, augmentation pipeline (GI-domain justified), LDAM / Balanced-Softmax / decoupled training / repeat-factor sampling, **ablation table per step**, leakage check |
| **Member B — Model (30%)** | CNN vs Transformer vs Hybrid | Reproduce DenseNet-121 baseline, add ConvNeXt-T / Swin-T / ViT-S / CoAtNet under the **same protocol**, fill the extended baseline table |
| **Member C — Transfer + Deploy + Eval harness** | Freeze/trainable study + product | Linear probe vs progressive unfreezing vs layer-wise LR vs full fine-tune (comparison table), ONNX export + latency/size, Gradio demo, and owns the **shared eval code** (macro-F1, per-class F1, confusion matrix, multi-seed runner) |

---

## 4. Getting started (Google Colab)

The base notebook `notebooks/00_baseline_gastrovision.ipynb` is written **Colab-first (T4)**.
Compute per run is light: **~15–25 min @ 224px** — budget for ≥ 15 experiments × 3 seeds.

**One-time setup:**
1. Download the GastroVision zip from the [official source](https://github.com/DebeshJha/GastroVision)
   and drop it on your Google Drive (e.g. `MyDrive/gastrovision.zip`).

**Each session:**
1. Colab → **Runtime → Change runtime type → T4 GPU**.
2. Open the notebook. In the **"Get the data"** cell, set `DRIVE_ZIP` to your zip path — it mounts
   Drive and unzips into `/content/gastrovision`.
3. Run the setup → data → split cells (shared; don't modify without telling the team).
4. Reproduce the DenseNet-121 baseline → confirm you land near macro-F1 0.65.
5. Each member branches into their section (7A/7B/7C), reusing the shared eval harness.

> **Local run** works too — the notebook auto-detects the environment. Just
> `pip install -r requirements.txt` and place the dataset under `data/gastrovision/`; it will
> read from there and write checkpoints to `checkpoints/`. Keep final reported numbers on the
> same T4 setup so they're comparable.

### Auto-detection (baked into the notebook)
- **Compute device** (GPU/CPU) is detected automatically; AMP + `pin_memory` enable only on GPU.
- **Environment** (Colab vs local) is detected → data/checkpoint/output paths switch automatically.
- **Checkpoints** save after each seed → to Drive on Colab, to `checkpoints/` locally (survive disconnects).
- `timm` is pip-installed in the first cell (not preinstalled on Colab); needed by Member B.
- **Dataset auto-downloads** on Colab via `gdown` from the official public Drive folder (first run only).
  If the folder download is rate-limited, fall back to a zip on your Drive (commented in the cell).
- Classes are filtered by the **paper's exact rule** (`>5 images` → 22 classes), with an
  `assert NUM_CLASSES == 22` sanity check that also catches a wrong `DATA_DIR` nesting level.

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
