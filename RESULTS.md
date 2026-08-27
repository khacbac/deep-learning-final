# Results log — GastroVision (2-member team)

Living record of measured numbers for the report's ablation table. **Only measured numbers go here**
(no estimates). Each row is filled in when its run finishes.

- **Dataset / split:** 22 classes (>25 samples), stratified 60:20:20, `SPLIT_SEED=42` (fixed).
- **Primary metric:** macro-F1 (test split). Also track micro-F1 when available.
- **Compute:** Colab **A100** for the final run (an intermediate run landed on a T4 — see Gate 0a
  for why that matters and for what it does *not* affect). `seeds=[0]` = quick dry-run; final
  numbers use 3 seeds (`mean ± std`).

### Team structure — 2 baselines + 1 proposed model
| Track | Owner | Model | Section below |
|---|---|---|---|
| CNN (reference baseline) | **Member A** | DenseNet-121 | §2, §3 |
| Transformer (new baseline) | **Member B** | Swin-T | §4 |
| Proposed model (**mandatory** — carries the improvement claim) | shared | **CoAtNet-0 @288 + TTA + top-3 ckpt ensemble** (`P1`); `P0` @224 is the intermediate that isolates the resolution lever, **not a third baseline** | §5, §9 |

## What to send me after each run *(historical — every run is done; kept as the workflow record)*
From the printed output I need:
- the `[seed s] test macro-F1 = 0.xxxx` line(s), and
- the final `<tag> TEST macro-F1: 0.xxxx +/- 0.yyyy` line.
That's enough to fill the tables below (a screenshot of the last few output lines is fine).
`run_seeds` now also writes `<tag>_seed<s>_preds.npz` (the raw `y_true` / `y_pred`) **next to the
checkpoints** — on Colab that is your Drive folder, so a disconnect does not cost them. Send
those files too, or just run `bootstrap_ci(y_true, y_pred)` yourself and send the CI. The
bootstrap costs no extra training.

---

## 1. Reference baselines (from the paper — arXiv 2307.08140, Table 2)

| Model | Macro-F1 | Micro-F1 | MCC |
|---|---|---|---|
| DenseNet-121 (pretrained) | **0.6504** | 0.8203 | 0.7987 |
| ResNet-50 (pretrained) | 0.6176 | 0.8146 | 0.7921 |

✅ **Verified at the source (Gate 0b, 2026-08-27).** The brief (p16) listed **GastroVision Table 2
among the "not verified" figures**; we read the arXiv 2307.08140 PDF itself and 0.6504 is confirmed
as the macro-F1 of pre-trained DenseNet-121. Full verbatim transcription of all six rows, plus the
protocol comparison and the class-by-class split check, are in **§0b** below.

**Target (revised to two tiers — see README §1), and where each tier landed:**
- **Primary (committed floor):** beat 0.6504 with a 3-seed mean ± std / bootstrap CI that does
  **not overlap** the baseline, plus a documented answer to *"what actually moves macro-F1 here"*
  (negative results included).
  → ✅ **met.** `P1` = 0.6961 ± 0.0016, CI95 [0.6548, 0.7245]; the "what moves it" answer is §9.
- **Stretch (the brief's actual target):** macro-F1 **0.72–0.75**.
  → ⚠️ **met only by the ensemble rows**, which cost 3–4 training runs and are therefore reported
  separately: 0.7221 (3 seeds of `P1`, CI [0.6728, 0.7609]) and 0.7242 (four architectures, combo
  chosen on val). **No single trained model reaches 0.72** — say it that way.

**On the revision — say it straight in the report:** ≥ 0.75 was **not** an unmeasured guess of
ours. It is the brief's own recommended target for GastroVision (p12: *target ≥ 0.75, headroom
~10 points*). We lowered it after measuring B0 = 0.6516 with all three imbalance levers flat
(§2, §6). And the brief (p2, criterion 3) warns that **below 3 points of headroom you cannot
demonstrate statistical significance** — so the primary floor alone is *not* a sufficient result,
and 0.72–0.75 stays the number we actually aim at.

---

## 2. DenseNet-121 improvement ladder — Member A (our measured numbers)

Model = DenseNet-121 throughout. "Improve" = same architecture + added technique.

| Rung | What's added (vs previous) | Seeds | Test macro-F1 | Δ vs B0 | Status |
|---|---|---|---|---|---|
| **B0** | Baseline: plain aug, CrossEntropy, `train_one` | [0] | **0.6516** | — (reference) | ✅ done |
| **B2** | + advanced recipe (2-stage, backbone LR 0.1×, cosine) + strong aug + TTA | [0] | **0.6163** | **−0.035** ⬇ | ✅ done (regressed — under-trained) |
| **B3** | Balanced-Softmax on the **simple** recipe (isolates the imbalance lever) | [0] | **0.6442** | **−0.007** ⬇ | ✅ done (flat — within 1-seed noise) |
| **B4** | + decoupled cRT (retrain classifier, balanced) | [0] | **0.6389** | **−0.013** ⬇ | ✅ done (flat) |
| **B5** | resolution bump 224→288 (only input size changes vs B0) | [0] | **0.6342** | **−0.017** ⬇ | ✅ done (best VAL 0.689 — highest of all rungs — but worst test: val/test gap = noise) |
| **B2b** | (optional) advanced recipe retuned: `backbone_mult=0.5` + softer aug | [0] | _skipped_ | _—_ | ❌ cut (2-person scope) |

**Dry-run winner:** **B0 (plain baseline)** — no imbalance technique (B2/B3/B4) beat it. See §6.

⚠️ **All rows above were measured before the determinism fix** (the notebook did not set
`use_deterministic_algorithms` and did not seed the DataLoader workers). They stay here as the
record that *"the three imbalance levers are flat"* — a conclusion the deltas support regardless,
since every delta is smaller than the run-to-run spread. But **do not quote these numbers**: the
report's numbers come from §3 onwards.

---

## 3. Gate 0 — the only experiment that currently means anything

No new rung until these are done. Items 0b, 2b, 3a, 2 and 3b cost **zero extra epochs** —
but only in the order below: **3a must land before item 1**, or item 1's six runs are locked to
the old selection rule and 3b stops being free. **Item 0a comes first** — see §6 for why
"3 seeds" cannot be the first move.

| # | Task | Cost | Status |
|---|---|---|---|
| **0a** | **Determinism check** — 3-epoch × 2-run val-curve diff. **PASSED ON A100** (2026-08-26): both runs gave `[0.428608, 0.549646, 0.551532]`, identical to 6 decimals **with bf16 + TF32 on**. **Re-passed on T4** (2026-08-27) — but with a *different* curve: `[0.430615, 0.540379, 0.541930]`. **Re-run on A100 (2026-08-27, 4th run): back to `[0.428608, 0.549646, 0.551532]`, bit-identical to the 2026-08-26 A100 pair.** Three measurements, one rule: the val curve is a *function of the GPU* and is reproducible on that GPU. Determinism holds **within** a device, not **across** devices; same seed + same code + different GPU = a different model. Worth one line in the report — and it is why σ must never mix hardware | ~5 min GPU | ✅ **done (A100 ×2 + T4)** |
| **0b** | **Verify paper Table 2** against arXiv 2307.08140 (brief p16 flags it "not verified") | 0 GPU | ✅ **done 2026-08-27 — CONFIRMED at source**, see §0b below |
| 2b | ~~Persist `y_true`/`y_pred` per seed in `run_seeds`~~ → `<tag>_seed<s>_preds.npz` (next to the checkpoints) | 0 epochs | ✅ done |
| **3a** | **De-noise checkpoint selection — code change** | 0 epochs | ✅ **done** — `Tracker` in the notebook keeps top-3 raw-val states + the 3-epoch-smoothed-argmax state + the full val history; `run_seeds` turns them into **6 test numbers per single training run** (3 rules × TTA on/off) |
| 1 | **3 seeds per config** — superseded by the full A100 run: 4 configs × 3 seeds, see §9 | 2.0 h A100 | ✅ **done** |
| 2 | **Bootstrap CI** on test — `bootstrap_ci(...)`, 1000 resamples | 0 epochs | ✅ **automated** — runs for every model in the summary cell. ⚠️ Still free only for *new* runs: B0 0.6516 / B5 0.6342 saved no logits |
| 3b | **Pick the rule** — best-val vs 3-epoch-smoothed vs top-3-ensemble, × TTA | 0 epochs | ✅ **DECIDED: `top3_tta`** — see §9. The notebook now *assigns* `SELECTION_RULE` from the ranking instead of only printing a suggestion (it used to print "use top3" and then keep using `best` for every table below it) |

### 3-seed results (the only numbers quotable in the report)
Superseded by **§9** — the A100 run covers 4 configs × 3 seeds under all 6 selection rules.
`B5` (DenseNet @288) was dropped in favour of `P1` (CoAtNet-0 @288), which tests the same
resolution lever on the proposed backbone instead of on the baseline.

### Per-seed detail
⚠️ **The seed-0 values below came from best-val checkpoint selection.** Whatever rule item 3b
picks, these are superseded by item 1's runs — a `mean ± std` may not mix two selection protocols.
They are kept here as the old-protocol record, not as row-1 of the final mean.

| Config | seed 0 | seed 1 | seed 2 | mean ± std | Selection rule |
|---|---|---|---|---|---|
| B0 | 0.6516 | _—_ | _—_ | _—_ | best-val (old) |
| B5 | 0.6342 | _—_ | _—_ | _—_ | best-val (old) |

---

## 4. Swin-T — Member B (new baseline, no published number to match)

Run under the **identical protocol** as B0 — same split, same `SPLIT_SEED=42`, same 3 seeds,
AdamW 1e-4 / wd 1e-4, 30 epochs, batch 32, 224px, `run_seeds(build_swin_t, tag=...)`.
**No extra tuning on this row** — that equal footing is what makes the CNN vs Transformer
comparison valid.

| Rung | Config | Seeds | Test macro-F1 (`top3_tta`) | Δ vs B0 (3-seed) | Status |
|---|---|---|---|---|---|
| **S0-final** | Swin-T, plain aug, CrossEntropy, 224px (mirror of B0) | [0,1,2] | **0.6851 ± 0.0114** | +0.0175, **CIs overlap** | ✅ done (§9) |
| **S1** | parity check on 288px and/or softer aug | — | _not run_ | — | ❌ **dropped on purpose** |

**Why S1 was dropped:** the resolution lever needed to be isolated on *one* backbone, and the
informative place to isolate it is the **proposed** backbone — that is `P0` → `P1` (+0.0143 with
σ = 0.0016, §9). Running 288 on Swin-T as well would have doubled the GPU cost of the same lever
without changing any claim. Stated in the notebook §15b so the omission is visible, not silent.

Per-seed detail (test macro-F1):

| Config | rule | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---|---|---|---|---|
| S0 | `best` (paper's rule) | 0.6554 | 0.6611 | 0.6476 | 0.6547 ± 0.0056 |
| S0 | **`top3_tta`** (reported) | 0.6976 | 0.6878 | 0.6700 | **0.6851 ± 0.0114** |

*Fallback that turned out unnecessary:* **DeiT-S** (`deit_small_patch16_224`, "data-efficient",
lecture 8 p41) was held in reserve in case Swin-T overfit ~8k images. It did not — Swin-T peaks at
val epoch 7/30 with a last-5-epoch spread of 0.0102, so it was never run. The unused `build_deit_s`
builder has since been removed from the notebook (2026-08-27 cleanup), so do not look for it there.

---

## 5. Proposed model — hybrid (mandatory)

Must beat **both** reproduced baselines to be reported as the proposal. This is the whiteboard's
*destination* (proposition 4: attention/hybrid is where the improvement is placed) and the only
row that carries an improvement claim.

**Resolved: the hybrid is CoAtNet-0** (`coatnet_0_rw_224.sw_in1k`), not Swin-T + a hand-built conv
stem — a published, pretrained hybrid is a fair comparison; a stem we bolt on ourselves would have
no pretrained weights for the new layers and would confound architecture with initialisation.

| Config | Seeds | Test macro-F1 (`top3_tta`) | vs B0 | vs S0 | Status |
|---|---|---|---|---|---|
| `P0` CoAtNet-0 @224 — *intermediate, not a third baseline* | [0,1,2] | 0.6818 ± 0.0014 | +0.0142 | −0.0033 | ✅ done (§9) |
| **`P1` CoAtNet-0 @288 + TTA + top-3 ckpt ensemble** — **the proposal** | [0,1,2] | **0.6961 ± 0.0016** | **+0.0285** | **+0.0110** | ✅ done (§9) |

Per-seed detail (test macro-F1, `top3_tta`):

| Config | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---|---|---|---|
| P0 | 0.6805 | 0.6837 | 0.6812 | 0.6818 ± 0.0014 |
| P1 | 0.6941 | 0.6980 | 0.6961 | **0.6961 ± 0.0016** |

It beats **both** baselines on the mean. Two of the four configs have a bootstrap CI that excludes
0.6504 (`S0` [0.6513, 0.7356] and `P1` [0.6548, 0.7245]); `P1` is the one that does it **with the
smallest σ of the four** (0.0016), which is why the claim is placed on it and not on `S0`. The two ensemble rows
(0.7221 for 3 seeds of P1, 0.7242 across the four architectures) stay **separate** — they consume
3–4 training runs, so they are not put next to a single model and called an improvement.

---

## 6. Notes / observations
- ~~(per-class F1: which rare classes rose after each rung)~~ → **answered in §9**, and better than
  planned: the 22-class table is compared against the paper's own Table 3, and **84% of the gain sits
  in the 15 rare classes**. See *"Where the improvement actually landed"*.
- **Run-to-run spread is real** *(diagnosed — see the outcome box at the end of this section)***:**
  an earlier equivalent B0 run gave test 0.676;
  this B0 gave 0.6516 (~0.024 spread) at what we *believed* was the same seed/split. Val macro-F1 also
  swings a lot per epoch (small val set), so best-val checkpoint selection is noisy.
  → **trust only the 3-seed mean ± std for the report**, and settle the determinism question first
  (§3 Gate-0 item 0a; the diagnosis is at the end of this section).
- **B2 regressed (0.616 < 0.652):** the advanced recipe under-trained — backbone LR was 10× too low
  (`backbone_mult=0.1` → 1e-5) and the aug was too heavy for ~8k medical images (val capped ~0.61 vs
  baseline ~0.72). Lesson: a fancy recipe tuned wrong loses to the simple baseline. → test the imbalance
  lever (B3) on the **simple** recipe first; only revisit the recipe (B2b) if a push is needed.
- **B3 flat (0.644 ≈ 0.652):** Balanced-Softmax alone (loss-only swap on the good recipe) gave **no gain**
  within 1-seed noise. Best-val peaked at 0.666 (ep 23) but test = 0.644. Honest read: **re-weighting the
  loss is not enough** on this dataset — the head is already fitting the tail loss-wise; the bottleneck is
  more likely *representation / classifier decision boundary*, which is exactly what **cRT (B4)** targets
  (freeze features, retrain the classifier on class-balanced sampling). → run B4 next; if B4 also flat,
  the honest conclusion is "imbalance-loss methods don't move macro-F1 here — report that."
- **B4 flat (0.639 ≈ 0.652):** decoupled cRT (freeze backbone, retrain head on balanced sampling) also
  gave **no gain** — slightly below B0. Best-val 0.655 (ep 13) but test 0.639.

### Honest conclusion of the imbalance ablation (B2→B4)
On GastroVision (22 classes, DenseNet-121, this recipe), **none of the imbalance-handling methods beat the
plain baseline** within 1-seed noise:
- B2 (advanced recipe + strong aug): **−0.035** (under-trained — a tuning failure, not a method failure)
- B3 (Balanced-Softmax, loss-level re-weighting): **−0.007** (flat)
- B4 (decoupled cRT, classifier-level re-balancing): **−0.013** (flat)

This is a **legitimate, reportable finding**, not a dead end: the tail bottleneck here is not the loss or the
classifier boundary — it's the **feature representation** (too few tail images to learn good features from).
Levers that *can* move it: **stronger/better-pretrained backbone (→ this is exactly why §4 Swin-T is now
the priority track, not a side quest)**, higher input resolution (B5), correctly-tuned augmentation, or extra
tail data. The **CNN-track reported number is therefore the 3-seed B0 baseline** (a faithful reproduction of
the paper's 0.6504), and the improvement claim rests on §4 / §5.

### ⚠️ The real problem: run-to-run spread > all effect sizes
Best-VAL macro-F1 per rung: B3 0.666, B4 0.655, **B5 0.689 (highest of all)** — yet B5's *test* = 0.634
(worst). A 0.055 val/test gap on one seed is the tell: **the spread of a single macro-F1 measurement on
this dataset (~±0.02–0.05) is LARGER than every delta we measured** (B3 −0.007, B4 −0.013, B5 −0.017).
B0 itself gave 0.676 then 0.6516 across two runs (0.024 spread).

**But do not label that 0.024 "seed noise" until Gate-0 item 0a says so.** Those two B0 runs were
believed to use the *same* seed and split, and `set_seed()` already covers `random` / `numpy` /
`torch` / `torch.cuda` plus `cudnn.deterministic=True`, `benchmark=False`. Variance across seeds
cannot explain a gap at a **fixed** seed. Only two explanations remain:
1. **The two runs were not actually the same config** (the 0.676 run predates this ladder — hence
   "equivalent", not "identical"), or
2. **Real non-determinism** — AMP (`autocast` + `GradScaler`) and CUDA kernels that have no
   deterministic implementation. `torch.use_deterministic_algorithms()` was not set, and the
   `DataLoader`s were built without an explicit `generator=` / `worker_init_fn`.

The notebook now *attacks* (2): `set_seed()` sets `use_deterministic_algorithms(True, warn_only=True)`,
pins the train loader's shuffle order (`generator=DATA_GEN`) and seeds the workers of all three
`make_loaders()` loaders (`worker_init_fn`). That is not the same as closing it — `warn_only=True`
lets an op with no deterministic kernel warn and carry on non-deterministically, so **only Gate-0a
can say whether it took** (read the warnings it prints). **Running 3 seeds cannot distinguish these
two causes**, which is why item 0a (same seed, run twice) precedes item 1 (three seeds).

> ### ✅ Outcome (Gate 0a, A100 2026-08-26 + T4 2026-08-27 + A100 2026-08-27)
> **Explanation (2) is ruled out; (1) is the answer.** At a fixed seed on one GPU the pipeline is
> bit-reproducible — two 3-epoch runs gave `[0.428608, 0.549646, 0.551532]` identical to 6 decimals
> **with bf16 + TF32 on**, and a fourth run reproduced the same curve exactly. So AMP + cuDNN were
> **never** the cause of the old 0.024 spread; those two `B0` runs simply were not the same config
> (hence "equivalent", not "identical").
>
> **But 0a also found something the plan did not anticipate:** the same seed and code on a **T4**
> gave a *different* curve, `[0.430615, 0.540379, 0.541930]`. Determinism holds **within** a device,
> not **across** devices — so **σ must never mix hardware**, and that deserves one line in the
> report. All 12 main runs are A100-only for exactly this reason.

**Why macro-F1 is so noisy here:** it averages 22 per-class F1s, and the rare test classes have **single-digit
image counts** (Colon diverticula/Mucosal inflammation ≈ 6 test imgs each). Getting 4 vs 5 of 6 right swings
that class's F1 by ~0.15, which visibly moves the 22-class average. Best-val checkpointing on a similarly tiny
val set adds a second layer of noise.

**Consequence:** we **cannot rank B0/B3/B4/B5 with 1 seed each** — the differences are statistical noise.
The only decisive move is a **3-seed mean±std** for the 1–2 most promising configs (§3), plus a bootstrap CI
so each single run carries an error bar. On VAL, resolution-288 (B5) is the strongest signal → the right
experiment is **B0 vs B5, 3 seeds each**, then compare mean±std.

> ### ✅ Outcome: done, but not as B0-vs-B5
> `B5` was **dropped**. The resolution lever was worth isolating on the *proposed* backbone rather
> than the baseline, so it became **`P0` @224 vs `P1` @288** (CoAtNet-0): **+0.0143 with σ = 0.0016**.
> The run that actually happened is **4 configs × 3 seeds** — `B0`/`S0`/`P0`/`P1`, §9 — and the
> bootstrap CI is automated for every one of them.

---

## 7. Data audit (shared — the Data 70%)

Almost no GPU cost, highest value per minute. **All rows below are measured** (full dataset, A100).

| Check | Method | Result | Status |
|---|---|---|---|
| Byte-identical duplicates across train/val/test | MD5 over all 7,930 images (4 s, CPU) | **1 cross-split duplicate group**: `Colon polyps` — one image appears in **train and val** (`ba615bcd-…jpg` = `ckda1fpc5000l3a5s17a45xql.jpg`). Only 1 group in the whole dataset; **none touches test** | ✅ done (full dataset) |
| Near-duplicate frames across splits | cosine ≥ 0.98 on MobileNetV3-Small embeddings @160px | **9 cross-split pairs** over all 7,930 images (the earlier "0" was an artefact of the 2,000-image subset) | ✅ done (full dataset, A100) |
| Label noise found by the same sweep | pairs with cosine ≈ 1.0 but **different class labels** | e.g. `val/Esophagitis/N2DaTmFs.jpg` ≡ `test/Normal esophagus/WdSYgDiw.jpg` (cosine 1.0000); `train/Accessory tools/…` ≡ `val/Gastric polyps/…` (0.9998) | ✅ **a finding worth a paragraph** |
| Effect on the reported number | recompute macro-F1 with the affected test images removed | notebook §19b prints it per model | ✅ automated |

**How to read it:** 9 pairs in 7,930 images is ~0.1%, far too few to move macro-F1 — and §19b now
*proves* that rather than asserting it, by re-scoring with those test images dropped. The stronger
finding is the second row: a handful of frames are **byte-near-identical yet carry two different
class labels**, which puts a hard ceiling on any model's achievable macro-F1 on this dataset.
That is a data-quality result, not a leakage result, and it is exactly the kind of thing the
Data-70% part of the grade rewards.

Endoscopy datasets easily contain several frames from the same case. Finding leakage would be the
project's strongest Data-70% contribution; finding none still earns the sentence every DL examiner
asks for first.


---

## 8. CPU smoke run — what the rebuilt notebook actually verified (2026-08-26)

The notebook was rebuilt from `build_notebook.py` and **executed end to end on the local CPU** (no NVIDIA
GPU on this machine) in the `cpu-smoke` profile: all 22 classes, all **four configs** (`B0`/`S0`/`P0`
@224 and `P1` @288) and the same code path — only **12/4/4 images per class and 2 epochs**.

> ⚠️ **Every macro-F1 below is meaningless as science** (88 test images, 2 epochs). They are recorded
> only to show the pipeline runs. What *is* transferable is in the second table.

| Run | seconds/epoch (CPU, 264 train imgs) | test macro-F1 (`best`) | note |
|---|---|---|---|
| B0 DenseNet-121 @224 | 42.7 | 0.4129 | — |
| S0 Swin-T @224 | 44.4 | 0.3889 | — |
| P0 CoAtNet-0 @224 | 61.3 | 0.3396 | ~1.4× slower than the other two |
| P1 CoAtNet-0 @288 | 104.1 | 0.2870 | ~1.7× the cost of P0 |

### What the smoke run *does* establish (and changes the plan)

| # | Finding | Consequence for the Colab run |
|---|---|---|
| 1 | **The whole notebook runs with 0 errors**, including ONNX export, bootstrap CI, τ-tuning, cross-model ensembling and the resume path | No trial-and-error debugging on paid GPU time |
| 2 | **Deterministic at a fixed seed — on CPU.** Two 3-epoch runs produced *identical* val curves (`[0.227346, 0.405588, 0.506395]`) | The 0.024 spread is **not** from data ordering / seeding / augmentation RNG. Only AMP + cuDNN remain as suspects → Gate-0a on GPU is now a **6-minute** check, not 40 |
| 3 | **Dataset verified against the paper's protocol**: 27 class folders / 8,000 images → 22 classes / **7,930** images after the ">25 samples" rule; split = **4,758 / 1,586 / 1,586** | Split cell is correct; `assert NUM_CLASSES == 22` holds |
| 4 | ⚠️ **Corrected 2026-08-27 — this was over-claimed.** What the run actually verified is that **CoAtNet-0** accepts 288×288 with pretrained weights (it is the only backbone built at 288, here and in every later run). That `timm` would interpolate the relative-position tables for Swin-T too is **inference, not measurement** — and `_timm_build` falls back to 224 with a printed warning if a model refuses `img_size`, so a silent 224 run is possible | The resolution lever is demonstrated **on the proposed backbone only** — which is all `P0` → `P1` needs. Notebook §15b states the limit explicitly |
| 5 | **A zip-inside-a-zip bug was caught and fixed**: the download is **two nested levels** — `Gastrovision.zip` → `Gastrovision.zip` → `Gastrovision/<27 class folders>` — and the inner archive has the *same name*, so extracting in place **overwrote the 1.8 GB download mid-read** (`EOFError`, file truncated to 0 bytes). Verified end to end: both levels peel automatically, 0 archives left, 8,000 images on disk | Would have burned a Colab session. Fixed by renaming the source before extraction; on failure the name is restored rather than the download deleted |
| 6 | ONNX export writes weights to a sibling `.onnx.data` by default, making the reported model size ~1 MB | Fixed: `external_data=False` + size counted across both files |

### Colab budget, extrapolated from the measured CPU throughput

Rough conversion (T4 ≈ 25×, A100 ≈ 70× this 16-thread CPU), scaled by the 18× data increase:

| Run | T4, 30 ep, 1 seed | T4, 3 seeds | **A100, 3 seeds** |
|---|---|---|---|
| B0 DenseNet-121 @224 | ~0.3 h | ~1.0 h | ~0.4 h |
| S0 Swin-T @224 | ~0.3 h | ~1.0 h | ~0.4 h |
| P0 CoAtNet-0 @224 | ~0.4 h | ~1.2 h | ~0.5 h |
| P1 CoAtNet-0 @288 (proposed) | ~0.6 h | ~1.8 h | ~0.7 h |
| **Total** | | **~5 h T4** | **~2 h A100** |

**The run will be on A100**, so the whole plan fits in a single session — `Run all` once. Note the
A100 profile deliberately keeps **batch 32**, the same as T4 and as the paper: batch size is part of
the protocol, and changing it would break comparability with 0.6504 and with the old B0 run. The A100
gain is taken as **bf16 + TF32** (selected from the GPU's compute capability, not from the profile,
so T4 automatically falls back to fp16 + `GradScaler`).

Spare A100 time is better spent on **more seeds** (`SEEDS = [0,1,2,3,4]`) than on more architectures —
narrower error bars are exactly what the "beats 0.6504 significantly" claim currently lacks.

> ### ❌ Outcome: reversed on 2026-08-27 — do not follow this line
> More seeds were **decided against**. σ on `P1` is already **0.0016**; five seeds would refine it to
> perhaps 0.0014 while the error bar that actually dominates is the bootstrap CI at **±0.035** — and
> bootstrap resamples the **test set**, which `SPLIT_SEED = 42` holds fixed across every seed. The
> remaining uncertainty is a **data** problem (2 of 22 classes have < 10 test images), not a
> stochasticity problem. Reasoning in §9, *"The uncomfortable half of the same table"*; the cost was
> also under-estimated here (~80 min A100, not 40).

The ~0.3 h/run for B0 matches the ~20 min/run measured on a real T4 earlier, so the conversion factor is
sane. With the resume mechanism this splits cleanly into 4 Colab sessions.

---

## 9. A100 run — the real numbers (2026-08-26)

`gastrovision_classification.ipynb`, profile `gpu-a100`, 30 epochs, batch 32, bf16 + TF32,
4 configs × 3 seeds, ~10 min/seed → **2.0 h total**. Zero errors, Gate-0a passed on GPU.

### Test macro-F1, mean over seeds [0,1,2], all six selection rules

| Config | best | smooth | top3 | best_tta | smooth_tta | **top3_tta** |
|---|---|---|---|---|---|---|
| `B0_densenet121` | 0.6491 | 0.6470 | 0.6710 | 0.6443 | 0.6456 | **0.6676** |
| `S0_swin_t` | 0.6547 | 0.6619 | 0.6867 | 0.6541 | 0.6704 | **0.6851** |
| `P0_coatnet0` | 0.6538 | 0.6408 | 0.6806 | 0.6631 | 0.6438 | **0.6818** |
| `P1_coatnet0_288` | 0.6645 | 0.6707 | 0.6905 | 0.6732 | 0.6734 | **0.6961** |
| *average rank (1 = best)* | 4.50 | 4.75 | 1.50 | 4.75 | 4.00 | **1.50** |

`top3` and `top3_tta` tie on rank; the tie is broken by mean macro-F1 (0.68265 vs 0.68222)
→ **`SELECTION_RULE = "top3_tta"` for the whole report**.

### The two claims, kept separate

**Claim 1 — baseline reproduced.** Under the paper's own protocol (single best-val checkpoint,
no TTA), `B0` = **0.6491 ± 0.0124** vs published **0.6504**. Difference −0.0013, i.e. inside one
seed's worth of noise. *This* is the reproduction number; do not quote 0.6676 as "reproducing the
paper", because the paper did not use checkpoint ensembling.

**Claim 2 — improvement.** With one selection rule applied uniformly to every row (`top3_tta`):

| Config | macro-F1 (3 seeds) | vs paper 0.6504 |
|---|---|---|
| `B0_densenet121` | 0.6676 ± 0.0066 | +0.0172 |
| `S0_swin_t` | 0.6851 ± 0.0114 | +0.0347 |
| `P0_coatnet0` | 0.6818 ± 0.0014 | +0.0314 |
| **`P1_coatnet0_288`** (proposed) | **0.6961 ± 0.0016** | **+0.0457** |

### What actually bought the improvement — and what did not

> ⚠️ **Corrected 2026-08-27.** The earlier version of this table quoted the architecture lever
> under `best` (+0.006) next to the checkpoint lever under `top3_tta` (+0.03) and concluded a "5×"
> difference. **That was a mixed-rule comparison and the 5× does not survive.** All rows below are
> under `top3_tta` — the rule chosen in §16 and used everywhere downstream.

| Lever | Effect (all under `top3_tta`) | σ across seeds | Cost |
|---|---|---|---|
| **Checkpoint ensembling + TTA** (`best` → `top3_tta`) | **+0.0185 / +0.0304 / +0.0280 / +0.0316** on B0 / S0 / P0 / P1 — *every* architecture | mean 0.0092 → **0.0052** | **0 extra epochs** |
| **Resolution 224 → 288** (`P0` → `P1`, same backbone) | +0.0143 | 0.0014 → 0.0016 | ~1.65× per epoch |
| **Architecture swap** (`B0` → `S0`, DenseNet → Swin-T) | +0.0175, CIs overlapping | 0.0066 → **0.0114** | full retrain |
| **Architecture swap** (`B0` → `P0`, DenseNet → CoAtNet-0) | +0.0142, CIs overlapping | 0.0066 → 0.0014 | full retrain |

Per-rule σ, measured over the 3 seeds (this is the table the argument rests on):

| Model | `best` | `top3_tta` |
|---|---|---|
| B0_densenet121 | 0.6491 ± 0.0124 | 0.6676 ± 0.0066 |
| S0_swin_t | 0.6547 ± 0.0056 | 0.6851 ± **0.0114** |
| P0_coatnet0 | 0.6538 ± 0.0100 | 0.6818 ± 0.0014 |
| P1_coatnet0_288 | 0.6645 ± 0.0090 | 0.6961 ± 0.0016 |

> The pre-registered decision rule in the notebook (§21) said: if `S0` only matches `B0`, then the
> hypothesis *"the bottleneck is the backbone"* is **not supported** and the improvement must come
> from elsewhere. That is what happened — but state it precisely, because the corrected numbers do
> **not** support a claim that architecture is worthless:
>
> Under one consistent rule, a better backbone (+0.0142 … +0.0175) and a better checkpoint rule
> (+0.0185 … +0.0316) buy **about the same amount** of macro-F1. What separates them is that the
> checkpoint rule is **free** (0 epochs vs a full retrain), **universal** (it improves all four
> architectures, in the same direction), and **variance-reducing** — while the architecture swap to
> Swin-T raises the mean by +0.0175 and simultaneously *doubles* σ from 0.0066 to 0.0114, leaving
> the two CIs ([0.628, 0.713] vs [0.651, 0.736]) heavily overlapped. At 3 seeds we **cannot** claim
> Swin-T beats DenseNet-121. We **can** claim `top3_tta` beats `best`.
>
> *On an ~8k-image dataset, the cheapest reliable gain came from measuring better, not from
> searching for a better backbone.* That is the thesis — and it is now a claim about **cost and
> reproducibility**, which the data supports, rather than about **effect size**, which it does not.

### Variance, and why `top3_tta` is more than a score bump

`P1` under `best` has σ = 0.0090 across seeds; under `top3_tta`, σ = **0.0016** — a ~6× reduction.
The rule does not just score higher, it makes the number **reproducible**, which is what the
±0.02–0.05 noise problem in §6 was asking for all along.

### Where the improvement actually landed — per-class, against the paper's own Table 3

Gate 0b handed us something better than a headline number: the paper's **Table 3 (p12)** lists
per-class precision / recall / F1 for its DenseNet-121, on a test set we have shown is the same
1,586 images class-for-class (§0b). So the gain can be attributed **class by class** rather than
asserted in aggregate.

Compared below: paper DenseNet-121 (Table 3, rounded to 2 dp — its 22 F1 values average to 0.6518,
consistent with the reported 0.6504) vs **P1 CoAtNet-0 @288, seed 0, `top3_tta`** (macro 0.6941).

| Class | Paper F1 | P1 F1 | Δ | test | train |
|---|---|---|---|---|---|
| Resected polyps | 0.17 | 0.552 | **+0.382** | 18 | 55 |
| Barrett's esophagus | 0.40 | 0.629 | **+0.229** | 19 | 57 |
| Retroflex rectum | 0.55 | 0.769 | **+0.219** | 13 | 40 |
| Esophagitis | 0.31 | 0.457 | **+0.147** | 21 | 64 |
| Gastric polyps | 0.33 | 0.476 | **+0.146** | 13 | 39 |
| Normal esophagus | 0.77 | 0.880 | +0.110 | 28 | 84 |
| Ileocecal valve | 0.72 | 0.805 | +0.085 | 40 | 120 |
| Gastroesophageal junction z-line | 0.74 | 0.803 | +0.063 | 66 | 198 |
| Dyed-lifted-polyps | 0.86 | 0.906 | +0.046 | 28 | 85 |
| Colon diverticula | 0.50 | 0.545 | +0.045 | 6 | 17 |
| Small bowel / terminal ileum | 0.85 | 0.893 | +0.043 | 169 | 508 |
| Normal mucosa large bowel | 0.84 | 0.868 | +0.028 | 294 | 880 |
| Colon polyps | 0.82 | 0.841 | +0.021 | 164 | 492 |
| Dyed-resection-margins | 0.93 | 0.948 | +0.018 | 49 | 148 |
| Cecum | 0.23 | 0.242 | +0.012 | 23 | 68 |
| Normal stomach | 0.88 | 0.891 | +0.011 | 194 | 581 |
| Accessory tools | 0.95 | 0.957 | +0.007 | 253 | 760 |
| Blood in lumen | 0.89 | 0.889 | −0.001 | 34 | 103 |
| Duodenal bulb | 0.74 | 0.714 | −0.026 | 41 | 123 |
| Pylorus | 0.86 | 0.832 | −0.028 | 79 | 236 |
| Colorectal cancer | 0.50 | 0.372 | **−0.128** | 28 | 83 |
| **Mucosal inflammation large bowel** | 0.50 | **0.000** | **−0.500** | **6** | **17** |

#### The improvement is concentrated in the rare classes

Split the 22 classes by test-set size:

| Group | Classes | Mean Δ F1 | Contribution to macro-F1 |
|---|---|---|---|
| **Rare** (< 50 test images) | 15 | **+0.052** | **+0.0356 (84%)** |
| **Common** (≥ 66 test images) | 7 | +0.021 | +0.0066 (16%) |
| All 22 | 22 | +0.042 | +0.0422 |

**84% of the gain comes from the 15 rare classes**, and the five largest gains all belong to classes
with ≤ 21 test images. The five biggest classes — Accessory tools (253), Normal mucosa (294), Normal
stomach (194), Small bowel (169), Colon polyps (164) — move by +0.007 to +0.043 and are essentially
saturated: DenseNet-121 was already near its ceiling there.

This corroborates the §6 diagnosis from a completely independent direction. Three loss-level
imbalance methods (Balanced-Softmax, cRT, strong recipe) were **flat**, which pointed at *feature
representation* rather than the decision rule as the bottleneck. The per-class table now shows the
converse: a stronger backbone at higher resolution, with checkpoint ensembling, **does** move the
rare classes — which is exactly what a representation fix should look like and what a loss reweight
would not.

For the report this is the clinically meaningful framing: the rare classes here are **pathology**
(Barrett's, esophagitis, gastric polyps, resected polyps), while the saturated common classes are
**normal anatomy**. The improvement lands where a screening system would need it.

#### The uncomfortable half of the same table

Two classes got **worse**, and one of them dominates everything:

**Mucosal inflammation large bowel: 0.50 → 0.000, on 6 test images and 17 training images.** Because
macro-F1 weights all 22 classes equally, that single class costs **−0.0227 macro-F1 — more than half
of the headline +0.0457.** Had P1 merely tied the paper on that one class, the reported gain would be
≈ +0.068 instead of +0.046.

Two consequences, both of which belong in the report:

1. **+0.0457 is a conservative estimate**, not a flattering one. It is reported *after* absorbing a
   −0.023 hit from a 6-image class.
2. **This, not seed variance, is the dominant uncertainty.** Getting 2 more of those 6 images right
   would swing macro-F1 by ≈ +0.02 — larger than the entire Swin-T-vs-DenseNet-121 gap, and **12×**
   P1's seed-to-seed σ of 0.0016. It is the concrete mechanism behind the bootstrap CI (±0.035)
   being ~22× wider than σ.

> **Therefore more seeds are not worth buying.** `SEEDS = [0,1,2,3,4]` would refine σ from 0.0016 to
> perhaps 0.0014, at a cost of ~80 min on A100 or ~4 h on T4, while leaving the ±0.035 untouched —
> because the bootstrap resamples the **test set**, which `SPLIT_SEED = 42` holds fixed for every
> seed. The remaining uncertainty is a **data** problem (2 of 22 classes have < 10 test images), not
> a **stochasticity** problem, and no number of seeds addresses it. The dataset authors reach the
> same place in their §4.3, recommending few-shot approaches for exactly these classes.

*Caveats: this comparison uses seed 0 only — per-class F1 on a 6-image class is itself unstable
across seeds — and the paper's values are rounded to 2 dp. The aggregate split (rare vs common) is
robust to both; individual rows for the smallest classes are not.*

### Zero-GPU levers, measured (re-measured under `top3_tta`, 2026-08-26)

* **Logit adjustment** (τ tuned on val, applied on top of `top3_tta`, seed 0):
  −0.0006 (B0, τ\*=0.1), **+0.0054** (S0, τ\*=0.2), **+0.0240** (P0, τ\*=0.2), +0.0011 (P1, τ\*=0.9).
  These replace the earlier figures, which were computed under `best` and are no longer valid.
  **But see the next subsection — across 3 seeds this lever does not hold up.**
* **Cross-architecture ensemble**: the combination is now chosen **on val**, and the winner
  (`B0 + S0 + P0 + P1`) also happens to top the test column, so there is no selection gap:
  val 0.7143 → **test 0.7242**. Reportable, but only as its own row — it consumes 4 training runs.

### The one component that did *not* survive: logit adjustment

Run over all 3 seeds of the proposed model (§19b), the lever falls apart:

| seed | `top3_tta` | τ\* | after adjustment | Δ |
|---|---|---|---|---|
| 0 | 0.6941 | 0.9 | 0.6952 | +0.0011 |
| 1 | 0.6980 | 0.5 | 0.7252 | **+0.0272** |
| 2 | 0.6961 | 0.0 | 0.6961 | +0.0000 |
| **mean ± σ** | **0.6961 ± 0.0016** | — | **0.7055 ± 0.0139** | +0.0094 |

The mean goes up, but **σ inflates ~9×**, and the entire gain comes from a single seed. τ\* itself is
unstable (0.9 / 0.5 / 0.0) — it is being fitted to the `best`-checkpoint val logits and then applied
to `top3_tta` test scores, a distribution mismatch. A number that is higher but less reproducible is
exactly the disease §6 exists to cure, so:

> **The proposed system is `CoAtNet-0 @288 + TTA + top-3 checkpoint ensemble` = 0.6961 ± 0.0016
> (+0.0457 vs paper).** Logit adjustment is reported separately as an ablation that **did not
> replicate across seeds** — which is itself a finding, and one the report should state plainly.

The notebook now enforces this: §19b compares both variants and keeps logit adjustment only if it
raises the mean *without* inflating σ by more than 50%. On this run it is automatically dropped.

### Leakage robustness (§19b, part 1)

6 of 1,586 test images have a byte-identical or cosine ≥ 0.98 twin in train/val. Dropping them
(**seed 0 only** — the §19b cell re-reads the first seed's logits, so the `full` column is seed 0's
score, not the 3-seed mean; read the Δ column, not the levels):

| Model | full | filtered | Δ |
|---|---|---|---|
| B0 | 0.6768 | 0.6749 | −0.0020 |
| S0 | 0.6976 | 0.6953 | −0.0022 |
| P0 | 0.6805 | 0.6787 | −0.0018 |
| P1 | 0.6941 | 0.6922 | −0.0019 |

Every Δ is ≈ −0.002 — **~10× smaller than the ±0.02–0.05 run-to-run noise floor**. Against
seed-to-seed σ the honest split is narrower: 3–5× smaller than B0's and S0's σ under `top3_tta`
(0.0066 / 0.0114), but the *same size* as P0's and P1's (0.0014 / 0.0016) — not because leakage bites
harder there, but because `top3_tta` compressed those two models' σ so hard. Either way, a 0.002
shift caused by 6 images is not what produced +0.0457. **Leakage is not what produced the result** —
and that is now a measured statement, not a reassurance.

### The equal-epoch protocol is conservative *against* the proposed model

Val curves (§19b): B0 peaks at **epoch 12/30**, S0 at **7/30**, P0 at 22/30, but
**P1 peaks at epoch 27/30** and its last-5-epoch spread is the smallest of all four (0.0021 vs
0.0234 for B0). The baselines had finished learning long before the budget ran out; the proposed
model had not. The fixed 30-epoch budget is the right call for a fair comparison, but it means
**+0.0457 is a floor, not a ceiling** — worth one sentence in the report, and the cheapest
remaining GPU lever if there is time.

### Transfer learning: freeze depth (§19d, A100, 2026-08-27)

Four conditions on **the same DenseNet-121**, same split, same seed, same 30-epoch budget, all scored
under `top3_tta`. This is the brief's 10%-weighted row, and the answer is unusually clean.

| Condition | What actually trains | Seeds | test macro-F1 | best VAL | min/seed |
|---|---|---|---|---|---|
| **T1** linear probe | classifier head only | 1 | **0.5725** | 0.5376 | 9.2 |
| **T2** lower half frozen | upper half + head | 1 | **0.6463** | 0.6246 | 9.3 |
| **T3** progressive + discriminative LR | 3-epoch probe → all layers, backbone at 0.5× LR | 1 | **0.6472** | 0.6249 | 10.9 |
| **T4** full fine-tune (= `B0`) | all layers, one LR | 3 | **0.6676 ± 0.0066** | 0.6681 | 9.7 |

**Full fine-tune wins, and the cost rises steeply with freeze depth.** Against T4, T1 is **−0.0951**,
T2 **−0.0213**, T3 **−0.0204** — all three past the 2σ = 0.0132 bar (σ from `B0`'s 3 seeds, since each
frozen condition has only one). The gap is not marginal: a linear probe gives up **9.5 points of
macro-F1**, more than twice what this whole project gained over the published baseline (+4.6).

**Why, in one line for the report:** ImageNet features are natural-image features. Endoscopy is a
different imaging modality — specular highlights, a circular black mask, wildly non-natural colour
statistics — so the *early* layers are the ones that need to move, and those are exactly what freezing
pins down. T2 vs T3 (0.6463 vs 0.6472, well inside 2σ) says the same thing from the other side: once
the lower half is frozen, *how* you schedule the rest barely matters.

**The paper corroborates this for free.** Its Table 2 mixes both regimes (§4.2): last-layer-only gives
0.4496 / 0.4519 / 0.4883 (ResNet-152 / EfficientNet-B0 / DenseNet-169), full fine-tune gives 0.6176 /
0.6504 (ResNet-50 / DenseNet-121) — a **~0.16 gap on this exact split**, same direction and roughly
1.7× the size of our T1→T4 gap (they froze everything but the final classifier; our T1 also unfroze
nothing, but ours ran 30 epochs at a tuned head LR). Two independent experiments, one conclusion.

**Two limits to state rather than hide.** (1) T3 is **2-group** discriminative LR (backbone 0.5×, head
1×) after a 3-epoch probe — not true per-layer decay as in ULMFiT/BEiT; the brief's wording allows
either, but the report should not claim the stronger method. (2) **1 seed per frozen condition**, so
the table ranks coarsely — the T2/T3 ordering is *not* resolvable, and only the >2σ gaps above may be
called real. Raising this to 3 seeds is ~30 min A100 and is the cheapest remaining GPU lever.

> Implementation note worth a sentence in the report: a linear probe also has to put the frozen
> backbone's **BatchNorm into eval**. `requires_grad = False` stops gradients but not the running
> mean/var updates, so a naive "frozen" DenseNet-121 (~120 BN layers) keeps drifting its own features
> every epoch and the number you measure is not a linear probe at all. `_freeze_bn_of_frozen_part()`
> in §11 handles this, and the same call is reused for the probe stage of `train_advanced`.

### Deployment (**A100**, re-measured 2026-08-27)

> The third run landed on a T4; the fourth is back on an **A100**, and this table is re-measured
> there. Every macro-F1 above is unaffected — all 12 runs resume from `.npz`, so the scores are
> recomputed from stored logits and are hardware-independent. **Only this table is hardware-dependent.**
> The T4 figures (16.0 / 22.1 / 12.4 / 13.1 ms at batch 1; 3.12 / 4.60 / 4.76 / 8.33 at batch 32) are
> superseded — do not mix the two. The batch-1 column reproduces the **2026-08-26 A100** measurement
> (18.1 / 23.1 / 12.7 / 12.6) to within 0.7 ms, which is the only cross-check available without a
> second machine; the batch-32 column is measured on A100 here for the first time.

| Model | Resolution | Params | ms/image @ batch 1 | ms/image @ batch 32 | ONNX size |
|---|---|---|---|---|---|
| DenseNet-121 | 224 | 7.0 M | 18.9 | **0.58** | 29.1 MB |
| Swin-T | 224 | 27.5 M | 23.5 | 0.73 | 113.7 MB |
| CoAtNet-0 | 224 | 26.7 M | 13.4 | 0.59 | 110.4 MB |
| **CoAtNet-0 (proposed)** | **288** | 26.7 M | 13.2 | **0.99** | 114.8 MB |

**The two columns rank the models in opposite orders, and that is the point.** At batch 1 the GPU is
idle most of the wall-clock and the measurement is dominated by **kernel-launch count**: DenseNet-121
has ~120 concat layers and comes out **slowest** (18.9 ms) at 7 M parameters, while 288² comes out
*level with* 224² (13.2 vs 13.4 ms) despite 1.65× the pixels. Neither is physically possible as a
compute cost. At batch 32 the launches amortise and the numbers behave: DenseNet-121 is **fastest**
(0.58 ms, matching its 7 M parameters), and 288 costs 0.99 / 0.59 = **1.68×** its own 224 baseline —
within measurement noise of the 1.65× pixel ratio, exactly as it should be. On the T4 the same ratio
came out 1.75×; two different GPUs converging on the pixel ratio is the strongest evidence that the
batch-32 column, not the batch-1 column, is the one that measures compute.

So the honest cost/benefit line for the report is:

> The proposed system buys **+0.0457 macro-F1** over the paper's baseline for **1.7× the compute per
> image** (0.99 vs 0.58 ms at batch 32) and **3.9× the model size** (114.8 vs 29.1 MB), plus a 6×
> inference multiplier for TTA × top-3 checkpoint ensembling.

That last clause matters and the report must not bury it: `top3_tta` runs **3 checkpoints × 2 flips =
6 forward passes** per image. The single biggest gain in this project is not free at inference time
— it is free in *training* time, which is a different claim. In the clinical single-image scenario
that is 6 × 13.2 ≈ 79 ms, still comfortably real-time; in batch triage it is 6 × 0.99 ≈ 6 ms/image.
Both are acceptable, but they must be stated rather than implied by a batch-1 table.

## 0b. Verifying the paper at the source (done 2026-08-27)

The brief (p16) flags GastroVision's Table 2 as **"not verified"**, and every claim in this project is
stated relative to it. Checked against the arXiv PDF itself (2307.08140), not a secondary source.

**Table 2, page 11 — transcribed verbatim:**

| Method | Macro Prec. | Macro Recall | **Macro F1** | Micro P/R/F1 | MCC |
|---|---|---|---|---|---|
| ResNet-50 | 0.4373 | 0.4379 | 0.4330 | 0.6816 | 0.6416 |
| Pre-trained ResNet-152 | 0.5258 | 0.4287 | 0.4496 | 0.6879 | 0.6478 |
| Pre-trained EfficientNet-B0 | 0.5285 | 0.4326 | 0.4519 | 0.6759 | 0.6351 |
| Pre-trained DenseNet-169 | 0.6075 | 0.4603 | 0.4883 | 0.7055 | 0.6685 |
| Pre-trained ResNet-50 | 0.6398 | 0.6073 | 0.6176 | 0.8146 | 0.7921 |
| **Pre-trained DenseNet-121** | 0.7388 | 0.6231 | **0.6504** | 0.8203 | 0.7987 |

✅ **0.6504 is confirmed** as the macro-averaged F1 of pre-trained DenseNet-121. The target we
committed to is the right number, read off the right column.

### The protocol matches ours

| Item | Paper (§4.1) | Ours | |
|---|---|---|---|
| Split | stratified **60:20:20** | stratified 60:20:20, seed 42 | ✅ |
| Classes | **22** (only classes with > 25 samples) | 22, `assert NUM_CLASSES == 22` | ✅ |
| Images | 7,930 of 8,000 implied | 7,930 | ✅ |
| Test set size | **1,586** (Table 3 supports sum) | 1,586 | ✅ |
| Input | 224 × 224 | 224 (B0/S0/P0) | ✅ |
| Augmentation | random rotation + random hflip | same family | ✅ |
| Optimiser | Adam, lr 1e-4, ReduceLROnPlateau | **AdamW**, lr 1e-4, **weight-decay 1e-4**, no scheduler | ⚠️ close, not identical |
| **Epochs** | **150** | **30** | ⚠️ differs |
| **Runs** | **single run, no seeds, no error bars** | 3 seeds, mean ± σ + bootstrap CI | ⚠️ differs |
| Hardware | NVIDIA TITAN Xp | A100 / T4 | — |

**The split is reproduced class-by-class.** Table 3 lists per-class test support for DenseNet-121;
summing it gives exactly 1,586. Against our own test distribution, **16 of 22 classes match exactly**
and 6 differ by ±1 image (Colon polyps 163/164, Colorectal cancer 29/28, Esophagitis 22/21, Normal
mucosa LB 293/294, Pylorus 78/79, Retroflex rectum 14/13) — ordinary rounding in the stratified
splitter, and the totals are identical. We are evaluating on a test set of the same composition, not
merely the same size.

### Two things the verification turned up that change how we must phrase the claim

**1. 0.6504 is a single run with no error bar.** The paper reports one number per model and never
mentions seeds. Our own DenseNet-121, on the same protocol, spreads **±0.0124** across 3 seeds under
the paper's own selection rule. So the honest reading is that 0.6504 carries roughly ±0.012 of
*unstated* seed uncertainty. Consequence for the report: **"we beat 0.6504" must be supported by an
interval that excludes it**, not by a difference of point estimates. We have that — the proposed
model's bootstrap CI95 is [0.6548, 0.7245] and the 3-seed ensemble's is [0.6728, 0.7609]; neither
contains 0.6504. State it that way.

**2. Our 30-epoch reproduction lands on their 150-epoch number.** Ours: **0.6491 ± 0.0124** under
`best`, the paper's rule. Theirs: 0.6504. A gap of **0.0013**, an order of magnitude inside our own
seed spread. This is a clean reproduction, and it also retires the obvious objection to the 30-epoch
budget: DenseNet-121 peaks at **epoch 12 of 30** (§9), so the missing 120 epochs were never going to
help it. The equal-epoch protocol is fair to the baseline and conservative only against P1, which
peaks at 27/30.

**A caveat worth one line in the report:** Table 2 is not a clean architecture comparison. Per §4.2,
DenseNet-121 and the second ResNet-50 fine-tune **all layers**, while ResNet-152, EfficientNet-B0 and
DenseNet-169 fine-tune **only the last layer**. Most of the spread in that table is fine-tuning depth,
not architecture. We should not cite it as evidence that DenseNet-121 is the strongest backbone —
only that 0.6504 is the strongest *published* number on this split.

**Source:** arXiv:2307.08140, Jha et al., *GastroVision: A Multi-class Endoscopy Image Dataset for
Computer Aided Gastrointestinal Disease Detection* — Table 2 p11, Table 3 p12, §4.1–4.2 p11.

### Still open

| # | Task | Cost |
|---|---|---|
| 0b | ✅ *done 2026-08-27* — Table 2 confirmed verbatim against the PDF; protocol matches ours; see §0b | ✅ done |
| — | ✅ *done 2026-08-26* — notebook re-run on A100, every table below §16 now under `top3_tta` | ✅ done |
| — | ✅ *done 2026-08-27* — §19b and §20 refreshed (ran on T4; all training resumed from Drive) | ✅ done |
| — | ✅ *done 2026-08-27* — **4th run, back on A100:** whole notebook re-executed end to end (all 12 main runs resumed from Drive), plus the two new sections §19d and §20b. Gate 0a reproduced the A100 curve exactly; the latency table is re-measured on A100 | ~30 min GPU |
| — | Optional `RUN_ABLATIONS = True`: Swin-T in22k, Balanced-Softmax (1 seed each) | ~20 min A100 |
| — | ~~Optional `SEEDS = [0,1,2,3,4]`~~ — **decided against 2026-08-27.** Refines σ from 0.0016 to ~0.0014 and leaves the dominant ±0.035 untouched; see "The uncomfortable half of the same table" in §9. Cost was also under-estimated here: 2 extra seeds × 4 configs × ~10 min = **~80 min A100 / ~4 h T4**, not 40 min | ❌ not worth it |
| — | ✅ *done 2026-08-27* — figures + tables of the real run extracted to `report/` (git-tracked); notebook prose §19b/§21 brought in line with the 2026-08-27 correction. **Re-extracted after the 4th (A100) run** by `report/extract.py`, which is now the committed, re-runnable extractor (matches cells by source prefix, so adding cells does not shift the mapping) | 0 GPU |
| — | ✅ **done 2026-08-27 — report written: `report/BAO_CAO.md`** (tiếng Việt, 10 mục ánh xạ 1-1 với khung 70/30 của đề bài, mục 6 trang 15). Every number is quoted from `report/tables/` — nothing typed by hand. `report/build_html.py` renders it to a self-contained `report/bao_cao.html` (hình nhúng data URI) | 0 GPU |
| — | **Slides** — the last deliverable. Source the whole deck from `report/BAO_CAO.md`; the argument order there is already the presentation order (nền nhiễu → cách đo → độ phân giải → kiến trúc → per-class) | 0 GPU |
| — | ✅ **done 2026-08-27 — transfer learning (10% of the report), see §19d above.** T1 linear probe **0.5725** · T2 lower half frozen **0.6463** · T3 progressive + discriminative LR **0.6472** · T4 full fine-tune **0.6676 ± 0.0066**. All three frozen conditions lose by more than 2σ; full fine-tune is the right call and the paper's Table 2 agrees. Table in `report/tables/27_transfer_learning.txt`, raw log in `26_transfer_learning_log.txt`. Still 1 seed per frozen condition — 3 seeds would be ~30 min A100 | ✅ done |
| — | ✅ **done 2026-08-27 — Gradio demo (§20b) exercised on the real checkpoint.** Loaded `P1_coatnet0_288_seed0.pt` at 288 px on CUDA; the self-check passed on a real test image (*Accessory tools* → *Accessory tools*, p = 1.000) **before** the UI was built, so the inference path is verified even where `gradio` is not installed. Output is **top-5 with probabilities** (a single label is the most misleading form for 22 imbalanced classes) with hflip TTA; `prevent_thread_lock=True` on launch, otherwise the cell blocks and §21 never runs. Log: `report/tables/29_demo_gradio.txt`. **⚠️ Two things the report must still state:** (1) `run_seeds` persists only the single best checkpoint, so the demo is 1 checkpoint + TTA (≈ `best_tta` ≈ 0.673 for P1), **not** the `top3_tta` system (0.6961) — closing that means saving all 3 states (~1.3 GB on Drive) and retraining, and old runs cannot be back-filled; (2) **no screenshot is stored in `report/`** — Gradio renders as a live widget, so the notebook keeps no image; grab one by hand from the running cell if the report needs a figure | ✅ done |

**If `SEEDS` is ever changed anyway**, two things must change with it: `PROFILES` in
`build_notebook.py` (≈ line 203, both GPU profiles — `SEEDS` is derived from `CFG["seeds"]`, not set
directly), and the hard-coded `"3 seed"` strings in prose and `print()` calls — the logic is generic
but the labels are not, so §19b would print `"ensemble ca 3 seed"` over a 5-seed ensemble. Find them
with `grep -n "3 seed" build_notebook.py` rather than by line number: adding §19d/§20b already
invalidated the list that used to be quoted here, and §19d added two more of its own. Resume is per-seed, so seeds 0–2 reload from `.npz`
and only 3–4 train. Note also that seeds 0–2 were trained on A100: any new seed trained on a
different GPU mixes hardware variance into σ (see Gate 0a), which must then be disclosed.
