# Final Project — GastroVision GI Endoscopy Classification

**Course:** AIN501 · Artificial Intelligence (Deep Learning) · MSE FSB
**Team:** 2 members · **Goal:** beat a published baseline on GastroVision — provably

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

### Structure: 2 baselines + 1 proposed model

With a 2-person team we run **two baselines, not three** — and the **hybrid moves from
"baseline" to "proposed model"**. This actually matches the assignment framing better than
the old 3-baseline plan: the brief's whiteboard proposition 4 says
*"CNN (1 part) + Transformer (1 part) + attention — **the destination is the hybrid
architecture, that is exactly where the 'improvement' is placed**"*. So the hybrid belongs in
the proposed row, measured against both baselines.

| Role | Model | Family | Number to compare against |
|---|---|---|---|
| **Reference baseline** (mandatory) | DenseNet-121 | CNN | **0.6504** — paper Table 2 |
| **New baseline** (our contribution) | Swin-T | Transformer | — (the paper has *no* Transformer) |
| **Proposed model** (**mandatory** — see below) | **CoAtNet-0 @288 + modern training recipe + top-3 ckpt ensemble + logit adjustment** (`P2`) — a published pretrained hybrid, *not* a hand-rolled conv stem (see §5) | CNN + attention | ✅ beats both: **0.7441 ± 0.0088** vs B0 0.6780 / S0 0.6813 |

> ⚠️ **The proposed model changed once, and this row is the current one.** Until `SESSION = 1`
> (2026-08-31) it was `P1` = CoAtNet-0 @288 + TTA + top-3, quoted at 0.6961 ± 0.0016 on the A100
> round. Two things then moved: the A100 configs were **retrained** on T4 rather than resumed, so
> every §9 number was replaced *and* `SELECTION_RULE` flipped `top3_tta` → **`top3`** on its own
> vote; and `P2` (the same backbone under a modern training recipe) cleared the decision gate by 2×.
> All numbers on this page are now `top3`, T4 round. Provenance: `RESULTS.md` §10.9, §10.10.

**The proposed model is required, not a bonus.** Because the hybrid is the whiteboard's
*destination* and the place the improvement is meant to live, skipping it means the report has
no improvement claim at all — only two reproduced baselines. Compute is not the binding
constraint either: the brief (p12) measures GastroVision at **~0.4 h/run**, and (p3, criterion 6)
treats **≥ 15 experiments + 3 seeds** as affordable at that cost. ⚠️ The brief budgets that
against a **30 h/week Kaggle** quota, but our notebook is **Colab-first** (§5): Colab Free has no
fixed weekly quota — it has a tighter, less predictable daily GPU allowance that can be cut
mid-session. So the headroom is real but not bankable; **Kaggle is the fallback** when Colab
throttles, and checkpoints already go to Drive for exactly that reason. Against either platform
the whole Gate-0 plan is **~3 h of T4** and the hybrid ~1.5 h. The binding constraint is
*people-time*, so the hybrid is scheduled (§4), not left to whatever compute is left over.

**Why Swin-T and not ViT-S:** lecture 8 (p35) states ViT's weakness itself — cost grows with
resolution, resolution is hard to change — and p36–40 teach how Swin fixes it with
W-MSA / SW-MSA / Patch Merging. For ~8k medical images, Swin's local-window + hierarchical
inductive bias is the right choice, and the motivation section can cite the course slides
directly. DeiT (slide p41, "data-efficient") was held as the fallback in case Swin overfit —
**it did not, so DeiT was never run**: Swin-T peaks at val epoch 7/30 with a last-5-epoch spread
of 0.0102. The unused `build_deit_s` builder has since been removed (`RESULTS.md` §4).

**Why the Transformer arm is now the priority, not a side quest:** our own ablation
(`RESULTS.md`) showed every imbalance method is flat — the bottleneck is the **feature
representation**, not the loss or the classifier boundary. A stronger, better-pretrained
backbone is the main untried lever, so the Swin-T track ran **in parallel**, not at the end.
*(Outcome: the lever turned out smaller than hoped and not separable at 3 seeds — see §4,
"Decide after `S0` lands".)*

### Goal (two tiers — and where the numbers come from)

**Where ≥ 0.75 came from:** it is **not** a number we guessed. The assignment brief
(`../mse-dl-de-bai-vuot-baseline.pdf`, p12 comparison table) sets GastroVision's target at
**≥ 0.75, headroom ~10 points** over DenseNet-121's 0.650. So 0.75 is the brief's own
recommended target, and we are the ones lowering it.

**And we must say so plainly:** the same brief (p2, topic-selection criterion 3) states
*"headroom 5–15 points — **below 3 points you cannot demonstrate statistical significance**"*.
Our revised primary target ("beat 0.6504 significantly", possibly by only ~1 point) therefore
sits **below the brief's own threshold for a meaningful result**. We accept that trade
deliberately, for one reason: with B0 = 0.6516 and all three imbalance levers measured flat,
a committed ≥ 0.75 would be a promise with no evidence behind it. The honest position is a
floor we can defend plus the brief's target kept as the real goal:

- **Primary (committed floor):** beat 0.6504 **with statistical significance** — our 3-seed
  mean ± std / bootstrap CI must not overlap the baseline — **and** answer
  *"what actually moves macro-F1 on this long-tail dataset"* with an ablation table that
  **reports the negative results too**. ⚠️ Note in the report that a ~1-point win is below
  the brief's 3-point significance floor, so this tier alone is **not** a sufficient result.
- **Stretch (the brief's actual target — this is what we aim at):** macro-F1 **0.72–0.75**,
  via Swin-T, the hybrid, and/or the resolution lever.

> ✅ **Outcome — the stretch tier was reached, and the worry above did not materialise.** Final:
> **0.7441 ± 0.0088**, i.e. **+0.0937 = 9.4 points** over 0.6504, with the bootstrap CI
> [0.6986, 0.7736] **not containing** the baseline. That lands inside the brief's own 5–15 point
> headroom band and well above its 3-point significance floor, so the caveat above (*"a ~1-point
> win is below the brief's floor"*) **no longer applies** — it is kept because it records the
> position we were honestly willing to defend before the numbers existed.
>
> ⚠️ **But not via the levers this section names.** Neither Swin-T, nor the hybrid on its own, nor
> resolution carried it: they measure **+0.0034** (architecture alone) and **+0.0041** (224 → 288),
> both below the ±0.035 resolution of the measurement. What paid was the **training recipe**
> (+0.0443) and then the **measurement** (logit adjustment +0.0143, checkpoint rule +0.0094 — both
> at **0 extra epochs**). That inversion is the report's main finding, not a footnote to it.

Everything is reported on the **original stratified 60:20:20 split**, as **mean ± std over
≥ 3 seeds**, with a **per-class F1 table** and confusion matrix.

> A report that proves *"three standard imbalance methods do not work here, and here is the
> evidence"* scores higher than one that promises 0.75 and lands at 0.68 unexplained.

### Published baseline (paper Table 2 — arXiv 2307.08140)

> ✅ **Verified at the source (Gate-0 item 0b, 2026-08-27).** The brief (p16) listed
> **GastroVision Table 2 among its "not verified" figures**, so we read the arXiv 2307.08140 PDF
> itself: **0.6504 is confirmed** as pre-trained DenseNet-121's macro-F1. All six rows transcribed
> verbatim, plus the protocol comparison and a class-by-class split check, are in `RESULTS.md` §0b.
> Two things that check turned up and that change the phrasing: 0.6504 is a **single run with no
> error bar**, and Table 2 **mixes fine-tuning depths** (§4.2), so it is the strongest *published*
> number on this split — not evidence that DenseNet-121 is the strongest backbone.
> We also **reproduced DenseNet-121 ourselves** on the same split: **0.6686 ± 0.0234** under the
> paper's own rule (`best`), i.e. **+0.0182 = 0.78 σ** — inside one σ, but a wide band. See the
> "Reproduction status" note below for why the earlier, much tighter 0.6491 ± 0.0124 no longer
> stands (`RESULTS.md` §10.9 finding 5).

| Model | Macro-F1 | Micro-F1 | MCC |
|---|---|---|---|
| **DenseNet-121 (pretrained)** | **0.6504** | 0.8203 | 0.7987 |
| ResNet-50 (pretrained) | 0.6176 | 0.8146 | 0.7921 |
| DenseNet-169 (pretrained) | 0.4883 | 0.7055 | 0.6685 |
| EfficientNet-B0 (pretrained) | 0.4519 | 0.6759 | 0.6351 |
| ResNet-152 (pretrained) | 0.4496 | 0.6879 | 0.6478 |
| ResNet-50 (from scratch) | 0.4330 | 0.6816 | 0.6416 |

**Note:** the paper baseline has **6 CNNs and zero Transformers** — adding Swin-T into this
exact table gives a clean **CNN vs Transformer** comparison, almost for free since the CNN
rows already exist.

### Links
- Dataset: [GitHub](https://github.com/DebeshJha/GastroVision) · [OSF](https://osf.io/84e7f/) · paper arXiv 2307.08140
- Original split: **stratified 60:20:20** — reproduce with a **fixed seed** (no ad-hoc re-splitting → avoids leakage).

### Baseline sources & how to reproduce each (proof the baselines are valid)

Be explicit about **two provenance categories** when presenting — it's what makes the comparison defensible:

- **Reference baseline (has a published number to match):** taken from the GastroVision paper, Table 2.
  Reproducing it on our split *proves our pipeline is correct* before we compare anything.
- **New baseline (no published number — our contribution):** Swin-T is **not** in the paper. Here
  "reproduce" means *train under the identical protocol*, not match a target number.

| Baseline | Family | Architecture source | Pretrained weights | Published macro-F1 to match |
|---|---|---|---|---|
| **DenseNet-121** | CNN | Huang et al., CVPR 2017 (arXiv:1608.06993) — the model used by the GastroVision paper | torchvision `DenseNet121_Weights.IMAGENET1K_V1` | **0.6504** (paper Table 2) |
| **Swin-T** | Transformer | Liu et al., ICCV 2021 (arXiv:2103.14030) | `timm` · `swin_tiny_patch4_window7_224` (ImageNet-1k) | — (new; no paper number) |
| *ViT-S / DeiT-S — backup, **never needed*** | Transformer | Dosovitskiy et al., ICLR 2021 (arXiv:2010.11929) | `timm` · `vit_small_patch16_224` (ImageNet-21k→1k) | — (Swin-T did not overfit, so neither was run) |
| **CoAtNet-0** — *proposed row, not a baseline* | Hybrid | Dai et al., NeurIPS 2021 (arXiv:2106.04803) | `timm` · `coatnet_0_rw_224` (ImageNet-1k) | — (proposed model; must beat both baselines) |

Weights come from [`timm`](https://github.com/huggingface/pytorch-image-models) (Wightman, *PyTorch Image Models*) and torchvision — pretrained on ImageNet, then we replace the classifier head with a fresh `NUM_CLASSES` linear layer.

**Identical reproduce protocol for both baselines** (this equal footing is what makes the comparison valid):
- **Same data:** 22 classes (paper rule *">25 samples"*), stratified **60:20:20**, `SPLIT_SEED=42` — fixed for both members.
- **Input** 224×224, ImageNet mean/std normalization; pretrained backbone + fresh linear head.
- **AdamW** (lr `1e-4`, weight-decay `1e-4`), **30 epochs**, batch 32, AMP on GPU, keep the **best-val-macro-F1** checkpoint.
- **≥ 3 seeds → report mean ± std**; primary metric **macro-F1** (+ per-class F1 + confusion matrix).
- One driver for both architectures: `run_seeds(build_fn, tag=...)` in the shared notebook.

> **Reproduction status — ✅ settled, but looser than the first round suggested.** `B0` =
> **0.6686 ± 0.0234** over 3 seeds under the paper's own rule (single best-val checkpoint, no TTA)
> vs published **0.6504** — a gap of **+0.0182**, i.e. **0.78 σ**. Inside one standard deviation, so
> it reproduces; but ±0.023 is a wide band and the report says so explicitly rather than calling it
> a match (`report/BAO_CAO.md` §1.4). **Do not** quote the `top3` number (0.6780) as "reproducing
> the paper" — the paper did not use checkpoint ensembling.
>
> ⚠️ *Superseded number, kept because it is the cleanest example of the hardware effect:* the A100
> round read **0.6491 ± 0.0124** here, a gap of 0.0013 that looked like an almost exact match. Same
> code, same split, same seeds — retrained on T4 it became 0.6686 ± 0.0234. `RESULTS.md` §10.9
> finding 5; magnitude measured in `report/tables-offline/30_*`.
>
> *History, kept because it is why Gate 0 exists:* two earlier single-seed `B0` runs gave **0.676**
> and **0.6516** — a 0.024 spread that could not be seed variance at a fixed seed.
>
> ⚠️ **Do not call that spread "seed noise" yet.** The two runs were *believed* to use the same
> seed and split, and `set_seed()` already covers `random` / `numpy` / `torch` / `cuda` +
> `cudnn.deterministic=True`. At a genuinely fixed seed, a 0.024 gap is **not** variance across
> seeds — it is either two different configs, or real non-determinism (AMP, plus CUDA kernels
> without deterministic implementations). Running 3 seeds cannot tell those apart, which is why
> Gate 0 starts with a determinism check (§2, item 0a) rather than with more seeds.
>
> → **Answered by 0a: it was "two different configs", not non-determinism.** On one GPU the pipeline
> is bit-reproducible at a fixed seed. But 0a also found that determinism does **not** hold *across*
> GPUs (A100 and T4 give different val curves from the same seed), so the standing rule is that
> **σ must never mix hardware**. `RESULTS.md` §3.

---

## 2. Gate 0 — fix the measurement before running anything new

> ### ✅ Gate 0 is closed (A100 run, 2026-08-26)
> Everything below is kept as the record of *why* each item existed. Outcomes:
> **0a passed on GPU** — two 3-epoch runs identical to 6 decimals with bf16 + TF32, so
> non-determinism was never the cause of the old 0.024 spread. **3a/3b done and decided** —
> the rule **was** **`top3_tta`** (top-3 checkpoint ensemble + horizontal-flip TTA), which lifted every
> architecture by 0.019–0.032 *and* cut seed-to-seed σ on the proposed model from 0.0090 to 0.0016.
> ⚠️ **The rule has since flipped to plain `top3`.** §16 re-votes on whatever numbers the session
> actually has, and on the T4 round it picked `top3` by itself. The *mechanism* (ensemble the top-3
> checkpoints) survived the hardware change; the TTA half did not. Anything below this block that
> still says `top3_tta` is the round-1 record — see the `SESSION = 1` block further down and
> `RESULTS.md` §10.9 finding 5.
> **Item 1 done** — 4 configs × 3 seeds. **0b done too** (2026-08-27): Table 2 confirmed verbatim
> against the arXiv PDF, protocol and split checked class-by-class — `RESULTS.md` §0b. **Every Gate-0
> item is now closed; nothing here is outstanding.**
> Numbers: `RESULTS.md` §9.
>
> One extra finding from 0a that belongs in the report: determinism holds **within** a GPU, not
> **across** GPUs. The same seed and code gave `[0.428608, 0.549646, 0.551532]` on A100 (twice,
> bit-identical) but `[0.430615, 0.540379, 0.541930]` on T4 — so **σ must never mix hardware**.

Our ablation found the **run-to-run spread (~±0.02–0.05) is larger than every effect we measured**
(B3 −0.007, B4 −0.013, B5 −0.017). Until that is fixed, extra runs buy nothing. Six tasks,
~3 h of T4 in total, and **four of the six cost no training at all** — but only if they are done
in the order below (3a before 1; see the ordering rule).

| # | Task | Cost | Why |
|---|---|---|---|
| **0a** | **Determinism check** — run B0 at seed 0 **twice** and diff the test macro-F1 | ~40 min T4 | Must come first. If the two runs differ, the 0.024 spread is non-determinism, not seed variance, and no number of seeds fixes it. The notebook now seeds the DataLoader and sets `use_deterministic_algorithms(warn_only=True)` — this verifies it actually took. → **Outcome:** implemented as a diff of the **whole 3-epoch val curve** (stricter than one final score) and it cost **~5 min**, not 40. Passed on A100; see the cross-GPU caveat above. `RESULTS.md` §3. |
| **0b** | **Verify the paper's Table 2** against arXiv 2307.08140 itself | **0 GPU** | The brief (p16) explicitly lists **GastroVision Table 2 as "not verified"** and requires self-checking *before it goes into the official report*. Our entire committed target is "beat 0.6504" — that number must be confirmed at the source, not inherited. |
| 2b | ✅ **done** — `run_seeds` now persists `y_true`/`y_pred` per seed to `<tag>_seed<s>_preds.npz` | **0 extra epochs** | `evaluate()` already returned both arrays; they were being discarded, so a finished run could not be given an error bar without retraining. |
| **3a** | **De-noise checkpoint selection — the code change.** `train_one` must keep, in one run, everything the three candidate rules need: the **top-3 raw-val states**, the state at the **running 3-epoch-smoothed val argmax** (that epoch need not be in the raw top-3, so it must be tracked separately), and the **per-epoch val-F1 history**. Today it keeps a single `best_state` and only *prints* the rest | **0 epochs** | Best-val on a tiny val set is the second noise source (B5: val 0.689 → test 0.634, a 0.055 gap). **Must land before item 1** — otherwise item 1's six runs are locked to the old rule and have to be repeated (~2.5 h wasted). |
| 1 | **3 seeds for B0 and B5** (6 runs × ~20 min), run **under the 3a-enabled `train_one`** | ~2.5 h T4 | The meaningful experiment, but only *after* 0a. B5 has the highest best-VAL (0.689), so it is a real candidate, not just noise. → **Outcome: superseded.** `B5` (DenseNet @288) was dropped and the run became **4 configs × 3 seeds** (`B0`/`S0`/`P0`/`P1`) — the resolution lever moved onto the proposed backbone as `P1`. `RESULTS.md` §3, §9. |
| 2 | **Bootstrap CI on the test split** — `bootstrap_ci(y_true, y_pred)`, 1000 resamples of the test images | **0 extra epochs — on item-1's runs** | Best value for a 2-person team: every run carries an error bar, so "beats 0.6504 significantly" is claimable without 5 seeds. ⚠️ B0 0.6516 / B5 0.6342 predate 2b and saved **no** `.npz`, so *their* CI is not free — it comes from item 1's re-runs, not from the old numbers. |
| 3b | **Pick the selection rule** — score best-val vs 3-epoch-smoothed vs top-3-ensemble on item 1's saved states | **0 extra epochs** | With 3a in place all three rules are computable from the **same** finished runs, so choosing between them costs no training. |

**Ordering rule:** no new rung on the ladder until items 0a–3b are done, and **3a must land before
item 1** — that ordering is what makes 3b free instead of a second ~2.5 h of training. The existing
seed-0 numbers (B0 0.6516, B5 0.6342) were measured under the old selection rule and are superseded
by item 1's runs either way — a mean ± std must not mix two selection protocols.

---

> **The report is written: [`report/BAO_CAO.md`](report/BAO_CAO.md)** — Vietnamese, 10 sections mapping 1-1
> onto the framework below, every number quoted from `report/tables/`. `python report/build_html.py` renders
> it to a self-contained `report/bao_cao.html`. **Rebased on the T4 run 2026-08-31** (`RESULTS.md`
> §10.9); `python report/check_numbers.py` verifies all 105 quoted numbers against their sources.
> **Both baselines are beaten with paired statistical support** — vs `B0` (the paper's own model)
> at 2/3 seeds plus the 3-seed-ensemble row, vs `S0` (our Transformer baseline) at **3/3** seeds
> (`report/tables-offline/33_vs_hai_baseline.txt`). Only the **slides** remain.

> ### Round 2 — raising the ceiling (`RESULTS.md` §10) — **closed**
>
> The diagnosis: 7 of 22 classes (mean F1 **0.378**, 17–83 train images each) hold ~**+0.10** of the
> remaining macro-F1, while the 7 common classes are saturated at 0.869 — so most of the headroom is
> a **data** problem. Reading `train_one` again turned up the other half: it runs **AdamW 1e-4 with
> no LR schedule**, augment = resize + hflip only, no mixup, no label smoothing, no EMA, 30 epochs.
> Part of what checkpoint ensembling earns is compensation for that missing schedule.
>
> A five-rung ladder was coded (2026-08-27) as flag-gated notebook sections, split across sessions by
> one variable `SESSION` in §6b: **M** paired bootstrap + McNemar (0 GPU) · **P2** modern recipe ·
> **P3** in22k/MIM backbone · **P4** domain pretraining on HyperKvasir · **P5** cosine head.
>
> | Session | Rungs | Outcome |
> |---|---|---|
> | `0` (A100, 28-08) | M | ✅ ran — §10.8 |
> | `1` (T4, 31-08) | P2, P2b | ✅ ran — §10.9 |
> | `4` (T4, 31-08) | P2c, P2b seeds 1–2, A1, A2 | ✅ ran — §10.10 |
> | `2` | P4 | ❌ **never run — out of GPU quota** |
> | `3` | P3, P5 | ❌ never run — `P3` was cut **on merit** (the standalone backbone lever measured +0.0034); `P5` was a maybe when the quota ran out |
>
> **The ladder is closed at P2**, and the two reasons are not the same reason — keep them apart when
> writing this up:
>
> * **`P3` and `P5` are model-family levers, and the measurement cannot resolve them.** Report §4.6
>   shows the whole common-class group has a ceiling of **+0.0420** against a ±0.035 resolution
>   threshold, so no "change the model" lever can be *proven* on this 1,586-image test set however
>   much GPU it gets. Stopping there is a result, not a shortfall.
> * **`P4` is the exception, and it is the one the quota stopped.** It adds *in-domain data*, which
>   is precisely the family §4.6's argument does **not** cover — and `A1` (+0.0254 from changing
>   pretraining data alone) is direct evidence the family is live. It is the report's headline piece
>   of future work.
>
> **What round 2 changed in this README's headline claim:** the proposed model moved `P1` → `P2` and
> the selection rule `top3_tta` → `top3` (see the box under the table in §1). Five results are worth
> carrying out of the log; the rest of the diary stays in `RESULTS.md` §10.8–§10.10:
>
> * **The lever is `recipe × ARCHITECTURE`, not `recipe × resolution`.** The 2×2 only closed once
>   `P2c` filled the empty cell: `recipe × architecture` at a fixed 224 px = **+0.0468** (above the
>   ±0.035 resolution threshold), `recipe × resolution` at a fixed backbone = **+0.0085**, 4× below
>   it. Never write the recipe up as a standalone gain — `P2b`, the same recipe on DenseNet-121,
>   is **−0.0110** over 3 seeds. `report/tables-offline/35_*`.
> * **288 px is a negative result, and the config to *deploy* is `P2c` @224.** Same seed 0, `P2c`
>   0.7172 vs `P2` 0.7166 = −0.0006, at **1.70×** less compute per image. The *reported* number
>   stays `P2` (3 seeds, known σ); `P2c` is the deployment recommendation.
> * **Changing the pretraining data beat changing the architecture.** `A1` (Swin-T, IN-1k → IN-22k,
>   nothing else touched) = **+0.0254**, larger than every architecture lever measured here. That
>   turns "the remaining headroom is data" from an argument into a measurement — and it is the
>   reason `P4` is named as the best future work.
> * **σ is the ruler every other lever is judged against, so a σ that does not replicate corrupts a
>   decision, not just a number.** The A100 round rejected logit adjustment only because `P1`'s σ
>   there was 0.0016 — four times smaller than the same config on T4 — and against a denominator
>   that small *any* variance-adding lever fails the ×1.5 test. On T4 the same test keeps it, and it
>   is worth +0.0143 on `P2` with σ *shrinking*. It is now part of the proposed system.
> * 🚨 **One tooling disease, three times: a table and the conclusion drawn from it not coming from
>   the same source.** (1) §16 printed "use top3" and then kept scoring under `best`; (2) cell 19b
>   hardcoded `"top3_tta"` while the locked rule was `top3`; (3) §15c–§15f read `SELECTION_RULE`
>   before §16 assigns it, so the 2×2 printed under `best` and its conclusion line carried the
>   **opposite sign** on the resolution term. All three are fixed by one discipline — every cell
>   calls the same `locked_rule()` — and `test_notebook.py` group 8 now guards it. Related: the
>   recursive image scan once counted a previous session's own Kaggle Output as 2 extra classes
>   (8,006 / 29 instead of 8,000 / 27); the `> 25 images` filter absorbed it, but the split would
>   have shifted silently had either folder held > 25 files.
>
> **Still missing, and not blocked by compute:** the **slides**, and a Gradio screenshot (§20b
> renders as a live widget, so the notebook stores no image; text evidence is in
> `report/tables-a100/29_*`). Everything else the brief asks for is measured and written up.
>
> One trap worth keeping visible even though the runs are over: Gate 0a measured that determinism
> holds **within** a GPU type, not **across** it, so any config whose seeds straddle two machines has
> a σ that mixes hardware — and σ is what the whole argument rests on. `run_seeds` records each
> seed's GPU in `checkpoints/gpu_log.json` and warns on mixing (`RESULTS.md` §10.7).

## 3. What the report should cover

**These are the brief's own rubric weights** (`../mse-dl-de-bai-vuot-baseline.pdf`, §6 *"Khung report
bám đúng tỉ lệ 70/30"*) — not our invention, and they add to 100. Note how they encode the 70/30 split:
the three data rows are 20 + 30 + 20 = **70%**, the three model rows are 5 + 15 + 10 = **30%**. Write the
report in these proportions; a brilliant architecture section is worth 15%, the augmentation ablation 30%.

| Part | Weight | What to do | Status |
|---|---|---|---|
| Problem & baseline | **5%** | Quote the paper table; state split / metric / protocol; report your reproduced baseline | ✅ `RESULTS.md` §0b + Claim 1 |
| Data analysis (EDA) | **20%** | Class distribution, image quality, artifacts, suspicious labels, **leakage / near-duplicate audit** (with figures) | ✅ `report/tables/06-08`, `report/figures/06_eda.png` |
| **Data processing** | **30%** | Normalize, domain-justified augmentation, class-balancing — **each step backed by an ablation, negative results included** | ✅ **complete after `SESSION = 4`** (`RESULTS.md` §10.10): a full **2×2** isolates the recipe lever — `recipe × architecture` **+0.0468** (above ±0.035) vs `recipe × resolution` **+0.0085** — and it ships **three** negative results the rubric asks for: `P2b` −0.0110 at **3 seeds**, 288 px −0.0006 under the modern recipe, and `A2` balanced softmax −0.0047 (which *replicates* group A's −0.007). Quote §10.10, not §2. ⚠️ `P2c` still has 1 seed, so every comparison using it is read on seed 0 |
| Labels & validation | **20%** | Long-tail handling, confusion matrix, per-class F1, typical error examples | ✅ `report/tables/18` — now **P2** (the cell auto-selects the best model), precision 0.810 vs recall **0.690**, i.e. the remaining bottleneck is the rare classes; per-class-vs-Table-3 analysis in `RESULTS.md` §9 |
| Architecture | **15%** | CNN vs Transformer under one protocol; argue why local vs global matters here (lecture 8: W-MSA / SW-MSA / Patch Merging) | ✅ **7** configs (6 × 3 seeds + `P2c` × 1), one protocol. ⚠️ The round-1 *ranking* does not survive a hardware change (`RESULTS.md` §10.9, `tables-offline/30_*`), and architecture **on its own** measures +0.0034 — so argue local-vs-global on the mechanism plus the **`recipe × architecture` interaction** (+0.0468, §10.10), never on the B0/S0/P0/P1 ordering |
| **Transfer learning: freeze vs trainable** | **10%** | The brief spells out the conditions: *linear probe / progressive unfreezing / layer-wise LR decay / full fine-tune*, **with a comparison table** | ✅ **measured, T4 numbers under `top3`** (`tables/27_*`). T1 linear probe **0.5674** · T2 lower half frozen **0.6596** · T3 progressive + discriminative LR **0.6394** · T4 full fine-tune (= `B0`) **0.6780 ± 0.0073** — all three frozen conditions lose by more than 2σ, and the paper's Table 2 shows the same ~0.16 gap on this split. Table: `report/tables/27_transfer_learning.txt`; analysis: `RESULTS.md` §19d. ⚠️ 1 seed per frozen condition, and T3 is 2-group discriminative LR, not per-layer decay — say so |
| Deployment | **—** | Export ONNX, measure latency + model size, Gradio/Streamlit demo. The brief calls this row *"Completeness of the Product"* and warns (p2) that *"a notebook that runs is not enough — a finished product with a demo is needed"* | 🔄 ONNX + latency + size — **re-measured on T4** (`report/tables/28`; the A100 table is at `tables-a100/28_*`, **do not mix them**). **§20b Gradio demo evidence is now only in `tables-a100/29_*`**: the T4 session skipped it because the resume path does not copy `.pt` files, and it will skip in every resumed Kaggle session until `P2_coatnet0_288_modern_seed0.pt` is added to the Kaggle Dataset (`RESULTS.md` §10.9). ⚠️ Two things the report must state: the demo runs **one** checkpoint + TTA, **not** the full `top3` + logit-adjustment system (0.7441), because `run_seeds` persists only the single best state; and no screenshot is stored in `report/` (Gradio renders as a live widget) — grab one by hand if needed |

> **Free corroboration for the transfer-learning row, already in hand:** the paper's own Table 2 is a
> freeze-depth experiment in disguise. Per its §4.2, DenseNet-121 and the second ResNet-50 fine-tune
> **all layers** (0.6504 / 0.6176) while ResNet-152, EfficientNet-B0 and DenseNet-169 fine-tune **only
> the last layer** (0.4496 / 0.4519 / 0.4883) — a ~0.16 macro-F1 gap on this exact split. So the
> published table already says freeze depth dominates architecture here, which is both a citation for
> our section and the reason our own 2-3 conditions are enough (`RESULTS.md` §0b).

### Method rules (keep results honest & comparable)
1. **Use the author's original split.** If you must re-split, publish the seed and explain.
2. Medical data → split at the level the authors defined; state that leakage was checked.
3. Imbalanced data → always report **macro-F1**, never accuracy or weighted-F1 alone.
4. **≥ 3 seeds, report mean ± std** (+ bootstrap CI). A single number beating baseline by 0.5 pt is meaningless.
5. **Reproduce the strongest baseline before claiming you beat it.** If you can't, say so plainly.
6. **Report the negative results.** "Method X did not help, and here is the evidence" is a finding.

---

## 4. Team split (2 people)

The **Data** work is done **together** (one shared recipe so all numbers are consistent); the
**Model** work is one baseline per person — which is exactly the CNN vs Transformer axis.

Both members build on the **shared baseline notebook** (`notebooks/final-gastrovision-classification.ipynb`)
so all measurements use the same split, seeds, and eval harness.

| Member | Track | Owns |
|---|---|---|
| **Member A** | **CNN — DenseNet-121** | 3-seed B0 → the reproduced baseline number; bootstrap CI; per-class F1 + confusion matrix; the **4-condition transfer-learning study (§19d, 10% of the report)**; ONNX + latency + the §20b Gradio demo. ⚠️ *B5 (DenseNet @288) was dropped* — the resolution lever is now tested as `P1` on the proposed backbone instead (`RESULTS.md` §3) |
| **Member B** | **Transformer — Swin-T** | `run_seeds(build_swin_t, ...)` under the **identical protocol, same 3 seeds** — **no extra tuning at all**, which is what keeps the comparison fair. ⚠️ *The planned 288 / softer-aug parity check (`S1`) was **dropped on purpose***: the resolution lever is isolated once, on the **proposed** backbone (`P0` → `P1`), rather than twice (`RESULTS.md` §4) |

### Shared by both — Data (the 70%)
- EDA: class distribution, image quality, artifacts, suspicious labels.
- **Leakage / near-duplicate audit across the 3 splits** — highest value per GPU-minute in the whole
  project: MD5 for byte-identical files, then cosine similarity on pretrained embeddings for
  near-duplicates. Endoscopy datasets easily contain multiple frames from the same case.
  Find something → it is the project's strongest Data-70% contribution. Find nothing → the report
  still gets to say *"leakage was checked"*, which is the first question a DL examiner asks.
- GI-domain augmentation pipeline (justified per step, with an **ablation table**).
- Class-imbalance handling — **already measured, all flat**: see the honest conclusion in `RESULTS.md`.
- Writing the report.

### The proposed model — a **system**, not just a different architecture

CoAtNet-0 and Swin-T sit ~1 ImageNet point apart — **smaller than this dataset's measurement noise**
(±0.02–0.05 macro-F1). Betting the entire improvement claim on one architecture swap is therefore a
weak bet, and the honest reading of our own ablation says why: the levers that showed signal are
*representation strength* and *variance reduction*, not the choice between two same-size backbones.

So the proposed model is:

> **CoAtNet-0 @ 288 + modern training recipe + top-3 checkpoint ensemble + logit adjustment**
> (`P2_coatnet0_288_modern` in the notebook) — **0.7441 ± 0.0088**, 3 seeds, rule `top3`.

> ⚠️ **This is the second version of this line.** Round 1 proposed `P1` = CoAtNet-0 @288 + hflip
> TTA + top-3, at 0.6961 ± 0.0016 on A100. It was superseded on 2026-08-31: `P2` — the *same*
> backbone under a modern recipe (cosine + LLRD + EMA + mixup, 80 epochs) — beat `P1` by
> **+0.0443**, more than the ±0.035 resolution of the measurement, and logit adjustment (dropped on
> `P1` because it inflated σ) **passes the same pre-registered test on `P2`**: +0.0143 with σ
> *shrinking* 0.0096 → 0.0088. `RESULTS.md` §10.9 findings 3 and 5.

⚠️ **Only CoAtNet-0 was ever run at 288.** In principle `timm` interpolates the relative-position
tables so all three backbones would accept 288×288, but **that was never measured**: no run in this
repo builds Swin-T or DenseNet-121 at 288, and `_timm_build` prints a warning and silently continues
at 224 if a model refuses `img_size`. Do not cite it as verified for all three.

**Two tables in the report, never mixed:**

| Table | Compares | Purpose |
|---|---|---|
| **a. Architecture only** | B0 vs S0 vs P0 — all @224, single model, no TTA | a strictly fair CNN / Transformer / Hybrid comparison |
| **b. System** | P2 (full proposal) vs both baselines | where the "beats 0.6504" claim lives — `report/tables-offline/33_*` |

Do **not** hand-roll "Swin-T + conv stem": a fresh conv stem is randomly initialised, which throws
away the pretrained stem and on ~8k images will almost certainly do worse than plain Swin-T. If the
proposal is a hybrid, it must be one with pretrained weights.

**Decide after `S0` lands, not before.** — **Answered by the A100 run, and it went to the second
branch.** Swin-T came out **statistically indistinguishable** from DenseNet-121 — +0.006 under the paper's
own selection rule, +0.0175 under the rule we chose, with bootstrap CIs overlapping heavily in both
cases — and CoAtNet-0 likewise, exactly as this rule predicted. (Note the honest phrasing: *not
separable at 3 seeds*, which is weaker than "flat".) So the backbone
hypothesis is **not supported**, and the improvement is carried by the levers instead:

> ⚠️ **Rewritten 2026-09-01 — this table has now been wrong twice, in two different ways.**
> Version 1 compared the checkpoint-rule lever measured under `top3_tta` against the architecture
> lever measured under `best`, and concluded the former was "~5× larger" — an artefact of mixing two
> selection rules. Version 2 fixed the rule but was built on the **A100** numbers, which
> `SESSION = 1` replaced wholesale when those configs were retrained on T4 (`RESULTS.md` §10.9
> finding 5). Numbers below are all **`top3`, T4 round, 3 seeds** — the rule actually locked in §16.

| Lever | Gain (under `top3`) | Above the ±0.035 resolution? | Cost |
|---|---|---|---|
| **modern training recipe** (`P1` → `P2`: cosine + LLRD + EMA + mixup, 80 ep) | **+0.0443** | ✅ **yes** | ~3.5× / run |
| logit adjustment at inference | +0.0143 | ❌ no | **0 epochs** |
| checkpoint rule `best` → `top3` | +0.0094 | ❌ no | **0 epochs** |
| resolution 224 → 288 (CoAtNet-0) | +0.0041 — and **−0.0006** under the modern recipe | ❌ no | 1.70× / image |
| architecture CNN → Hybrid (B0 → P0), *standalone* | +0.0034 | ❌ no | full retrain |

**Two of the levers this project was set up to test — architecture and resolution — buy +0.003 and
+0.004, indistinguishable from zero.** That is the report's central finding, and it is the opposite
of the hypothesis we started from ("the bottleneck is the backbone").

**But the recipe lever is not universal either, and that is the part most likely to be over-claimed.**
The same recipe on the paper's own backbone (DenseNet-121 @224, `P2b`) gives **−0.0110** over 3
seeds, **3/3 negative** — it makes DenseNet *worse*. `SESSION = 4` filled the empty cell of the 2×2
(`P2c` = CoAtNet-0 @224 + the same recipe) and split the two readings apart:

| Term, rule `top3` | Δ | Above ±0.035? |
|---|---|---|
| **recipe × architecture** (both at 224) | **+0.0468** | ✅ **yes** |
| recipe × resolution (both on the hybrid backbone) | +0.0085 | ❌ no |

On the **same seed 0**, each factor alone is ≤ 0 (architecture −0.0160; recipe −0.0043) while the
pair is +0.0294 — so this is an **interaction, not a sum of two independent levers**. It also kills
an implicit assumption: **288 px is not what made the difference.** Under the modern recipe, 288 vs
224 is **−0.0006** for **1.70×** the compute, so the config to *deploy* is `P2c` @224 — while the
*reported* number stays `P2`, which has 3 seeds. `RESULTS.md` §10.10.

Write the whole thing up as **negative results with evidence**, not as a disappointment: three of
them are exactly what the brief's 30% "data processing" criterion asks for (`P2b` at 3 seeds, 288 px
under the new recipe, and `A2` balanced softmax at −0.0047, which *replicates* an earlier −0.007).

**Ordering rule if time runs out:** finish both baselines first — an unfinished baseline hurts
the report more than a thin hybrid section. But "no hybrid at all" means the report has *no
proposed model*, so the fallback is a **1-seed hybrid reported honestly as preliminary**, never
a silent omission.

### Cut from the old 3-person plan
CoAtNet **as a baseline** (it moves to the proposed row, it is not dropped) · ViT-S (backup only) ·
B2b recipe retune · the full transfer-learning grid (now 3 conditions × 1 seed, DenseNet only).
**Kept:** deployment — it is whiteboard proposition 5 (*"Deployment: W → W′, freeze / trainable"*)
and feeds the "Completeness of the Product" grading criterion, and costs only a few hours.

> Notebook §8 ("Team plan") is the 2-person plan; if you are reading an older copy that still
> lists Member C / CoAtNet-as-baseline, **this README is the source of truth**.

---

## 5. Running it

The notebook is **generated**, not hand-edited: `build_notebook.py` emits
`notebooks/final-gastrovision-classification.ipynb`. Edit the builder, re-run `python build_notebook.py`,
and the notebook is rebuilt with no stale cells and no leftover outputs. (Same convention as the
`build_bt*.py` scripts in `../homeworks/`.)

### One notebook, three hardware profiles — auto-detected

`PROFILE` is picked from the hardware at import time. **The code path is identical in all three** —
same 22 classes, same four configs (`B0`/`S0`/`P0` @224 and `P1` @288), same train/eval functions.
Only the amount of data and the epoch count change.

| Profile | Selected when | Data | Epochs | Batch | Seeds | Purpose |
|---|---|---|---|---|---|---|
| `cpu-smoke` | no CUDA | 12/4/4 imgs per class | 2 | 8 | [0] | **debug the notebook**, measure throughput, run the data audit |
| `gpu-t4` | CUDA, VRAM < 24 GB | full | 30 | 32 | [0,1,2] | the real run (free Colab), fp16 AMP |
| `gpu-a100` | CUDA, VRAM ≥ 24 GB | full | 30 | **32** | [0,1,2] | the real run, **bf16 + TF32**, ~3× faster |

**A100 deliberately keeps batch 32.** Batch size is part of the protocol — changing it changes both the
effective learning rate and the number of update steps, so the result would no longer be comparable to
the paper's 0.6504 (batch 32) or to our earlier B0 run. The A100 advantage is taken as **bf16** (same
exponent range as fp32, so no `GradScaler` and none of fp16's overflow fragility) and **TF32** matmuls:
the same experiment, just faster. On A100 the whole plan fits in **one ~2 h session**.

Override with `PROFILE_OVERRIDE` — e.g. set it to `"cpu-smoke"` **on Colab** to walk the entire
notebook in ~5 minutes before committing a GPU session to it.

### On Colab
1. `Runtime → Change runtime type → GPU` (T4 is enough; A100 is auto-detected and gets **bf16 + TF32**
   — **not** a bigger batch: batch stays 32 on every profile, see above).
2. `Run all`. First run downloads `Gastrovision.zip` (~1.8 GB) via `gdown`, unzips it (the archive is a
   **zip inside a zip**, handled), and hard-fails if fewer than ~8,000 images land.
3. Checkpoints, per-seed **logits** and the val history go to Google Drive.

**Resume is built in.** If a seed's `.npz` already exists, that seed is **not retrained** — a dropped
Colab session costs nothing but the seed that was in flight. `FORCE_RERUN = True` overrides.

### On Kaggle (when Colab compute units run out)
Kaggle gives ~30 GPU-hours/week free (P100 or T4). The notebook detects Kaggle and skips
`drive.mount`, which raises `NotImplementedError` there.

> **Detection trap:** Kaggle **also ships the `google.colab` package**, so
> `try: from google.colab import drive` **succeeds** in both places — using the import as the Colab
> test is wrong, and wrong silently until `drive.mount()` throws. The correct signal is
> `/var/colab/hostname`, which is exactly what `drive.mount()` itself checks before mounting. Kaggle
> is detected first, via `KAGGLE_KERNEL_RUN_TYPE` or `/kaggle/working`.

The real difference from Colab is that Kaggle has **no single directory that is writable, readable and
survives between sessions**. It has two separate places: `/kaggle/working` is writable but only becomes
permanent when you hit **Save Version**, and `/kaggle/input/*` is **read-only**. So `CKPT_DIR` (write)
and `CKPT_READ_DIRS` (read) are now separate, and every checkpoint read goes through `ckpt_path()` /
`ckpt_ls()`; `gpu_log_read()` merges the `gpu_log.json` of every session it can see.

1. `Settings → Accelerator → GPU`, and turn **`Internet` on** if you want the dataset auto-downloaded.
2. Best is to attach GastroVision as an **Input dataset** — the notebook picks the first input folder
   holding > 1000 images and uses it in place, instead of re-downloading 1.8 GB every session.
3. Set `SESSION` in §6b, `Run all`.
4. **Hit `Save Version` when it finishes.** Otherwise `/kaggle/working` is discarded and the next
   session starts from scratch.
5. Next session: `Add Input → Your Work →` pick the previous session's output. The notebook scans
   `/kaggle/input/**` for `*_seed*.npz`, prints where it is resuming from, and lets the copy in
   `/kaggle/working` win over any older copy in the inputs.

> ⚠️ Kaggle hands out **P100 or T4**; round 1 ran on **A100**. Gate 0a measured that determinism does
> **not** hold across GPU types (~0.010 apart) — the same order as `P2`'s +0.02 decision threshold.
> The §6b inventory warns about it. Run first anyway; only if the result lands in the contested band
> (+0.015…+0.025) do you need to re-run `P1` on the **same** hardware as an anchor.

### Locally, with no GPU
Put `Gastrovision.zip` (or the extracted folders) under `final-project/data/` and run the notebook.
It detects the missing CUDA, switches to `cpu-smoke`, and completes in ~10 minutes on 16 CPU threads.
This is how the notebook is regression-tested before every Colab session — see `RESULTS.md` §8 for what
the last CPU run verified.

`python check_cells.py` is the cheap pre-flight: it syntax-checks **every** code cell in the generated
`.ipynb`. It replaces magic lines (`!pip`, `%%capture`) with `pass` rather than skipping the whole cell —
skipping by cell is what once let an unterminated f-string through both the checker and
`build_notebook.py` (the generator treats cell bodies as data, so it never parses them). The script
self-tests against that exact bug before it checks anything.

### What the notebook does that costs zero extra GPU
Everything after training reads back **saved logits**, so the following need no retraining at all:
bootstrap CIs · checkpoint-selection rules (best / 3-epoch-smoothed / top-3 ensemble) · TTA ·
class-prior logit adjustment (τ tuned on val) · cross-architecture ensembles.

**Auto-detection baked in:** device (GPU/CPU, AMP only on GPU) · Colab vs local paths · dataset
download + nested-zip extraction · `num_workers = 0` on Windows (notebook-defined `Dataset` cannot be
pickled by spawn) · integrity check with a hard fail · the paper's ">25 samples" class filter with an
`assert NUM_CLASSES == 22`.

---

## 6. Repo layout

```
final-project/
├── README.md                          # this file - the plan and its outcomes
├── RESULTS.md                         # measured numbers only (the ablation + session log)
├── build_notebook.py                  # SOURCE OF THE NOTEBOOK - edit here, never the .ipynb
├── rebuild_notebook.py                # RUN THIS instead: rebuild + keep the real run's outputs
├── test_notebook.py                   # 57 checks on the notebook's logic cells (no GPU, no data)
├── check_cells.py                     # syntax-checks every code cell of the .ipynb
├── requirements.txt
├── .gitignore                         # data/ checkpoints/ outputs/ ckpt-t4/ kout/ *.pt *.npz *.onnx
├── notebooks/
│   └── final-gastrovision-classification.ipynb # GENERATED - do not hand-edit. Keeps the real run's outputs
├── report/               # THE REPORT + its only source of numbers
│   ├── BAO_CAO.md        #   *** THE REPORT *** (Vietnamese, 10 sections = the brief's 70/30 framework)
│   ├── bao_cao.html      #   generated: self-contained page (figures inlined), published as an Artifact
│   ├── build_html.py     #   BAO_CAO.md -> bao_cao.html
│   ├── check_numbers.py  #   cross-checks all 105 numbers in the report against tables/ - exits non-zero on drift
│   ├── extract.py        #   re-runnable extractor: python report/extract.py (after every Kaggle/Colab session)
│   ├── offline_tables.py #   the 7 tables no notebook cell can print (30-36) - 0 GPU, from ckpt-t4/*.npz
│   ├── README.md         #   provenance, and which table feeds which report section
│   ├── figures/          #   3 PNGs extracted from the executed notebook (T4 round)
│   ├── figures-a100/     #   2 of those 3, from the A100 round before it was retrained
│                       #   (06_eda was byte-identical - EDA does not depend on hardware - so it was dropped)
│   ├── tables/           #   33 verbatim text outputs (T4 round, SESSION = 4)
│   ├── tables-a100/      #   30 from the A100 round; 7 are live sources (11 / 28 / 29, and 12-15
│                       #   feed table 30). The other 23 are archive only - do not quote them
│   └── tables-offline/   #   7 recomputed at 0 GPU (30-36)
├── data/          # (gitignored) dataset - created at runtime
├── ckpt-t4/       # (gitignored) per-seed logits (.npz) pulled back from Kaggle - offline_tables.py reads these
├── ckpt-a100/     # (gitignored) the same, from the A100 round - plus the T1/T2/T3 logits ckpt-t4/ lacks
├── checkpoints/   # (gitignored, recreated on demand) weights + logits written by a *local* run
└── outputs/       # (gitignored, recreated on demand) figures, CSV, ONNX written by a *local* run
```

> ⚠️ **Only `report/` holds the project's numbers.** `checkpoints/` and `outputs/` are whatever the
> *last local run* wrote; both were deleted on 2026-09-01 because they still held the CPU smoke run
> (macro-F1 0.29-0.41 at 1 seed, plus 352 MB of ONNX) and existed only to be warned about - but they
> come back the moment anyone runs the notebook locally, so the rule stands. `ckpt-t4/` and
> `ckpt-a100/` are real, but they are `.npz` logits feeding `report/offline_tables.py`, not something
> to read by hand. Take every number and figure for the report from `report/`. See `report/README.md`.

> Edit `build_notebook.py`, never the `.ipynb` — a hand edit is silently overwritten by the next build.
> ⚠️ **But `python build_notebook.py` also wipes the executed outputs**, and the committed notebook is
> the only record of the real run (its embedded figures are where `report/figures/` came from). So use
> the wrapper instead — it backs up, rebuilds, and merges the prose back into the executed notebook:
>
> ```bash
> python rebuild_notebook.py     # then: python report/extract.py
> ```
>
> It **refuses to merge** (and restores the backup) if a *code* cell changed or the cell count moved —
> because then the stored outputs no longer correspond to the code, and the honest move is to re-run on
> Colab. Prose-only edits merge cleanly and keep Colab's formatting, so the diff stays reviewable.
> The three runtime folders are gitignored, so `git add -A` from this folder is now safe (it was not
> before: only `data/` was ignored).
