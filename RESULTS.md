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
| Proposed model (**mandatory** — carries the improvement claim) | shared | **CoAtNet-0 @288 + modern training recipe + top-3 ckpt ensemble + logit adjustment** (`P2`); `P0` @224 and `P1` @288 are the intermediates that isolate the resolution lever, **not baselines** | §10.9, §10.10 |

> ### 🚨 Read this before quoting any macro-F1 from §2–§9
>
> This file is a **chronological log**, so the oldest numbers come first. **Every macro-F1 in §2,
> §3, §4, §5 and §9 is superseded.** §2 predates the determinism fix (it says so itself); §3–§5 and
> §9 are the **round-1 record (A100, 26–27 Aug 2026, rule `top3_tta`)**, and the four round-1
> configs were later **retrained** rather than resumed on the T4 round — so every score moved,
> `SELECTION_RULE` re-voted itself from `top3_tta` to **`top3`**, and the proposed model changed
> from `P1` to **`P2`**. Those sections are kept as the record of *how* each decision was reached,
> not as numbers to quote.
>
> **Not** superseded, because they do not depend on the training round: §1 (the paper's own
> numbers), §0b (verification at the source), §7 (the data audit — it depends only on the split,
> and the T4 round re-found the same 1 cross-split MD5 group and the same 9 near-duplicate pairs:
> `report/tables/07,08`).
>
> **The current numbers live in §10.9, §10.10, and `report/tables/21_bang_tong_ket.txt`:**
>
> | | round 1 — A100, `top3_tta` (§9) | **current — T4, `top3`** |
> |---|---|---|
> | `B0_densenet121` (reference baseline) | 0.6676 ± 0.0066 | **0.6780 ± 0.0073** |
> | `S0_swin_t` (new baseline) | 0.6851 ± 0.0114 | **0.6813 ± 0.0081** |
> | proposed model | `P1` 0.6961 ± 0.0016 | **`P2` 0.7298 ± 0.0096** — with logit adjustment, **0.7441 ± 0.0088** |
>
> Provenance of the switch: §10.9 finding 5. Cross-hardware spread: `report/tables-offline/30_*`.

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
  → ✅ **met.** `P2` + logit adjustment = **0.7441 ± 0.0088**, CI95 [0.6986, 0.7736]; the "what
  moves it" answer is §10.9 / §10.10 (and §9 for the round-1 version of it).
- **Stretch (the brief's actual target):** macro-F1 **0.72–0.75**.
  → ✅ **met by a single trained model**, which is stronger than what this line originally claimed.
  `P2` alone = **0.7298 ± 0.0096**; with logit adjustment (0 extra epochs) **0.7441 ± 0.0088**. The
  3-seed ensemble row, 0.7587, stays reported **separately** because it costs 3 training runs.

> ⚠️ **Both bullets above were rewritten 2026-08-31 / 09-01.** They used to read *"met: `P1` =
> 0.6961 ± 0.0016"* and *"stretch met only by the ensemble rows … no single trained model reaches
> 0.72"*. Both were true of the A100 round and are false now: `SESSION = 1` replaced those numbers
> (§10.9 finding 5) and `P2` cleared 0.72 as a single model.

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

## 3. Gate 0 — the gate that had to clear before any new rung *(closed 2026-08-27)*

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
| 3b | **Pick the rule** — best-val vs 3-epoch-smoothed vs top-3-ensemble, × TTA | 0 epochs | ✅ **decided by vote, and the vote is re-run every session.** Round 1 (A100) picked `top3_tta` (§9); the T4 round re-voted and picked **`top3`**, which is the rule in force — §10.9 finding 5. The notebook now *assigns* `SELECTION_RULE` from the ranking instead of only printing a suggestion (it used to print "use top3" and then keep using `best` for every table below it) |

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

> ⚠️ **Round-1 record (A100, `top3_tta`).** `S0` is now **0.6813 ± 0.0081** under `top3` on the T4
> round (`report/tables/21_bang_tong_ket.txt`); the 0.6851 ± 0.0114 below is the superseded value.
> The *protocol* argument in this section is unaffected — it is why the row is kept.

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
| **`P1` CoAtNet-0 @288 + TTA + top-3 ckpt ensemble** — ~~**the proposal**~~ | [0,1,2] | **0.6961 ± 0.0016** | **+0.0285** | **+0.0110** | ✅ done (§9) — **superseded**, see below |

> ⚠️ **`P1` is no longer the proposal.** Since `SESSION = 1` (2026-08-31) the proposed model is
> **`P2`** = the same backbone under a modern training recipe, + top-3 ensemble + logit adjustment
> = **0.7441 ± 0.0088** under rule `top3`. This whole section is the round-1 record: the numbers in
> it are A100 / `top3_tta` and were replaced when those configs were retrained on T4. Current
> numbers: §10.9, §10.10, and `report/tables/21_bang_tong_ket.txt`.

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

## 9. A100 run — round 1 (2026-08-26) — **superseded**

> 🚨 **Every number in this section has been replaced.** These four configs were **retrained** (not
> resumed) on the T4 round, so all scores moved and the rule re-voted itself to `top3`; the proposed
> model moved from `P1` to `P2`. Current numbers: §10.9, §10.10,
> `report/tables/21_bang_tong_ket.txt`. What this section is still good for: the *reasoning* — Claim
> 1 vs Claim 2 kept separate, the per-class comparison against the paper's Table 3, and the
> lever-by-lever accounting, all of which the T4 round re-confirmed in form if not in value. The
> A100 tables themselves are preserved in `report/tables-a100/`.

`final-gastrovision-classification.ipynb`, profile `gpu-a100`, 30 epochs, batch 32, bf16 + TF32,
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
→ `SELECTION_RULE = "top3_tta"` **for round 1**. ⚠️ The tie is this thin — 0.0004 — which is why
the T4 round's re-vote flipped it to **`top3`**, the rule the report actually uses. Do not read this
line as the report's rule.

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
| **`P1_coatnet0_288`** (~~proposed~~ — round 1 only) | **0.6961 ± 0.0016** | **+0.0457** |

⚠️ Superseded on both counts: the values are round-1/A100, and the proposal is now `P2`
(0.7298 ± 0.0096, or 0.7441 ± 0.0088 with logit adjustment). See the banner above.

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

---

## 10. Kế hoạch vòng 2 — nâng trần kết quả (lập 2026-08-27)

Mục này ghi lại **chẩn đoán** dẫn tới kế hoạch, **từng bước làm gì**, và quan trọng nhất là
**ý tưởng của mỗi bước lấy từ đâu** — để báo cáo truy nguyên được, và để phân biệt rõ cái gì là
*đọc được ở tài liệu ngoài* với cái gì là *đo được từ chính dự án này*.

### 10.1 Chẩn đoán: trần nằm ở đâu

Tách macro-F1 của `P1` (0.6941, seed 0) theo cỡ lớp:

| Nhóm | Số lớp | Mean F1 | Đóng góp |
|---|---|---|---|
| Common (≥ 66 ảnh test) | 7 | **0.869** | 6.085/22 |
| Rare (< 50 ảnh test) | 15 | **0.612** | 9.184/22 |

Nhóm common đã bão hoà. Toàn bộ khoảng cách còn lại nằm ở **7 lớp**: Mucosal inflammation LB
(0.000), Cecum (0.242), Colorectal cancer (0.372), Esophagitis (0.457), Gastric polyps (0.476),
Colon diverticula (0.545), Resected polyps (0.552) — mean **0.378**. Nếu riêng nhóm này lên 0.70 thì
macro-F1 tăng **+0.103** → ~0.80. Đó là toàn bộ headroom, và nó là headroom **dữ liệu**.

Ba bằng chứng nói đây **không** chỉ là lỗi tối ưu hoá:
1. Bốn kiến trúc + 288px + ensemble vẫn không nhấc nổi Cecum (paper 0.23 → ta 0.242); Colorectal
   cancer còn tệ hơn paper.
2. Các lớp yếu đều có precision cao / recall thấp (Cecum 0.400/0.174) — hình dạng của bias long-tail
   — **nhưng cả ba đòn bẩy nhắm đúng bias đó đều trượt** (§6, §9): Balanced-Softmax −0.007, cRT
   −0.013, logit adjustment +0.0094 với σ phình 9×. Không phải lệch ngưỡng, mà là **đặc trưng không
   tách được**.
3. 17 ảnh train cho 2 lớp; mất cân bằng 50.6×; và §7 đã đo được các cặp ảnh cosine 1.0000 mang **hai
   nhãn khác nhau** → trần cứng cho mọi mô hình.

**Nhưng** đọc lại `train_one` thì phía mô hình cũng chưa cạn: nó chạy **AdamW 1e-4 hằng số, không
scheduler**, augment chỉ `Resize` + lật ngang, không mixup, không label smoothing, không EMA, 30
epoch. Đây là phát hiện làm đổi thứ tự ưu tiên — sửa công thức rẻ và chắc hơn đổi kiến trúc.

> Ghi chú tự phê: `top3_tta` ăn tới +0.018…+0.032 **một phần vì** không có LR schedule → val dao
> động mạnh → trung bình 3 checkpoint mới ổn định. Nói cách khác, một phần "đòn bẩy đo lường" mà §9
> tự hào là đang **bù cho một công thức thiếu**. Câu này phải nằm trong báo cáo.

### 10.2 Bậc thang, chi phí, và **nguồn của từng ý tưởng**

| Bậc | Nội dung | Chi phí A100 | Ước lượng | **Nguồn ý tưởng** |
|---|---|---|---|---|
| **M** | **Paired bootstrap + McNemar** thay cho việc so hai CI độc lập | 0 (đọc từ `.npz`) | không đổi điểm, **siết CI** | Dietterich 1998, *Approximate statistical tests for comparing supervised classification learning algorithms*; McNemar 1947; bootstrap cặp — Efron & Tibshirani. **Động cơ là của dự án**: §9 kết luận "CIs overlap → không claim được", đó là phép thử yếu nhất có thể dùng vì nó cộng cả độ khó bộ test vào hai bên |
| **P2** | Công thức huấn luyện hiện đại, **giữ nguyên** CoAtNet-0 @288: cosine+warmup, LLRD 0.75, mixup/CutMix, label smoothing 0.1, EMA, 80 epoch | ~1.5 h (3 seed) + ~26 ph (`P2b`) | **+0.02…+0.04** | Wightman et al., *ResNet strikes back* ([2110.00476](https://arxiv.org/abs/2110.00476)) — đổi công thức ăn hơn đổi kiến trúc. Chi tiết từng thành phần: notebook §15c. **Số 80 epoch là đo tại chỗ**: `report/tables/24_duong_hoc_val.txt`, P1 đỉnh val ở epoch 27/30 |
| **P3** | Backbone pretrain mạnh hơn (in22k / MIM): EVA-02-S, ConvNeXt-S in12k, Swin-B in22k, MaxViT-T. Bake-off 1 seed rồi mới 3 seed cho người thắng | ~1.5 h | +0.01…+0.03 | Lựa chọn EVA-02 cho ảnh nội soi lấy từ [arXiv 2410.21302](https://arxiv.org/abs/2410.21302) (EndoExtend24). **Phải làm sau P2**: backbone to hơn trên 4,758 ảnh sẽ overfit nếu chưa có mixup + LLRD |
| **P4a** | **Domain pretrain**: huấn luyện backbone trên HyperKvasir (10,662 ảnh, 23 lớp GI) rồi mới fine-tune GastroVision | ~1.5 h + tải ~4 GB | **+0.03…+0.08** (ít chắc chắn nhất) | HyperKvasir: [Nature Sci Data 2020](https://www.nature.com/articles/s41597-020-00622-y), [datasets.simula.no/hyper-kvasir](https://datasets.simula.no/hyper-kvasir/). Nguyên tắc "SSL/pretrain in-domain thắng pretrain ảnh tự nhiên": [GastroNet-5M, Gastroenterology 2025](https://www.sciencedirect.com/science/article/pii/S001650852505797X) và [Med Image Anal 2024](https://www.sciencedirect.com/science/article/pii/S1361841524002238). **Bằng chứng nội bộ mạnh hơn cả hai**: §19d đo được linear probe mất 9.5 điểm vì "đặc trưng ImageNet là đặc trưng ảnh tự nhiên, các tầng *đầu* mới phải dịch chuyển" |
| **P4b** | SSL (DINO/MAE) trên 99,417 ảnh không nhãn của HyperKvasir | nhiều giờ | — | cùng nguồn P4a; chỉ làm nếu dư ngân sách |
| **P5** | **Cosine classifier + prototype init** cho các lớp 6–17 ảnh | ~0 | +0.005…+0.02 | Chen et al., *A Closer Look at Few-shot Classification* ([1904.04232](https://arxiv.org/abs/1904.04232)); Qi et al., *Low-shot learning with imprinted weights* ([1712.07136](https://arxiv.org/abs/1712.07136)). Khác B3/B4 về **cơ chế**: sửa hình học không gian đặc trưng, không sửa trọng số loss — nên không dính lý do khiến B3/B4 phẳng |

**Tổng ~5.3 h A100.** Cột "ước lượng" là **phỏng đoán**, không phải số đo — neo vào hai đòn bẩy đã
đo được (224→288 cho +0.0143; `best`→`top3_tta` cho +0.03).

### 10.3 Hai kỷ luật bắt buộc

1. **Ngưỡng phân giải.** CI bootstrap là **±0.035** và σ giữa seed chỉ 0.0016 (§9). Bất kỳ bậc nào
   dưới ~0.02 sẽ **không chứng minh được** trên bộ test 1,586 ảnh này. Vì vậy P2 được gộp thành một
   **bundle** rồi mới ablate 2–3 thành phần lớn nhất — không đo lẻ từng thứ.
2. **Kiểm rò rỉ trước khi công bố P4.** Cả HyperKvasir lẫn GastroVision đều từ bệnh viện Na Uy, và
   HyperKvasir có đúng những tên lớp `dyed-lifted-polyps` / `dyed-resection-margins` như
   GastroVision → **khả năng trùng ảnh là có thật**. Phải chạy lại pipeline MD5 + near-dup
   MobileNetV3 của §7 giữa HyperKvasir và **test split** trước khi trích dẫn bất kỳ con số P4 nào.
   Trùng mà không kiểm thì kết quả vô giá trị; kiểm mà sạch thì đó là một đóng góp Data-70% nữa.

### 10.4 Đã quyết định KHÔNG làm

| Việc | Vì sao |
|---|---|
| Thêm seed (`SEEDS=[0,1,2,3,4]`) | đã bác ở §9 — σ đã 0.0016, cái chi phối là ±0.035 của bộ test |
| Thêm biến thể loss imbalance (LDAM-DRW, focal…) | B3/B4/logit-adjust đều phẳng; nút thắt không nằm ở loss |
| Đóng băng backbone ở mọi mức | §19d: cả ba điều kiện freeze đều thua > 2σ |
| Logit adjustment ở dạng hiện tại | không lặp lại qua seed (§9); τ đang fit trên val logits của `best` rồi áp lên điểm `top3_tta` — lệch phân phối. Sửa được nhưng phải lưu cả 3 state và train lại, **không free** |
| GastroNet-5M làm phụ thuộc của kế hoạch | kiểm tra 2026-08-27: repo HF `tgwboers/GastroNet-5M_Pretrained_Weights` **chỉ có `README.md`, không có file weights** (đã chuyển sang "Theta Vision Cortex") → coi như phải xin quyền, không đưa vào đường găng |

### 10.5 Trạng thái thực thi

Tất cả **đã code xong 2026-08-27** và nằm trong notebook dưới dạng cờ bật/tắt, để một lần
`Run all` chạy hết. **Bậc M đã chạy 2026-08-28** (`SESSION = 0`, 0 GPU) — kết quả và hai lỗi nó
làm lộ ra nằm ở §10.8. **Bậc P2 + P2b đã chạy 2026-08-31** (`SESSION = 1`, T4) — §10.9.
**Phiên 4 đã chạy 2026-08-31** (`SESSION = 4`, T4): `P2c`, `P2b` đủ 3 seed, `A1`, `A2` — §10.10.
🔒 **Vòng 2 đã đóng ở P2 (01-09-2026): `P3` / `P4` / `P5` sẽ không chạy** — hết quota GPU. `P3`
đã bị cắt vì lý do chuyên môn từ trước (đòn bẩy backbone đứng một mình chỉ +0,0034); `P4` và `P5`
dừng vì hết giờ máy, không phải vì hết lý do. Cả hai đi vào báo cáo dưới dạng **hướng phát triển
tiếp**, và `P4` là bậc mạnh nhất trong số đó (§10.2, và `A1` ở §10.10 là bằng chứng đo được cho
nó). Bảng dưới giữ nguyên chi phí ước tính như bản ghi của kế hoạch.

| Bước | Mục | Cờ | Chi phí A100 | Trạng thái |
|---|---|---|---|---|
| **P2** công thức hiện đại | §15c | `RUN_P2`, `RUN_P2_RECIPE_CHECK` | ~2 h | ✅ **xong 31-08 (T4)** — §10.9 |
| **P2c/P2b/A1/A2** lấp lô-gic | §15c, §21 | `RUN_P2C`, `RUN_P2B_FULL`, `RUN_ABLATIONS` | ~1,4 h | ✅ **xong 31-08 (T4)** — §10.10 |
| **P3** bake-off backbone | §15d | `RUN_P3` | ~2 h | 🔒 **bỏ vì lý do chuyên môn** (trước khi hết quota) — đòn bẩy backbone đứng một mình đo được +0,0034 |
| **P4** pretrain HyperKvasir | §15e | `RUN_P4` | ~2,5 h | 🔒 **không chạy — hết quota**; bậc mạnh nhất còn lại, chuyển thành hướng phát triển tiếp |
| **P5** đầu cosine | §15f | `RUN_P5` | ~1,5 h | 🔒 **không chạy — hết quota**; kỳ vọng có tăng lại (§10.9 phát hiện 3) nhưng không kịp đo |
| **M** paired bootstrap + McNemar | §16b | luôn chạy | 0 | ✅ xong 28-08 (§10.8) + chạy lại 31-08 |
| | | | **~8 h** | |

**8 giờ không vừa một phiên Colab** — nhưng resume xử lý được: mỗi seed xong là ghi `.npz` xuống
Drive, phiên sau nạp lại và chỉ chạy phần còn thiếu. Cách chia phiên nằm ở §10.6.

### 10.6 Công tắc phiên (§6b của notebook)

Sáu cờ nằm rải rác ở sáu ô khác nhau — sửa tay giữa các phiên là chuyện sớm muộn sẽ quên một cái.
Notebook có **§6b** với đúng **một biến `SESSION`** để bật cả bộ:

| `SESSION` | Chạy gì | A100 | T4 | Trạng thái |
|---|---|---|---|---|
| `0` | không huấn luyện gì — đọc lại các run cũ + §16b | ~10 ph | ~10 ph | ✅ 28-08 (§10.8) |
| `1` | `P2` + `P2b` | ~2 h | **~7 h** *(đo thật)* | ✅ 31-08 (§10.9) |
| `2` | `P4` | ~2,5 h | ~8,8 h ⚠️ sát trần | 🔒 **không chạy — hết quota** |
| `3` | `P3` + `P5` | ~3,5 h | ~12,3 h ⚠️ không vừa 1 phiên | 🔒 **không chạy** — `P3` bỏ vì lý do chuyên môn, `P5` vì hết quota |
| `4` | `P2c` + `P2b` đủ 3 seed + `A1`/`A2` | ~1,4 h | **5,6 h** *(đo thật)* | ✅ 31-08 (§10.10) |
| `"all"` | tất cả, kể cả Gate 0a | ~9,4 h | ~33 h | — |
| `"manual"` | không đụng cờ nào — mỗi ô giữ mặc định của nó | — | — | — |

⚠️ **Cột T4 của bảng này từng sai gần một nửa** (5,5 / 7 / 10 h) vì suy ra từ A100 bằng hệ số 2,75×
đoán bừa. Hệ số thật đo được ở phiên 1 là **~3,5×**, và các con số trên đã tính lại theo nó. Riêng
phiên 4 thì suy ra từ **số đo trực tiếp** của phiên 1, không qua hệ số — xem §6b của notebook.

Thứ tự đề xuất là **0 → 1 → 4**, không phải chạy tuần tự hết:

* **`SESSION = 0` trước tiên** vì bậc **M** không tốn một epoch nào mà lại là thứ duy nhất có thể
  biến các dòng *"CI chồng nhau → chưa kết luận được"* ở §9 thành kết luận thật. Đây là điểm có
  tỉ lệ giá trị/chi phí cao nhất trong toàn bộ vòng 2.
* **`P2` là điểm quyết định**, không phải một bậc trong chuỗi: nếu `P2 − P1 < +0,02` thì `P3` (vốn
  chạy *dưới* công thức của `P2`) mất phần lớn lý do tồn tại → bỏ luôn, đi thẳng `SESSION = 2`.
  Nếu `P2` tăng nhưng `P2b − B0 ≈ 0` thì công thức hợp riêng với hybrid và `P3` mới đáng ~3,5 h.
* Ngưỡng `+0,02` lấy từ CI bootstrap ±0,035 ở §9 — dưới mức đó thì bộ test 1.586 ảnh này không
  chứng minh được gì, dù chạy bao nhiêu seed.

> ✅ **Đã chạy xong 2026-08-31 (§10.10).** Cả bốn hạng mục dưới đây đều có số; ba hạn chế mà nó nhắm
> vào đã được xoá khỏi `report/BAO_CAO.md` §9. Phần bên dưới giữ nguyên làm **hồ sơ lý do chọn phiên
> 4**, vì kết luận nó thu được (`công thức × kiến trúc`, không phải `× độ phân giải`) **ngược** với
> phỏng đoán ngầm lúc lên kế hoạch — và đó chính là lý do phải chạy chứ không suy đoán.
>
> ~~**Phiên tiếp theo, nếu còn quota:** `P2c` seed 1+2 (~3,3 h T4) trước `SESSION = 2`.~~
> 🔒 **Không còn quota — không phiên nào chạy nữa (chốt 01-09-2026).** Chỗ yếu mà nó nhắm vào vẫn
> còn: số hạng tương tác +0,0497 tựa lên một ô **1 seed**, và đó là một **hạn chế của báo cáo**,
> không phải một việc đang chờ.

**Cập nhật 2026-08-31 sau phiên 1: phiên tiếp theo nên là `SESSION = 4`, không phải 2 hay 3.** Lý do
là `P2` đã vượt ngưỡng, nên câu hỏi đổi từ *"còn đòn bẩy nào nâng được con số"* sang *"tuyên bố hiện
tại đã đủ chống đỡ chưa"* — và câu trả lời là chưa, ở ba chỗ đã được ghi tên thành hạn chế 2, 3, 5
của `report/BAO_CAO.md` §9:

| Hạng mục của phiên 4 | Giờ T4 | Lỗ nó lấp |
|---|---|---|
| **`P2c`** CoAtNet-0 @224 + công thức mới, 1 seed | ~1,1 | **Ô còn trống của bảng 2×2.** `P2`/`P1` so ở 288 còn `P2b`/`B0` so ở 224 → đòn bẩy +0,0443 vẫn lẫn giữa *công thức × kiến trúc* và *công thức × độ phân giải*. `P2c` đặt hybrid ở **đúng 224 của `P2b`**, nên `P2c − P0` so trực tiếp được với `P2b − B0`. Đây là lỗ **logic** trong tuyên bố mạnh nhất của báo cáo |
| **`P2b`** seed 1 + 2 | ~3,1 | Kết quả **âm** quan trọng nhất (−0,0024) hiện chỉ 1 seed → chỉ đọc được ở mức *"không có dấu hiệu dương"*. Rubric đòi *"negative results included"*; một kết quả âm có σ đáng giá hơn nhiều |
| **`A2`** Balanced-Softmax, 1 seed | ~0,4 | Ablation cân bằng lớp — thuộc hạng mục **30%**, nặng điểm nhất. Số đang dùng là nhóm A: 1 seed, quy tắc cũ, **trọng số không còn tồn tại**, và §2 tự ghi *"đừng trích các số này"* |
| **`A1`** Swin-T in22k, 1 seed | ~0,4 | Ablation *"nguồn pretrain mạnh hơn mua thêm bao nhiêu"* — hứa ở §5.1 của báo cáo, **chưa từng chạy** |

**Thứ tự, và phiên 4 KHÔNG thay thế cái gì.** Các số `SESSION` là một **thực đơn, không phải bậc
thang** — chúng được đánh số theo thứ tự *viết ra*, không phải thứ tự *nên chạy*. Phiên 4 đi trước vì
nó rẻ nhất (5,0 h) và vì nó sửa những tuyên bố hiện **chưa được chống đỡ**, thay vì cộng thêm một con
số lên trên chúng. Nhưng hai bậc còn lại không tương đương nhau:

* **`SESSION = 2` (P4, domain pretrain trên HyperKvasir) là bậc mạnh nhất còn lại và nên đi ngay sau
  phiên 4** nếu còn quota. Báo cáo kết luận **bốn đường độc lập** rằng nút thắt là biểu diễn đặc
  trưng ở phần đuôi — và P4 là lever **duy nhất** còn lại *thêm dữ liệu trong domain* thay vì đổi mô
  hình. Lý lẽ của nó là chính phép đo §19d của nhóm (linear probe mất 11,1 điểm vì *"đặc trưng
  ImageNet là đặc trưng ảnh tự nhiên, và chính các tầng đầu mới phải dịch chuyển"*), đúng thứ mà
  pretrain trên 10.662 ảnh nội soi nhằm vào. Kỳ vọng **+0,03…+0,08 — lớn nhất trong mọi bậc**, và
  cũng ít chắc chắn nhất (§10.2).
  ⚠️ Hai rủi ro thực tế: ước tính ~8,8 h là **suy ra** ×3,5 từ A100 chứ không phải đo, trong khi ngân
  sách là 9,0 h và trần Kaggle là 12 h — gần như không còn dư; và nó cần tải ~4 GB. Phải đọc output
  kiểm rò rỉ của `RUN_P4` trước khi tin bất kỳ con số nào của nó (quyết định thiết kế #3, §10.9).
* **`SESSION = 3`: bỏ `P3`, để `P5` là "có thể".** `P3` chạy *dưới* công thức của `P2` nên chỉ đổi
  backbone — đúng đòn bẩy vừa đo được là +0,0034, tức trong nhiễu. `P5` đã lấy lại lý do tồn tại
  (§10.9 phát hiện 3) nhưng nó *nâng con số*, mà con số đã vượt baseline với CI không chồng lấn, còn
  phép đo thì không phân giải nổi mức tăng tiếp theo.

🔒 **Chốt 01-09-2026: không phiên nào trong hai bậc trên được chạy — hết quota GPU.** Phân tích bên
trên giữ nguyên vì nó vẫn đúng về *thứ tự đáng chạy*, và vì nó là cơ sở để báo cáo xếp `P4` lên đầu
mục hướng phát triển tiếp. Trạng thái chính thức: §10.5.

`P1` luôn bật ở mọi phiên (chỉ đọc lại `.npz`, vài giây) vì §15c, §16 và §19 đều trỏ tới nó.
`_plan()` mặc định **tắt hết** rồi mới bật lại từng cờ, nên sau này thêm một bậc `P6` vào
`_ALL_FLAGS` thì các phiên cũ tự động không chạy nó, thay vì lặng lẽ dài thêm vài giờ.

### 10.7 Nhật ký GPU — chống trộn phần cứng giữa các seed

Gate 0a (§3) đo được tính tất định giữ **trong một loại GPU** chứ không giữ **giữa hai loại**:
cùng seed, A100 cho `[0.428608, 0.549646, 0.551532]` còn T4 cho `[0.430615, 0.540379, 0.541930]`.
Vì vòng 2 chia thành nhiều phiên và Colab cấp GPU nào là tuỳ lúc, có một kịch bản hỏng rất dễ xảy
ra: seed 0–1 của `P2` chạy A100, phiên sau được T4 và chạy nốt seed 2 → **`σ` của `P2` trộn hai
loại phần cứng**, mà `σ` chính là thứ toàn bộ lập luận ở §9 dựa vào. Nhìn vào `.npz` không thấy
được lỗi này.

Nên `run_seeds` giờ ghi loại GPU của **từng seed** vào `checkpoints/gpu_log.json` (trên Drive) ngay
khi seed đó train xong — ghi ngay chứ không đợi cả tag chạy hết, vì phiên Colab hay đứt giữa chừng.
Sau mỗi tag, `gpu_log_check()` in cảnh báo nếu các seed không cùng một loại máy; §6b cũng kiểm kê
đầu phiên và báo nếu máy hiện tại khác loại với máy đã dùng cho các cấu hình đã có. Đọc lại kết quả
cũ trên máy khác loại thì không sao — chỉ **huấn luyện thêm seed** mới là vấn đề.

Các run trước 2026-08-28 không có bản ghi này (in ra `khong ro`); theo §3 chúng đều chạy trên A100.

**Cập nhật 2026-08-29 — vòng 2 nhiều khả năng chạy trên máy khác loại.** Colab đã hết compute unit,
nên vòng 2 chuyển sang **Kaggle** (P100 hoặc T4, ~30 h/tuần miễn phí). Điều này biến cảnh báo ở trên
từ giả thuyết thành chuyện chắc chắn xảy ra: `P2` sẽ chạy trên P100/T4 còn `P1` đang là A100, nên
phép so `P2 − P1` **vắt qua hai loại phần cứng**, lệch ~0,010 theo Gate 0a — cùng bậc độ lớn với
ngưỡng quyết định +0,02.

Cách xử lý đã chọn: **chạy trước, vá sau nếu cần.** Nếu `P2 − P1` ra ngoài vùng ±0,015…+0,025 thì
0,010 không đổi được kết luận nào. Chỉ khi nó rơi đúng vào vùng tranh chấp mới cần chạy lại `P1` trên
cùng loại máy làm mỏ neo (~1,8 h T4, và phải đặt **tag mới** vì `.npz` cũ sẽ bị resume đọc lại).
Chi trước 1,8 giờ cho một mỏ neo có thể không cần đến là sai thứ tự.

Kaggle cũng buộc phải tách `CKPT_DIR` (nơi ghi) khỏi `CKPT_READ_DIRS` (nơi đọc): `/kaggle/working`
ghi được nhưng chỉ vĩnh viễn sau `Save Version`, còn `/kaggle/input/*` chỉ đọc. Mọi chỗ đọc
checkpoint đi qua `ckpt_path()`; `gpu_log_read()` gộp nhật ký của **mọi** phiên nhìn thấy được, nếu
không thì mỗi phiên Kaggle đều báo `khong ro` và cảnh báo trộn phần cứng thành vô dụng.

#### Hai cái bẫy đã dính khi chuyển sang Kaggle (2026-08-29)

**1. `import google.colab` thành công trên Kaggle.** Phép nhận diện `try: from google.colab import
drive / except ModuleNotFoundError` là **sai**: Kaggle cài sẵn gói đó, nên import chạy ngon và
notebook đi thẳng vào nhánh Colab rồi chết ở `drive.mount()`. Dấu hiệu đúng là **`/var/colab/hostname`**
— chính là thứ `drive.mount()` kiểm tra bên trong trước khi chịu mount, đọc được từ traceback của
nó. Kaggle được nhận ra **trước**, bằng `KAGGLE_KERNEL_RUN_TYPE` hoặc `/kaggle/working`. Ô đó giờ in
ra môi trường nó tự nhận, để lần sau sai thì thấy ngay dòng đầu tiên thay vì thấy traceback.

**2. Ô kiểm tra cú pháp đã *giấu* một lỗi thật.** Luật cũ là "ô nào có dòng bắt đầu bằng `!` hoặc `%`
thì coi là ô magic, bỏ qua". Một chuỗi bị xuống dòng giữa chừng làm dòng kế tiếp bắt đầu bằng `!!`
→ cả ô được miễn, `ast.parse` không bao giờ chạy, và một `unterminated f-string literal` đi lọt qua
cả bước kiểm tra lẫn `python build_notebook.py` (generator chỉ coi nội dung ô là **dữ liệu** trong
`r"""..."""`, nên nó không parse gì cả). Chỉ đến khi chạy thật ô đó mới nổ.

Luật đã sửa trong `check_cells.py`: **thay từng dòng magic bằng `pass` rồi parse phần còn lại** — ô
magic thật vẫn sạch, còn chuỗi hỏng thì vẫn hỏng, không ô nào được miễn nữa. Script tự kiểm tra
chính nó trên đúng đoạn code đã gây lỗi trước khi kiểm notebook.

Bài học chung cho cả hai: **một luật "bỏ qua" đặt sai chỗ thì tệ hơn không có luật nào**, vì nó biến
một lỗi ồn ào thành một lỗi im lặng. Cả hai lần đều là như vậy — nhận diện môi trường bỏ qua nhánh
đúng, và trình kiểm tra bỏ qua ô hỏng.

### 10.8 Phiên 0 đã chạy (A100, 2026-08-28) — bậc **M** không cứu được kết luận

`SESSION = 0`, không huấn luyện một epoch nào. Phần cơ học đạt hết: đúng 7 tag trên Drive với đủ
số seed, cả 5 bậc bị bỏ qua in đúng thông báo, và **mọi con số đọc lại khớp §9 tới từng chữ số**
(`P1 = 0,6961 ± 0,0016`, `B0 = 0,6676`, `S0 = 0,6851`, `P0 = 0,6818`).

Nhưng kết quả của bậc **M** thì không như kỳ vọng — và đó là phần đáng viết vào báo cáo.

| A vs `B0_densenet121` | d macro-F1 | CI95 của d | McNemar b/c | p |
|---|---|---|---|---|
| `S0_swin_t` | +0,0207 | [−0,0150, +0,0557] | 80/62 | 0,1534 |
| `P0_coatnet0` | +0,0037 | [−0,0305, +0,0382] | 87/70 | 0,2015 |
| `P1_coatnet0_288` | +0,0173 | [−0,0249, +0,0590] | 94/64 | **0,0208** |

**Kết luận trung thực:** `P1` hơn `B0` một cách có ý nghĩa **ở mức từng tấm ảnh** (McNemar p = 0,021
trên 158 ảnh hai mô hình bất đồng), nhưng **không** chứng minh được ở **macro-F1** — chỉ số mà bài
báo và toàn bộ báo cáo này dùng. Hai phép kiểm trả lời hai câu hỏi khác nhau và cả hai đều đúng;
câu phải viết là câu thứ hai, vì macro-F1 mới là thứ được báo cáo.

#### Phát hiện 1 — ghép cặp **không** làm CI hẹp lại, và lý do lại là chẩn đoán của chính §17

Kỳ vọng ban đầu (đã ghi sai trong docstring của `paired_bootstrap`) là CI của hiệu sẽ hẹp hơn nhiều
so với CI riêng lẻ. Đo thực tế thì **ngược lại**:

| | độ rộng CI95 |
|---|---|
| CI riêng của `P1` | 0,0697 |
| CI của Δ (`P1 − B0`) | **0,0839** |

Ghép cặp chỉ triệt tiêu được sai số ở những ảnh mà hai mô hình *cùng* đúng hoặc *cùng* sai — tức là
ở các lớp nhiều ảnh, nơi macro-F1 gần như **không có** phương sai để triệt tiêu. Phương sai của
macro-F1 nằm ở các lớp 6 ảnh (`Mucosal inflammation` F1 = 0,000 / 6 ảnh; `Colon diverticula` 6 ảnh),
mà ở đó hai mô hình gần như **độc lập** vì cả hai đều đoán gần như ngẫu nhiên → phương sai của hiệu
bằng **tổng** hai phương sai chứ không phải hiệu.

Nói cách khác: **phép ghép cặp thất bại vì đúng cái nút thắt mà §17 đã chỉ ra.** Đây là một xác nhận
**độc lập** cho chẩn đoán "thiếu dữ liệu ở lớp hiếm", đến từ một hướng hoàn toàn khác (thống kê so
sánh chứ không phải bảng per-class) — nên nó vào báo cáo như một kết quả, không phải một lỗi.

#### Phát hiện 2 — `PAIR_SEED = 0` làm yếu **cả ba** so sánh cùng lúc

`B0` là mẫu số chung của mọi cặp, và seed 0 tình cờ là seed **tốt nhất** của nó (0,6768 so với trung
bình 0,6676 trên 3 seed). May mắn của riêng nó bị trừ vào cả ba dòng:

| Cặp | d ở seed 0 | seed 1 | seed 2 | TB 3 seed |
|---|---|---|---|---|
| `P1 − B0` | **+0,0173** | +0,0343 | +0,0338 | **+0,0285** |
| `P0 − B0` | **+0,0037** | +0,0200 | +0,0189 | +0,0142 |
| `S0 − B0` | +0,0208 | +0,0241 | +0,0077 | +0,0175 |

`PAIR_SEED = 0` là mặc định lúc viết code, không phải một lựa chọn có lý do. Nhưng vì điều này chỉ
lộ ra **sau** khi đã nhìn kết quả, đổi sang một seed khác là chọn số liệu. Cách xử lý đã áp dụng:
**in hết cả ba seed**, thêm dòng biên độ, và thêm một dòng `ens3seed` (trung bình softmax 3 seed —
bỏ hẳn xổ số seed, nhưng so **hệ thống** chứ không so mô hình đơn nên vẫn là dòng riêng, đúng quy
ước ensemble ở §16). Người đọc tự thấy dao động; không ai chọn hộ họ.

#### Phát hiện 3 — hiệu chỉnh logit trên `P1` ≈ 0, và điều đó hạ kỳ vọng của **P5**

§16 đo lại: `τ*` chọn trên val ra **0,9 / 0,5 / 0,0** tuỳ seed, lãi test tương ứng **+0,0011 /
+0,0271 / +0,0000**. Trung bình cao hơn +0,0094 nhưng σ phình từ 0,0016 lên 0,0139 — mức tăng là
của **một** seed, không phải của phương pháp (kết luận này đã có từ §9, nay lặp lại y nguyên).

**P5** (đầu phân loại cosine) nhắm vào **cùng một thiên lệch** — ưu thế của lớp nhiều ảnh trong lớp
phân loại cuối. Không phải bằng chứng quyết định, vì P5 sửa lúc *huấn luyện* còn hiệu chỉnh logit chỉ
vá lúc *suy luận*; nhưng nó là dữ kiện trực tiếp nhất hiện có về nút thắt đó, và nó nói: **không có
gì để ăn ở đây trên `P1`.** Cộng với việc `P3` vốn đã xếp cuối từ vòng 1, **`SESSION = 3` yếu đi từ
cả hai phía** và nên là thứ cuối cùng được cân nhắc, nếu còn cân nhắc.

#### Hai con số mạnh nhất hiện có (đều đã có sẵn, 0 GPU)

* **Ensemble 3 seed của `P1`: macro-F1 = 0,7221, CI95 [0,6728, 0,7609]** — CI **không** chồng lấn
  0,6504 → được quyền viết "vượt baseline công bố". Dòng riêng: tốn 3 lần huấn luyện.
* **Ensemble 4 kiến trúc, chọn tổ hợp trên val: test = 0,7242** — và tổ hợp tốt nhất trên val cũng
  chính là tốt nhất trên test (chênh +0,0000), nên không có rò rỉ quy trình ở đây.

#### Điều này đổi gì trong kế hoạch

Không đổi thứ tự. `SESSION = 1` vẫn là bước tiếp theo, vì **P2 là bậc duy nhất còn lại có thể nâng
số của một mô hình đơn** — mà mô hình đơn mới là thứ §9 so với bài báo. Nhưng ngưỡng quyết định giờ
có thêm một cách đọc: nếu `P2` không vượt `+0,02`, thì cùng với phát hiện 3 ở trên, **cả `SESSION =
3` lẫn phần lớn vòng 2 đều nên dừng**, và báo cáo chốt ở kết quả âm có bằng chứng — vốn đã là một
kết quả tốt, và giờ có ba nguồn độc lập cùng chỉ về một nguyên nhân.

### 10.9 Phiên 1 đã chạy (T4 / Kaggle, 2026-08-31) — P2 vượt ngưỡng, và một phát hiện ngoài kế hoạch

`SESSION = 1`, notebook chạy hết 49 ô code, **0 lỗi**, tốn **94 phút** trong ngân sách 9 h: `P2` đã
có đủ 3 seed từ phiên 2026-08-30 nên được đọc lại từ `.npz`, chỉ `P2b` thực sự huấn luyện. Bảng:
`report/tables/15b_*`, `15c_*`, `17b_*`, `21_*`.

#### Phát hiện 1 — `P2` vượt ngưỡng quyết định, điều khoản "dừng vòng 2" KHÔNG kích hoạt

§10.8 đặt trước: nếu `P2 − P1 < +0,02` thì bỏ `P3` và phần lớn vòng 2. Đo được (quy tắc **`top3`** —
quy tắc đã chốt; `report/tables/17b_so_sanh_theo_cap.txt`):

| A vs B | Δ macro-F1 (TB 3 seed) | seed 0 / 1 / 2 | dòng `ens3seed` |
|---|---|---|---|
| `P2 − P1` | **+0,0443** | chưa KL / **có ý nghĩa** / **có ý nghĩa** | **+0,0414 [+0,0053, +0,0760]** |
| `P2 − B0` | **+0,0518** | chưa KL / **có ý nghĩa** (McNemar p = 0,0002) / **có ý nghĩa** (p = 0,0012) | **+0,0430 [+0,0147, +0,0691]** |

Và `P2` là **cấu hình đầu tiên** có CI riêng **không chồng lấn** 0,6504: `0,7166 [0,6660, 0,7531]`.
Bốn cấu hình vòng 1 vẫn "chưa kết luận được" (`tables/17_bootstrap_ci.txt`).

⚠️ **Nhãn quy tắc: +0,0443 là dưới `top3`, không phải `top3_tta`.** Bảng 15c đọc cùng đòn bẩy
dưới `top3_tta` và ra **+0,0471**. Hai con số đến từ hai phép tính khác nhau — 17b lấy trung bình
các Δ ghép cặp *từng seed*, 15c lấy hiệu của hai *trung bình 3 seed* — nên chúng không phải cùng
một đại lượng. Điều đáng nói là đòn bẩy này **không phụ thuộc quy tắc chấm điểm**: +0,0443 và
+0,0471 đều vượt ngưỡng 0,035. (Bản đầu của mục này ghi sai nhãn là `top3_tta`;
`report/check_numbers.py` bắt được.)

`PROPOSED_TAG` đã đổi sang `P2_coatnet0_288_modern` (ô 19b) vì cả hai cổng của nó đều mở: mức tăng
lớn hơn ngưỡng phân giải ±0,035, **và** `P2b` đã trả lời được câu "công thức hay kiến trúc".

#### Phát hiện 2 — đòn bẩy **không** phải "công thức hiện đại" nói chung: `P2b` là kết quả âm sạch

Cùng công thức đó áp lên backbone của baseline (`tables/15c_P2_tach_don_bay.txt`):

| Cặp | Backbone / độ phân giải | Δ vs công thức cũ |
|---|---|---|
| `P2 − P1` | CoAtNet-0 @288 | **+0,0471** |
| `P2b − B0` | DenseNet-121 @224 | **−0,0024** (1 seed) |

Trên val còn rõ hơn: `P2b` đỉnh 0,6539 so với `B0` 0,6600 — công thức mới làm DenseNet **tệ đi**.
Nên trong báo cáo tuyệt đối không được viết "công thức hiện đại đáng +0,047" như một phát biểu
chung; phải viết là **công thức × (kiến trúc hoặc độ phân giải)**, và dòng `P2b` phải đi kèm.

⚠️ **Bảng 2×2 còn thiếu một ô.** `P2`/`P1` so ở 288, `P2b`/`B0` so ở 224 → chưa tách được
`công thức × kiến trúc` khỏi `công thức × độ phân giải`. Ô còn trống là **CoAtNet-0 @224 + công
thức mới** (~52 phút T4, 1 seed). Và `P2b` hiện chỉ 1 seed nên chưa có σ để xếp hạng.

#### Phát hiện 3 — cái bẫy nằm ở **σ**, không ở lever: `P5` hồi sinh

Output lưu trong `.ipynb` vẫn in `P1` (nó sinh ra *trước* khi `PROPOSED_TAG` được đổi), nên con số
của `P2` được tính lại từ logits đã lưu bằng `report/offline_tables.py` →
`report/tables-offline/31_he_thong_p2_hieu_chinh_logit.txt`:

| Lần chạy / mô hình | quy tắc | τ* mỗi seed | TB Δ | độ lệch |
|---|---|---|---|---|
| **A100**, `P1` (§10.8) | `top3_tta` | 0,9 / 0,5 / 0,0 | +0,0094 | 0,0016 → 0,0139 **×8,7 → BỊ LOẠI** |
| T4, `P1` | `top3` | 0,2 / **0,0** / 0,1 | +0,0052 | 0,0068 → 0,0095 (×1,40 → **giữ, sát mép**) |
| **T4, `P2`** | `top3` | **0,5 / 0,5 / 0,3** | **+0,0143** | 0,0096 → **0,0088 co lại → giữ** |

**Phát hiện 3 của §10.8 không sai về số liệu, nhưng sai về đối tượng.** Mẫu số của phép kiểm ×1,5 là
σ, và ở vòng A100 σ của `P1` là **0,0016** — nhỏ hơn 4 lần cùng cấu hình đo trên T4. Với mẫu số như
thế thì **bất kỳ** lever thêm chút phương sai nào cũng thất bại phép kiểm. Tiêu chí không phát hiện
lever vô dụng; nó phát hiện rằng **σ của lần chạy đó nhỏ một cách không tái lập được** — và phát
hiện 5 dưới đây cho biết chính xác vì sao.

Trên T4, cùng tiêu chí **giữ** hiệu chỉnh logit cho cả `P1` lẫn `P2`. Nhưng chúng đạt theo hai cách
khác nhau về chất: `P1` có Δ (+0,0052) **nhỏ hơn σ của chính nó**, một seed chọn τ* = 0, và σ phồng
×1,40 (sát ngưỡng ×1,50); `P2` có Δ gấp 1,5 lần σ, τ* ổn định xa 0, và σ **co lại**. Lý do hợp lý:
per-class của `P2` là precision **0,810** so với recall **0,690** — mô hình mạnh hơn thì phần thiên
lệch còn lại chủ yếu là thiên lệch **prior**, đúng thứ mà hiệu chỉnh logit sửa.

**`P5` (đầu cosine) lấy lại kỳ vọng** — nó nhắm đúng chỗ đó nhưng sửa lúc huấn luyện. Và bài học
chung đắt hơn cả `P5`: **σ là cái thước dùng để chấp nhận hay loại mọi lever khác, nên một σ không
tái lập được thì làm sai không phải một con số mà một QUYẾT ĐỊNH.**

Hệ thống đề xuất mới, dưới quy tắc `top3` đã chốt:

| | macro-F1 (3 seed) | vs paper 0,6504 | CI 95% (seed đầu) |
|---|---|---|---|
| `P2` + `top3` | 0,7298 ± 0,0096 | +0,0794 | — |
| **`P2` + `top3` + hiệu chỉnh logit** | **0,7441 ± 0,0088** | **+0,0937** | [0,6986, 0,7736] **vượt** |
| + ensemble 3 seed (dòng riêng) | 0,7587 | +0,1083 | [0,7110, 0,7924] **vượt** |

#### Phát hiện 4 — ensemble nhiều kiến trúc giờ là kết quả **âm**

Tổ hợp tốt nhất **chọn trên val** (`S0 + P1 + P2`) cho test **0,7130**, *thấp hơn* `P2` chạy một
mình (0,7166) — `tables/20_donbay_ensemble_kientruc.txt`. Một khi đã có một mô hình vượt hẳn, ghép
thêm mô hình yếu chỉ kéo xuống. Đòn bẩy 3 phải báo cáo là **âm**, không phải "+0,03" như vòng 1.
(Tổ hợp cao nhất trên *test* là `B0 + P2` = 0,7283, chênh +0,0153 — dòng đó vẫn không được báo cáo,
vì chọn tổ hợp bằng chính tập test là rò rỉ quy trình.)

#### Phát hiện 5 — 4 cấu hình gốc đã bị huấn luyện lại trên T4; toàn bộ §9 và `report/tables/12–24` cũ đã bị thay thế

Đây là thứ không nằm trong kế hoạch nào và là hệ quả quan trọng nhất của phiên này.

Bằng chứng, ba tầng độc lập:

1. `sec_per_epoch` trong `ckpt-t4/*.npz` là **32–56 s**, còn A100 đo được ~20 s cho `B0` — đúng
   bậc chênh T4/A100, tức là các seed này **có** chạy huấn luyện.
2. Bốn bảng phía **dữ liệu** giống nhau **từng byte** giữa hai vòng (`04_loc_lop_22`,
   `05_chia_split`, `06_eda`, `09_bo_danh_gia`) → split, luật lọc lớp và bộ đánh giá **không**
   đổi. Vậy nguyên nhân lệch số không thể là chia lại dữ liệu.
3. Nhưng **mọi** con số của **mọi** cấu hình dưới **mọi** quy tắc đều lệch §9.

Ba điều đó chỉ đồng thời đúng trong một trường hợp: **trọng số đã được huấn luyện lại**, chứ
không phải "đọc lại `.npz` rồi tính lại từ logits đã lưu".

Hai hệ quả trực tiếp:

1. **`SELECTION_RULE` đã tự lật `top3_tta` → `top3`.** Quyết định thiết kế #1 ("khoá phiếu bầu ở
   `B0`/`S0`/`P0`/`P1`") không cứu được, vì chính **bốn cử tri** đã bị train lại. Mọi con số ở §9
   giờ nằm dưới một quy tắc khác **và** một bộ trọng số khác.
2. **Câu khẳng định trong `report/README.md` là sai**, và chính nó là lý do việc train lại đi lọt:
   *"Điểm macro-F1 không phụ thuộc phần cứng — cả 12 lượt chạy chính đều được khôi phục từ `.npz`."*
   Câu đó đúng **nếu** resume thật sự xảy ra; nó không kiểm tra rằng resume **đã** xảy ra. Đã sửa.

Nhưng nó đồng thời tặng không một phép **lặp lại A100 ↔ T4** hoàn chỉnh: cùng code, cùng split,
cùng seed, 4 cấu hình × 3 seed × 6 quy tắc, 0 GPU để phân tích
(`report/tables-offline/30_lap_lai_a100_vs_t4.txt`):

| quy tắc | chênh trung bình (tuyệt đối) trên 4 cấu hình | lệch lớn nhất ở một seed |
|---|---|---|
| **`top3`** | **0,0046** | 0,0263 |
| `top3_tta` | 0,0092 | 0,0310 |
| `smooth` | 0,0153 | 0,0710 |
| `best` | 0,0182 | 0,0435 |
| `smooth_tta` | 0,0185 | 0,0717 |
| `best_tta` | 0,0185 | 0,0428 |

Bốn điều đọc ra được, tất cả đều 0 GPU:

* **`top3` bền với phần cứng gấp ~4× các quy tắc một-checkpoint.** Đây là **lý lẽ thứ ba, độc lập**
  cho luận điểm chính của báo cáo — bên cạnh "miễn phí" (0 epoch) và "giảm phương sai" ở §9. Cả ba
  đều là lập luận về **chi phí và tái lập**, không phải về **độ lớn hiệu ứng**, đúng như §9 đã chốt.
* **Xếp hạng kiến trúc vòng 1 không sống nổi qua một lần đổi máy.** Dưới `top3_tta`: A100 cho
  `P1 > S0 > P0 > B0`, T4 lật thành `B0 > S0 > P1 > P0`. Dưới `top3` thì xếp hạng **giữ được**
  (`P1` nhất, `B0` bét ở cả hai máy) — thêm một điểm nữa cho `top3`. Nói cách khác: §9 đã **đúng**
  khi từ chối tuyên bố "Swin-T hơn DenseNet-121"; giờ có bằng chứng trực tiếp thay vì chỉ có CI
  chồng lấn. Biên độ 0,015–0,019 của các quy tắc một-checkpoint **lớn hơn** cả đòn bẩy kiến trúc
  mà vòng 1 đo được (+0,0142…+0,0175).
* **Gate 0a đã đánh giá thấp hiệu ứng này.** Gate 0a đo lệch ~0,010 trên đường val 3 epoch; ở 30
  epoch cộng thêm bước chọn checkpoint thì lệch tích luỹ tới **0,04–0,07** ở một seed.
* **Kết luận về `P2` thì an toàn.** +0,0443 lớn hơn mọi hiệu ứng phần cứng trong bảng, và
  `P2`/`P2b`/`B0` đều trên T4 nên các phép so của phiên này **không** vắt qua hai loại máy — rủi ro
  mà §10.7 lo (`P2` trên T4 so với `P1` trên A100) đã tự biến mất, chỉ theo một cách không ai chọn.

#### Bốn bảng bị xuống cấp so với bản A100 — bản cũ đã lưu ở `report/tables-a100/`

Phiên này không chạy Gate 0a, ablation, demo; và ô 19b in ra trước khi `PROPOSED_TAG` được đổi:

| Bảng | Bản T4 (`tables/`) | Bản A100 (`tables-a100/`) |
|---|---|---|
| `11_gate0a_tat_dinh` | "bỏ qua Gate 0a" | hai lần chạy trùng 6 chữ số ✅ |
| `23_he_thong_de_xuat_3seed` | vẫn là `P1` (0,6865) | `P1` 0,6961 — cả hai đều lỗi thời; số đúng ở `tables-offline/31_*` |
| `25_ablation_tuy_chon` | "bỏ qua ablation" | cũng bỏ qua |
| `29_demo_gradio` | **"không thấy `P1_coatnet0_288_seed0.pt`"** | nạp checkpoint thật + tự kiểm tra ✅ |

Bảng `29` là dòng **Deployment** của rubric và nó sẽ **hỏng ở mọi phiên Kaggle resume**: nhánh
resume cố tình không copy `.pt` (~100 MB/file) từ `/kaggle/input`, mà Kaggle Dataset `ckpt-t4` cũng
chỉ chứa `.npz`. Muốn §20b sống lại thì phải đưa `P2_coatnet0_288_modern_seed0.pt` vào Dataset, hoặc
attach Output của phiên 2026-08-30 làm input.

#### Hai lỗi công cụ đã sửa trong phiên này

1. **`extract.py` lệch 4 prefix.** Notebook đã đổi `RUN_DETERMINISM_CHECK = True` / `RUN_P1_288 = True`
   sang `SESSION_FLAGS.get(...)`, và ô 16 có thêm dòng bình luận ở đầu. Script `sys.exit` khi không
   khớp — đúng thiết kế — nhưng nó exit **sau khi** đã xoá sạch `tables/`. Đã sửa cả 4 prefix và
   thêm 3 bảng của vòng 2 chưa từng được trích: `15b` (P2), `15c` (tách đòn bẩy), `17b` (so sánh
   theo cặp — chính là kết quả của bậc **M**).
2. **`flatten_cr` ăn mất mọi output dùng CRLF.** Kaggle chạy `!nvidia-smi` qua subprocess kiểu
   Windows nên mỗi dòng kết thúc bằng `\r\n`; luật "giữ đoạn sau CR cuối" biến `"abc\r"` thành
   chuỗi **rỗng**, nên `tables/00_gpu.txt` ra file trắng và script chỉ cảnh báo *"cell chưa chạy?"*
   thay vì báo lỗi. Đã chuẩn hoá CRLF → LF **trước** khi áp luật CR. Cùng một bài học với hai cái
   bẫy ở §10.7: **một luật "bỏ qua" đặt sai chỗ thì tệ hơn không có luật nào.**

### Bốn quyết định thiết kế đã cứng hoá trong code

1. **Quy tắc checkpoint bị khoá phiếu bầu.** `RULE_VOTERS` trong §16 chỉ gồm `B0`/`S0`/`P0`/`P1`.
   Trước đó ô này xếp hạng trên *mọi* mô hình trong `RESULTS_STORE`, nên chỉ cần thêm P2…P5 là
   `SELECTION_RULE` có thể lật từ `top3_tta` sang quy tắc khác — và **mọi con số đã báo cáo ở §9 sẽ
   lệch đi mà không ai thấy**. Các bậc mới được *chấm* dưới quy tắc đã chốt, không được *chọn* nó.
2. **`P3` chọn người thắng trên VAL.** Xếp hạng 4 ứng viên trên test rồi báo cáo điểm test của người
   thắng là chọn mô hình trên tập test — ở mức nhiễu ±0,035 thì con số ấy gần như chắc chắn không
   lặp lại. Vòng loại 40 epoch, chung kết 80 epoch, và ô in ra lời nhắc rằng hai số đó **không** so
   trực tiếp với nhau được.
3. **`P4` kiểm rò rỉ trước, huấn luyện sau.** Ảnh HyperKvasir trùng test GastroVision bị loại khỏi
   tập pretrain **trước** khi train, số lượng in ra và bắt buộc vào báo cáo.
4. **`macro_f1` / `evaluate` / `train_modern` giờ nhận số lớp.** Giai đoạn pretrain HyperKvasir có
   23 lớp; để mặc định `NUM_CLASSES = 22` thì mixup `scatter_` ném `IndexError` và macro-F1 trung
   bình trên nhầm tập lớp. Mặc định vẫn là 22 lớp GastroVision cho mọi con số báo cáo.

⚠️ **Notebook đã được dựng lại (60 → 79 ô) nên output của vòng chạy A100 trong file `.ipynb` đã bị
xoá.** Không mất dữ liệu: `report/tables/` và `report/figures/` đã commit, và bản notebook kèm
output nằm ở commit `4001cd1` (`git show 4001cd1:final-project/notebooks/gastrovision_classification.ipynb`).
Lần chạy Colab tới sẽ nạp lại toàn bộ 12 run cũ từ `.npz` trên Drive, nên chỉ `P2` + `P2b` thực sự
tốn GPU.

---

### 10.10 Phiên 4 đã chạy (T4 / Kaggle, 2026-08-31) — bảng 2×2 khép lại, và câu trả lời ngược dự đoán

`SESSION = 4`, Tesla T4, **5,57 giờ** trong ngân sách 9,0 h (5,36 h huấn luyện thật: `P2b` seed 1+2,
`P2c` seed 0, `A1`, `A2`; mọi thứ khác đọc lại từ `.npz`). Notebook kèm output đã thay bản cũ tại
`notebooks/final-gastrovision-classification.ipynb`. Báo cáo đã được cập nhật toàn bộ và
`report/check_numbers.py` chạy **105/105 khớp**.

**Phiên này không nâng macro-F1, và nó không nhằm nâng** — con số báo cáo vẫn là 0,7441 ± 0,0088.
Nó đổi ba câu *"chưa kết luận được"* thành ba câu có bằng chứng.

#### Phát hiện 1 — đòn bẩy là `công thức × KIẾN TRÚC`, không phải `× độ phân giải`

Bảng 2×2 dưới quy tắc `top3` (`report/tables-offline/35_bang_2x2_tuong_tac.txt`):

| | công thức cũ | công thức mới | đòn bẩy | seed |
|---|---|---|---|---|
| CoAtNet-0 @288 | `P1` 0,6855 | `P2` 0,7298 | **+0,0443** | 3 vs 3 |
| CoAtNet-0 @224 | `P0` 0,6814 | `P2c` 0,7172 | **+0,0358** | 3 vs 1 |
| DenseNet-121 @224 | `B0` 0,6780 | `P2b` 0,6670 | **−0,0110** | 3 vs 3 |

* `công thức × kiến trúc` (cùng 224) = **+0,0468** → **vượt** ngưỡng ±0,035
* `công thức × độ phân giải` (cùng hybrid) = **+0,0085** → dưới ngưỡng 4 lần

Trên cùng seed 0 còn sạch hơn: kiến trúc một mình **−0,0160**, công thức một mình **−0,0043**, cả
hai **+0,0294** → số hạng tương tác **+0,0497**. Đọc lại toàn bảng dưới `top3_tta` cho +0,0529 /
+0,0083 — **kết luận không phụ thuộc quy tắc chấm điểm**, điều đáng kiểm vì §10.9 vừa cho thấy quy
tắc chấm điểm từng đảo cả một xếp hạng.

Hạn chế 2 của báo cáo (*"chưa tách được công thức × kiến trúc khỏi công thức × độ phân giải"*) đã
**xoá**. Thay vào đó là một hạn chế hẹp hơn: ô mang số hạng tương tác mới có **1 seed**.

#### Phát hiện 2 — 288 px là kết quả âm; cấu hình nên triển khai là `P2c` @224

`P2c` @224 = 0,7172 so với `P2` @288 = 0,7166 trên **cùng seed 0** → **−0,0006**, trong khi 288 tốn
**1,70×** chi phí tính toán mỗi ảnh (8,74 vs 5,13 ms @ batch 32, `tables/28_*`). `P2c` cũng đạt đỉnh
val ở epoch **11/80** thay vì 40/80 — hội tụ nhanh hơn gần 4 lần.

`P2c` là cấu hình **thứ hai** có CI không chồng lấn 0,6504: [0,6664; 0,7568], McNemar p = 0,0018 so
với `B0`. Nhưng con số **được báo cáo** vẫn giữ là `P2`, vì `P2` có đủ 3 seed và σ đã biết. Hai vai
trò tách bạch: `P2` = kết quả đã đo đủ, `P2c` = cấu hình nên triển khai.

Hệ quả cho đường cộng dồn ở `BAO_CAO.md` §5.3: bậc "độ phân giải +0,0041" **chỉ đúng dưới công thức
cũ**. Bảng cộng dồn giờ được gán nhãn rõ là *mô tả con đường đã đi*, không phải phân rã nhân quả —
phân rã nhân quả là bảng 2×2.

#### Phát hiện 3 — `P2b` đủ 3 seed: kết quả âm giờ xếp hạng được

−0,0043 / −0,0086 / −0,0200, **3/3 seed đều âm**, mean −0,0110, σ = 0,0119, và thấp hơn cả trên val
(0,6539 vs 0,6600). |Δ| vẫn dưới ±0,035 nên **không** được nói "công thức làm hại DenseNet-121";
được nói **"chắc chắn không giúp"**. Hạn chế 3 của bản trước đã xoá.

#### Phát hiện 4 — mất cân bằng: chữa lúc suy luận thắng chữa trong hàm mất mát

Cùng DenseNet-121, cùng split, cùng seed 0, cùng 30 epoch, cùng `top3`
(`report/tables-offline/36_mat_can_bang_va_pretrain.txt`):

| | macro-F1 | Δ |
|---|---|---|
| `B0` cross-entropy | 0,6878 | — |
| `A2` balanced softmax — sửa **hàm mất mát** | 0,6831 | **−0,0047** |
| `B0` + hiệu chỉnh logit — sửa **lúc suy luận** | 0,7091 | **+0,0213** |

−0,0047 **lặp lại** −0,007 của nhóm A (giao thức cũ, phần cứng khác) — một trong số ít lever của dự
án lặp lại được, tiếc là lặp lại một kết quả âm. Cặp số này là phép so sạch nhất mà báo cáo có cho
luận điểm ở §3.3, và nó thay một đối chiếu vốn phải bắc cầu qua hai cấu hình khác nhau.

#### Phát hiện 5 — `A1`: đổi **dữ liệu pretrain** mua được nhiều hơn đổi kiến trúc

Swin-T, chỉ đổi bộ trọng số khởi tạo IN-1k → IN-22k: 0,6774 → **0,7028**, **+0,0254** — lớn hơn
*mọi* lever kiến trúc dự án này đo được (+0,0033 … +0,0034 trên 3 seed; −0,0160 trên seed 0). Đây là
số đo trực tiếp đầu tiên ủng hộ hướng "thêm dữ liệu" ở `BAO_CAO.md` §4.6, thay cho một lập luận
thuần lý thuyết. Nó **không** chứng minh `P4` sẽ dương — IN-22k không phải dữ liệu nội soi — nên
+0,03…+0,08 của `P4` vẫn phải đọc là phỏng đoán.

#### Hai lỗi công cụ mà phiên này làm lộ ra

1. **§15c in bảng dưới sai quy tắc.** Ô đọc `SELECTION_RULE`, nhưng biến đó chỉ được **chốt ở §16**,
   chạy *sau* nó — nên bảng 2×2 in ra dưới `best` thay vì `top3`, và dòng kết luận đi kèm đọc
   **ngược dấu** ở thành phần độ phân giải (`+0,0526` thay vì `+0,0085`). Bốn ô §15c–§15f đều mắc
   lỗi này; ba ô kia im lặng chỉ vì `P3`/`P4`/`P5` chưa chạy.
   Đây là **lần thứ hai** cùng một loại lỗi: §10.9 là ghim cứng tên quy tắc, lần này là đọc một biến
   chưa được gán. Cùng một bệnh — **bảng số và kết luận rút ra từ nó không sinh ra từ cùng một
   nguồn**. Đã chữa bằng cách đưa `RULE_VOTERS` + `vote_rule()` lên §13 và cho **cả §15c–§15f lẫn
   §16 gọi chung một hàm**; §16 vẫn là nơi chốt `SELECTION_RULE`. Hàm mới được đối chiếu với bản
   pandas cũ trên đúng dữ liệu phiên 4: cùng người thắng, cùng thứ tự, cùng hạng trung bình.
   Output cũ trong `.ipynb` thì không sửa được nếu không chạy lại GPU, nên bảng đúng nằm ở
   `tables-offline/35_*`, và ô 15c tự in ra cảnh báo này.

2. **Phép quét ảnh đệ quy nhặt luôn thư mục Output của phiên trước.** Kaggle gắn Output của notebook
   lần trước làm input, nên `scan` thấy `outputs/` và `__results___files/` — mỗi thư mục 3 **hình vẽ
   matplotlib** — và tính chúng thành hai "lớp": **8.006 ảnh / 29 lớp** thay vì 8.000 / 27. Luật
   `> 25 ảnh` của bài báo loại cả hai nên **split không đổi** (`tables/05,06,09` byte-identical với
   vòng trước), nhưng nếu một thư mục đó có > 25 file thì split đã âm thầm đổi và không có gì báo.
   Ghi lại đây như một rủi ro có thật của cơ chế resume qua thư mục input; biện pháp rẻ nhất là
   `assert` số lớp == 22 **và** tổng ảnh == 7.930 ngay sau bước lọc.

#### Trạng thái sau phiên 4

| | |
|---|---|
| Đã xong | `B0` `S0` `P0` `P1` `P2` `P2b` × 3 seed · `P2c` `A1` `A2` × 1 seed · T1–T4 · bậc M |
| Không chạy (**chốt 01-09-2026, hết quota**) | `P3` (bỏ vì lý do chuyên môn), `P4`, `P5` — đều thành hướng phát triển tiếp |
| Rẻ nhất đã bỏ lỡ | `P2c` seed 1+2 (~3,3 h T4) — sẽ biến số hạng tương tác từ 1 seed thành 3 seed; **đây là hạn chế còn lại của báo cáo**, không phải một việc đang chờ |
| Còn **thiếu** thật sự | ~~slide~~ ✅ *02-09-2026* — `report/slides.html` (dựng từ `report/build_slides.py`, nội dung theo đúng thứ tự lập luận của `BAO_CAO.md`) · ~~bằng chứng demo Gradio~~ ✅ *02-09-2026* — demo chạy thật đầu-cuối trên CPU kèm screenshot, xem §11.2; điều còn thiếu đúng nghĩa là demo với chính checkpoint `P2` của vòng T4 (`.pt` không đi theo dataset checkpoint) |

---

## 11. CPU full-data run — notebook CPU-only trên toàn bộ 7.930 ảnh (2026-08-29, khôi phục 2026-09-02)

> Mục này từng là §10 của commit `52c191a` và bị rơi mất khi RESULTS.md được viết lại toàn bộ ở
> `e773138` (dán đè từ bản trên máy khác — không cố ý). Khôi phục lại đây, cập nhật phần so sánh
> theo bộ số T4 hiện hành, và bổ sung §11.2 (bằng chứng demo, 02-09-2026).

`gastrovision_classification_cpu.ipynb` (sinh bởi `build_cpu_notebook.py`) — bản **CPU-only**:
ẩn CUDA trước khi `import torch`, chặn mọi lời gọi `.cuda()`, fp32 thuần không AMP. Đã chạy trọn
vẹn hai lần bằng `nbconvert` trên máy cá nhân (12 threads, torch 2.13.0+cpu, Python 3.14), hồ sơ
**`cpu-full`**: toàn bộ 7.930 ảnh, đúng split `SPLIT_SEED = 42` (4.758 / 1.586 / 1.586 — trùng mọi
vòng GPU), **15 epoch, batch 16, chỉ seed 0**. Dữ liệu tải mới từ OSF (Google Drive chặn `gdown`),
MD5 khớp bản công bố (`90aabf906e153f7bac4548765402d4c7`), quét lớp tái lập đúng 27 thư mục /
8.000 ảnh → 22 lớp / 7.930.

### 11.1 macro-F1 test, seed 0, cả 6 quy tắc

| Cấu hình | best | smooth | top3 | best_tta | smooth_tta | top3_tta |
|---|---|---|---|---|---|---|
| `B0_densenet121_cpu` @224 | **0.6844** | 0.6665 | **0.6969** | 0.6764 | 0.6689 | 0.6883 |
| `M0_mobilenetv3_cpu` @224 | 0.6191 | 0.6191 | 0.6649 | 0.6434 | 0.6434 | 0.6737 |

Bootstrap CI95 (1.000 lần lấy lại mẫu, logits seed 0) cho DenseNet-121: `best` **[0.6343, 0.7204]**,
`top3` **[0.6495, 0.7328]**. Accuracy (micro-F1) của `M0` = 0.789.

### So với các mốc khác — đọc kèm cảnh báo giao thức

| Lần chạy | Giao thức | best | top3 |
|---|---|---|---|
| Paper DenseNet-121 (Table 2) | 150 ep, batch 32, TITAN Xp, 1 run | 0.6504 | — |
| `B0` T4 (§10.9, bộ số hiện hành) | 30 ep, batch 32, fp16, 3 seed | 0.6686 ± 0.0234 | 0.6780 |
| `B0_densenet121_cpu` | **15 ep, batch 16**, fp32 CPU, 1 seed | 0.6844 | 0.6969 |
| `P2` T4 (hệ thống đề xuất) | 80 ep công thức hiện đại, 3 seed | — | 0.7298 (hệ thống đầy đủ 0.7441) |

**Caveat bắt buộc khi trích 0.6844:** (1) **không cùng giao thức** — batch 16 ≠ 32 đổi cả LR hiệu
dụng lẫn số bước cập nhật, nên đây là *phép đo tính khả thi trên CPU*, không phải một lần tái lập
nữa; (2) **1 seed** — σ của `B0` trên T4 là ±0.0234, và Gate 0a đã chứng minh đổi phần cứng = đổi
mô hình, nên 0.6844 nằm gọn trong vùng "seed may mắn + giao thức khác"; (3) val đạt đỉnh **epoch
7/15** (0.6806) — 15 epoch không phải ràng buộc; (4) điều nó *gợi ý* (chưa chứng minh): batch nhỏ
đáng một dòng ablation có kiểm soát batch 16 vs 32 trên GPU. **Kết luận nhất quán với §10:** dưới
`top3_tta`, MobileNetV3 (4,2M tham số) chỉ thua DenseNet 0.015 (0.6737 vs 0.6883, CI chồng lấn
mạnh) ở **3,3× nhanh hơn** — đòn bẩy *cách đo* bù gần hết khoảng cách kiến trúc, đúng câu chuyện
của báo cáo.

Chi phí & số triển khai CPU (12 threads, fp32): DenseNet-121 train 225 phút (~810 s/epoch), suy
luận **72.1 ms/ảnh** @ batch 1; MobileNetV3-L train 82 phút, **21.7 ms/ảnh**, 16.9 MB. Artifact:
`checkpoints_cpu/*.npz|pt` (local, gitignore), notebook đã chạy + `outputs_cpu/` trong git.

### 11.2 Bằng chứng demo Gradio — chạy thật đầu-cuối trên CPU (2026-09-02)

Khoảng trống "không có screenshot demo" (ghi ở cuối §10 và BAO_CAO mục 7.2) được khép một nửa bằng
`report/demo/demo_gradio_cpu.py`: nạp `B0_densenet121_cpu_seed0.pt`, **tự kiểm đường suy luận trên
20 ảnh test thật trước khi dựng UI** (14/20 đúng — khớp accuracy 0.789 của vòng CPU), dựng Gradio
(top-5 + TTA lật ngang, đúng dạng đầu ra của §20b), rồi Playwright (Chromium headless) upload một
ảnh test, bấm Submit và chụp màn hình khi kết quả hiện ra (top-1 *Accessory tools* p = 0.961).

* Bằng chứng: `report/demo/29b_demo_gradio_cpu.png` (screenshot UI đầy đủ) + `29b_demo_gradio_cpu.txt` (log).
* Vẫn đúng hai caveat của BAO_CAO 7.2: demo là **1 checkpoint + TTA**, yếu hơn hệ thống được báo
  cáo, và checkpoint là của vòng CPU chứ **không phải** `P2` của vòng T4 — hai điều đó không đổi.
