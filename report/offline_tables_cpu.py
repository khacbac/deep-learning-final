# -*- coding: utf-8 -*-
"""Tinh lai 0-GPU cac bang cua VONG CPU tu ../checkpoints_cpu/*.npz -> tables-cpu/.

Cung ky luat voi offline_tables.py: moi con so cua BAO_CAO muc 7.3 va RESULTS.md
muc 11 duoc trich tu file do script nay ghi ra, khong go tay.

  37_b0_cpu_3seed.txt          — B0 CPU 3 seed x 6 quy tac, mean±sigma, ensemble
                                 3 seed (he thong tuyen bo truoc) + bootstrap CI
  38_b0_cpu_zero_epoch.txt     — cac lever 0-epoch tren B0 CPU seed 0: hieu chinh
                                 logit (tau do tren VAL), ensemble 2 kien truc

Chay:  python report/offline_tables_cpu.py
Can:   checkpoints_cpu/B0_densenet121_cpu_seed{0,1,2}.npz, M0_mobilenetv3_cpu_seed0.npz,
       va data/ (chi de dem lai TRAIN_CLASS_COUNTS cho hieu chinh logit).
"""
import io
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE / "tables-cpu"
OUT.mkdir(exist_ok=True)
CKPT = REPO / "checkpoints_cpu"
BASE = 0.6504
ALL = list(range(22))
SPLIT_SEED, MIN_PER_CLASS = 42, 25
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def mf1(y, p):
    return f1_score(y, p, labels=ALL, average="macro", zero_division=0)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def as_probs(a):
    a = np.asarray(a, dtype=np.float64)
    return a if (a.min() >= 0 and np.allclose(a.sum(1), 1.0, atol=1e-3)) else softmax(a)


def as_logits(a):
    a = np.asarray(a, dtype=np.float64)
    return np.log(a + 1e-12) if (a.min() >= 0 and np.allclose(a.sum(1), 1.0, atol=1e-3)) else a


def boot_ci(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    st = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        pres = np.unique(y[i])
        st.append(f1_score(y[i], p[i], labels=pres, average="macro", zero_division=0))
    lo, hi = np.percentile(st, [2.5, 97.5])
    return float(lo), float(hi)


D = {s: dict(np.load(CKPT / f"B0_densenet121_cpu_seed{s}.npz", allow_pickle=True)) for s in (0, 1, 2)}
M0 = dict(np.load(CKPT / "M0_mobilenetv3_cpu_seed0.npz", allow_pickle=True))
y = D[0]["y_true"]
assert all(np.array_equal(D[s]["y_true"], y) for s in (1, 2))

KEY = {"best": "logits_best", "best_tta": "logits_best_tta", "smooth": "logits_smooth",
       "smooth_tta": "logits_smooth_tta", "top3": "probs_top3", "top3_tta": "probs_top3_tta"}
RULES = list(KEY)

# ============================ 37: 3 seed + ensemble ============================ #
buf = io.StringIO()
w = lambda s="": buf.write(s + "\n")
w("B0_densenet121_cpu — 3 seed, giao thuc CPU (15 epoch, batch 16, fp32, 12 threads)")
w(f"Baseline cong bo (DenseNet-121, arXiv 2307.08140): {BASE}")
w()
w(f"{'quy_tac':12s} {'seed0':>7s} {'seed1':>7s} {'seed2':>7s} {'mean':>7s} {'sigma':>7s}")
for r in RULES:
    vals = [mf1(y, np.asarray(D[s][KEY[r]]).argmax(1)) for s in (0, 1, 2)]
    w(f"{r:12s} {vals[0]:7.4f} {vals[1]:7.4f} {vals[2]:7.4f} {np.mean(vals):7.4f} {np.std(vals):7.4f}")
top3_vals = [mf1(y, np.asarray(D[s]['probs_top3']).argmax(1)) for s in (0, 1, 2)]
w()
w(f"mean top3 - baseline = {np.mean(top3_vals) - BASE:+.4f}"
  f"  ({(np.mean(top3_vals) - BASE) / max(np.std(top3_vals), 1e-9):.1f} sigma)")
w()
w("HE THONG (tuyen bo truoc): ensemble 3 seed cua top3 (trung binh xac suat)")
ens = sum(as_probs(D[s]["probs_top3"]) for s in (0, 1, 2)) / 3
pred = ens.argmax(1)
lo, hi = boot_ci(y, pred)
w(f"  macro-F1 = {mf1(y, pred):.4f}   CI95 [{lo:.4f}, {hi:.4f}]"
  f"   -> {'KHONG chua' if lo > BASE else 'CHUA loai duoc'} {BASE}")
w(f"  micro-F1 (accuracy) = {float((pred == y).mean()):.4f}   (bai bao: 0.8203)")
w()
w("Doi chieu (exploratory):")
for name, arr in (("ensemble 3 seed top3_tta", sum(as_probs(D[s]["probs_top3_tta"]) for s in (0, 1, 2)) / 3),
                  ("ensemble 3 seed best", sum(as_probs(D[s]["logits_best"]) for s in (0, 1, 2)) / 3)):
    p = arr.argmax(1)
    l, h = boot_ci(y, p)
    w(f"  {name:26s}: {mf1(y, p):.4f}  CI95 [{l:.4f}, {h:.4f}]")
for s in (0, 1, 2):
    p = np.asarray(D[s]["probs_top3"]).argmax(1)
    l, h = boot_ci(y, p)
    w(f"  seed {s} top3 don le        : {mf1(y, p):.4f}  CI95 [{l:.4f}, {h:.4f}]")
w()
w("Doi chieu T4 (tu tables/21_bang_tong_ket.txt): B0 top3 = 0.6780 +/- 0.0073, best = 0.6686 +/- 0.0234")
w("MobileNetV3-L CPU seed 0 (tu M0_mobilenetv3_cpu_seed0.npz):")
for r in ("best", "top3", "top3_tta"):
    w(f"  {r:9s} = {mf1(y, np.asarray(M0[KEY[r]]).argmax(1)):.4f}")
(OUT / "37_b0_cpu_3seed.txt").write_text(buf.getvalue(), encoding="utf-8")
print("da ghi 37_b0_cpu_3seed.txt")

# ===================== 38: lever 0-epoch tren seed 0 ====================== #
raw = {}
for r, _, fs in os.walk(REPO / "data"):
    imgs = [f for f in fs if os.path.splitext(f)[1].lower() in IMG_EXT]
    if imgs:
        raw.setdefault(Path(r).name, []).extend(imgs)
CLASSES = sorted(c for c, v in raw.items() if len(v) > MIN_PER_CLASS)
labels = np.array([i for i, c in enumerate(CLASSES) for _ in sorted(raw[c])])
tr_idx, _ = train_test_split(np.arange(len(labels)), test_size=0.40,
                             stratify=labels, random_state=SPLIT_SEED)
COUNTS = np.bincount(labels[tr_idx], minlength=22)


def adjust(lg, tau):
    prior = COUNTS / COUNTS.sum()
    return lg - tau * np.log(prior + 1e-12)[None, :]


buf = io.StringIO()
w = lambda s="": buf.write(s + "\n")
w("Lever 0-epoch tren B0_densenet121_cpu seed 0 (doc lai tu logits da luu)")
w()
vy, vlg = D[0]["val_y"], D[0]["val_logits"].astype(np.float64)
sweep = [(float(t), mf1(vy, adjust(vlg, t).argmax(1))) for t in np.arange(0.0, 1.01, 0.1)]
tau = max(sweep, key=lambda x: x[1])[0]
w(f"Do tau tren VAL: tau* = {tau:.1f}  (val {sweep[0][1]:.4f} -> {max(s for _, s in sweep):.4f})")
w()
w(f"{'quy_tac':12s} {'raw':>8s} {'+adjust':>9s}")
for r in ("best", "best_tta", "top3", "top3_tta"):
    lg = as_logits(D[0][KEY[r]])
    w(f"{r:12s} {mf1(y, lg.argmax(1)):8.4f} {mf1(y, adjust(lg, tau).argmax(1)):9.4f}")
w()
w("-> hieu chinh logit PHANG/AM tren B0 CPU (khong co cong thuc hien dai) — tai xac")
w("   nhan doc lap phat hien 3 cua RESULTS.md muc 10.9: lever nay chi tra tien tren P2.")
w()
w("Ensemble 2 kien truc (exploratory, trong so bang nhau):")
for k in ("probs_top3", "probs_top3_tta"):
    e = (as_probs(D[0][k]) + as_probs(M0[k])) / 2
    w(f"  B0+M0 {k:14s}: {mf1(y, e.argmax(1)):.4f}   (B0 don: {mf1(y, as_probs(D[0][k]).argmax(1)):.4f})")
(OUT / "38_b0_cpu_zero_epoch.txt").write_text(buf.getvalue(), encoding="utf-8")
print("da ghi 38_b0_cpu_zero_epoch.txt")
