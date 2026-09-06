#!/usr/bin/env python3
"""Regenerate the paper figures with English labels from the saved report tables.

Fig 1  fig_eda.png     -- per-class distribution (from ../tables/06_eda.txt)
Fig 2  fig_perclass_f1.png -- per-class F1 bar chart (from ../tables/18_per_class_va_confusion.txt)

Run with a matplotlib-enabled interpreter, e.g. a venv:
    python -m venv v && v/bin/pip install matplotlib && v/bin/python make_figures.py
"""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = __file__.rsplit("/", 1)[0]
TABLES = HERE + "/../tables"
OUT = HERE + "/figures"

# ---------------------------------------------------------------- Fig 1: EDA
rows = []
with open(f"{TABLES}/06_eda.txt") as f:
    for line in f:
        m = re.match(r"\s*(.+?)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", line)
        if not m:
            continue
        name, total, tr, va, te = m.group(1), *map(int, m.groups()[1:])
        if name.lower().startswith("lop"):   # header row
            continue
        rows.append((name, total, tr, va, te))

rows.sort(key=lambda r: r[1], reverse=True)
names = [r[0].replace("_", " ") for r in rows]
train = [r[2] for r in rows]
val = [r[3] for r in rows]
test = [r[4] for r in rows]
x = range(len(rows))

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.bar(x, train, label="train", color="#3b6fb0")
ax.bar(x, val, bottom=train, label="val", color="#7aa8d8")
ax.bar(x, [t + v for t, v in zip(train, val)], width=0)  # spacer keeps stacking clean
ax.bar(x, test, bottom=[t + v for t, v in zip(train, val)], label="test", color="#c9dcf0")
ax.set_yscale("log")
ax.set_ylabel("images (log scale)")
ax.set_title("Per-class image distribution over the 22 kept classes "
             "(imbalance ratio 50.6x)")
ax.set_xticks(list(x))
ax.set_xticklabels(names, rotation=90, fontsize=6.5)
ax.legend(loc="upper right", frameon=False)
ax.margins(x=0.01)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_eda.png", dpi=200)
plt.close(fig)
print("wrote fig_eda.png with", len(rows), "classes")

# ------------------------------------------------------ Fig 2: per-class F1
pc = []
with open(f"{TABLES}/18_per_class_va_confusion.txt") as f:
    for line in f:
        m = re.match(r"\s*(.+?)\s+([01]\.\d{3})\s+([01]\.\d{3})\s+([01]\.\d{3})\s+(\d+)\s*$", line)
        if not m:
            continue
        name = m.group(1).strip()
        if name in ("accuracy", "macro avg", "weighted avg"):
            continue
        prec, rec, f1, sup = float(m.group(2)), float(m.group(3)), float(m.group(4)), int(m.group(5))
        pc.append((name.replace("_", " "), f1, sup))

pc.sort(key=lambda r: r[1])           # weakest at the bottom
names = [r[0] for r in pc]
f1s = [r[1] for r in pc]
sup = [r[2] for r in pc]
# colour rare (<50 test images) vs common
colors = ["#c44e52" if s < 50 else "#3b6fb0" for s in sup]

fig, ax = plt.subplots(figsize=(8.4, 6.6))
y = range(len(pc))
ax.barh(list(y), f1s, color=colors)
for i, (v, s) in enumerate(zip(f1s, sup)):
    ax.text(v + 0.008, i, f"{v:.2f} (n={s})", va="center", fontsize=6.5)
ax.set_yticks(list(y))
ax.set_yticklabels(names, fontsize=7)
ax.set_xlim(0, 1.12)
ax.set_xlabel("per-class F1  (proposed model P2, seed 0; macro-F1 = 0.717)")
ax.axvline(0.717, color="gray", ls="--", lw=0.8)
ax.set_title("Per-class F1 of the proposed model\n"
             "red = rare classes (<50 test images), blue = common")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_perclass_f1.png", dpi=200)
plt.close(fig)
print("wrote fig_perclass_f1.png with", len(pc), "classes")
