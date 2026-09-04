# -*- coding: utf-8 -*-
"""Train them seed cho B0_densenet121_cpu, GIU DUNG giao thuc cua
notebooks/gastrovision_classification_cpu.ipynb (cpu-full: 15 epoch, batch 16,
fp32, CPU-only). Ghi ra checkpoints_cpu/B0_densenet121_cpu_seed<s>.npz|.pt.

Day la script da sinh ra seed 1, 2 cua bang RESULTS.md muc 11 (chay dem 04-09-2026,
~230 phut/seed tren CPU 12 threads). Resume: seed nao da co .npz thi bo qua.

Chay:  python train_cpu_seeds.py            # mac dinh seed 1, 2
       python train_cpu_seeds.py 3 4        # seed tuy chon
Sau do: python report/offline_tables_cpu.py # sinh lai bang tables-cpu/
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
assert not torch.cuda.is_available()
torch.set_num_threads(os.cpu_count() or 1)

REPO = Path(__file__).resolve().parent
CKPT_DIR = REPO / "checkpoints_cpu"
SPLIT_SEED, MIN_PER_CLASS, IMG_SIZE = 42, 25, 224
EPOCHS, BATCH_SIZE, NUM_WORKERS = 15, 16, 0
KEEP_TOP_K, SMOOTH_WIN = 3, 3
SEEDS_TO_TRAIN = [int(a) for a in sys.argv[1:]] or [1, 2]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ---------------- du lieu: dung het code cua notebook ----------------
raw = {}
for r, _, fs in os.walk(REPO / "data"):
    imgs = [f for f in fs if os.path.splitext(f)[1].lower() in IMG_EXT]
    if imgs:
        raw.setdefault(Path(r).name, []).extend(str(Path(r) / f) for f in imgs)
CLASSES = sorted(c for c, v in raw.items() if len(v) > MIN_PER_CLASS)
NUM_CLASSES = len(CLASSES)
assert NUM_CLASSES == 22
samples = [(p, i) for i, c in enumerate(CLASSES) for p in sorted(raw[c])]
labels = np.array([y for _, y in samples])
idx = np.arange(len(samples))
train_idx, tmp = train_test_split(idx, test_size=0.40, stratify=labels, random_state=SPLIT_SEED)
val_idx, test_idx = train_test_split(tmp, test_size=0.50, stratify=labels[tmp], random_state=SPLIT_SEED)
print(f"du lieu: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} | 22 lop", flush=True)

DATA_GEN = None


def seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s); random.seed(s)


def set_seed(s):
    global DATA_GEN
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s)
    torch.use_deterministic_algorithms(True, warn_only=True)
    DATA_GEN = torch.Generator(); DATA_GEN.manual_seed(s)


IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class GastroDataset(Dataset):
    def __init__(self, indices, transform):
        self.items = [samples[i] for i in indices]
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        return self.transform(Image.open(path).convert("RGB")), y


def make_loaders():
    common = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=False,
                  worker_init_fn=seed_worker)
    tr = DataLoader(GastroDataset(train_idx, train_tf), shuffle=True, generator=DATA_GEN, **common)
    va = DataLoader(GastroDataset(val_idx, eval_tf), shuffle=False, **common)
    te = DataLoader(GastroDataset(test_idx, eval_tf), shuffle=False, **common)
    return tr, va, te


ALL_LABELS = list(range(NUM_CLASSES))


def macro_f1(y, p):
    return f1_score(y, p, labels=ALL_LABELS, average="macro", zero_division=0)


def softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@torch.no_grad()
def evaluate(model, loader, tta=False):
    model.eval()
    ys, ls = [], []
    for x, yb in loader:
        out = model(x).float()
        if tta:
            out = (out + model(torch.flip(x, dims=[3])).float()) / 2
        ls.append(out.numpy()); ys.append(yb.numpy())
    logits = np.concatenate(ls).astype(np.float32)
    y_true = np.concatenate(ys)
    return dict(macro_f1=macro_f1(y_true, logits.argmax(1)), y_true=y_true, logits=logits)


class Tracker:
    def __init__(self, keep_top_k=KEEP_TOP_K, smooth_win=SMOOTH_WIN):
        self.keep_top_k, self.smooth_win = keep_top_k, smooth_win
        self.history, self.top = [], []
        self.smooth = (-1.0, -1, None)

    def update(self, model, vf1, ep):
        self.history.append(float(vf1))
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.top.append((float(vf1), ep, state))
        self.top.sort(key=lambda t: -t[0])
        del self.top[self.keep_top_k:]
        sm = float(np.mean(self.history[-self.smooth_win:]))
        if sm > self.smooth[0]:
            self.smooth = (sm, ep, state)

    def finalize(self, model):
        model.load_state_dict(self.top[0][2])
        return dict(best_val=self.top[0][0], best_epoch=self.top[0][1],
                    history=self.history, top=self.top, smooth=self.smooth)


def train_one(model, tr, va, epochs=EPOCHS, lr=1e-4):
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=1e-4)
    tk, dt = Tracker(), float("nan")
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        for x, yb in tr:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), yb)
            loss.backward()
            opt.step()
        dt = time.time() - t0
        vf1 = evaluate(model, va)["macro_f1"]
        tk.update(model, vf1, ep)
        print(f"  ep {ep + 1:02d}/{epochs}  val_macroF1={vf1:.4f}  (best={tk.top[0][0]:.4f})  {dt:.0f}s/epoch", flush=True)
    out = tk.finalize(model)
    out["sec_per_epoch"] = dt
    return out


def build_densenet121(nc):
    m = tv.models.densenet121(weights=tv.models.DenseNet121_Weights.IMAGENET1K_V1)
    m.classifier = nn.Linear(m.classifier.in_features, nc)
    return m


def ensemble_logits(list_of_logits):
    probs = [softmax_np(l) for l in list_of_logits]
    return sum(probs) / len(probs)


TAG = "B0_densenet121_cpu"
for s in SEEDS_TO_TRAIN:
    path = CKPT_DIR / f"{TAG}_seed{s}.npz"
    if path.exists():
        print(f"[{TAG} seed {s}] da co {path.name} -> bo qua", flush=True)
        continue
    set_seed(s)
    tr, va, te = make_loaders()
    model = build_densenet121(NUM_CLASSES)
    print(f"[{TAG} seed {s}] bat dau huan luyen ({EPOCHS} epoch, batch {BATCH_SIZE}, CPU)", flush=True)
    t0 = time.time()
    r = train_one(model, tr, va)
    train_sec = time.time() - t0

    cand = list(r["top"])
    smooth_state = r["smooth"][2]
    smooth_i = next((i for i, (_, _, st) in enumerate(cand) if st is smooth_state), None)
    if smooth_i is None:
        cand.append((r["smooth"][0], r["smooth"][1], smooth_state))
        smooth_i = len(cand) - 1

    L, y_true = {}, None
    for i, (_, _, st) in enumerate(cand):
        model.load_state_dict(st)
        for tta in (False, True):
            e = evaluate(model, te, tta=tta)
            L[(i, tta)] = e["logits"]
            y_true = e["y_true"]

    model.load_state_dict(cand[0][2])
    ev = evaluate(model, va)
    n_top = min(KEEP_TOP_K, len(r["top"]))

    sc, store = {}, {}
    for tta in (False, True):
        sfx = "_tta" if tta else ""
        ens = ensemble_logits([L[(i, tta)] for i in range(n_top)])
        store[f"logits_best{sfx}"] = L[(0, tta)]
        store[f"logits_smooth{sfx}"] = L[(smooth_i, tta)]
        store[f"probs_top{n_top}{sfx}"] = ens
        sc[f"best{sfx}"] = macro_f1(y_true, L[(0, tta)].argmax(1))
        sc[f"smooth{sfx}"] = macro_f1(y_true, L[(smooth_i, tta)].argmax(1))
        sc[f"top3{sfx}"] = macro_f1(y_true, ens.argmax(1))
    sc["best_val"] = r["best_val"]
    sc["train_sec"] = train_sec
    sc["sec_per_epoch"] = r.get("sec_per_epoch", float("nan"))

    np.savez_compressed(path, y_true=y_true, val_y=ev["y_true"], val_logits=ev["logits"],
                        history=np.array(r["history"], dtype=np.float32),
                        scores_json=json.dumps(sc), **store)
    torch.save(cand[0][2], CKPT_DIR / f"{TAG}_seed{s}.pt")
    print(f"[{TAG} seed {s}] xong sau {train_sec / 60:.1f} phut -> {path.name}", flush=True)
    print("   " + "  ".join(f"{k}={sc[k]:.4f}" for k in
                            ("best", "best_tta", "smooth", "smooth_tta", "top3", "top3_tta")), flush=True)

print("PLAN2_DONE", flush=True)
