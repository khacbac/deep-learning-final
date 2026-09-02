"""Bang chung demo Gradio — chay that tren CPU voi checkpoint da huan luyen.

Nap `checkpoints_cpu/B0_densenet121_cpu_seed0.pt` (DenseNet-121, macro-F1 test 0.6844
quy tac `best`, xem RESULTS.md muc 11), tu kiem tra duong suy luan tren mot anh test
that TRUOC khi dung UI (dung ky luat cua §20b notebook GPU), roi:
  1. dung Gradio Interface (top-5 xac suat + TTA lat ngang),
  2. dung Playwright (Chromium headless) upload chinh anh test do, bam Submit,
  3. chup man hinh UI sau khi co ket qua -> 29b_demo_gradio_cpu.png,
  4. ghi log -> 29b_demo_gradio_cpu.txt.

Chay tu goc repo:  python report/demo/demo_gradio_cpu.py
Yeu cau: data/ da giai nen (8.000 anh), gradio, playwright (+ chromium).
"""
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # demo CPU-only, nhu notebook CPU

import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
CKPT = REPO / "checkpoints_cpu" / "B0_densenet121_cpu_seed0.pt"
DATA_DIR = REPO / "data"
SPLIT_SEED, MIN_PER_CLASS, IMG_SIZE = 42, 25, 224
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

log_lines = []


def log(s):
    print(s)
    log_lines.append(s)


# --- Dung lai DUNG phep chia cua notebook (SPLIT_SEED=42, loc >25 anh -> 22 lop) ---
raw = {}
for r, _, fs in os.walk(DATA_DIR):
    imgs = [f for f in fs if os.path.splitext(f)[1].lower() in IMG_EXT]
    if imgs:
        raw.setdefault(Path(r).name, []).extend(str(Path(r) / f) for f in imgs)
CLASSES = sorted(c for c, v in raw.items() if len(v) > MIN_PER_CLASS)
assert len(CLASSES) == 22, f"ky vong 22 lop, thay {len(CLASSES)}"
samples = [(p, i) for i, c in enumerate(CLASSES) for p in sorted(raw[c])]
labels = np.array([y for _, y in samples])
idx = np.arange(len(samples))
_, tmp = train_test_split(idx, test_size=0.40, stratify=labels, random_state=SPLIT_SEED)
_, test_idx = train_test_split(tmp, test_size=0.50, stratify=labels[tmp], random_state=SPLIT_SEED)

# --- Nap mo hinh + checkpoint ---
model = tv.models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, len(CLASSES))
state = torch.load(CKPT, map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()
log(f"demo: da nap B0_densenet121_cpu seed 0 @ {IMG_SIZE}px tren cpu")
log(f"checkpoint: {CKPT.name} ({CKPT.stat().st_size / 1e6:.1f} MB)")

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@torch.no_grad()
def predict(pil_img):
    """Top-5 xac suat, TTA lat ngang — dung cong thuc `best_tta` cua notebook."""
    x = tf(pil_img.convert("RGB")).unsqueeze(0)
    logits = (model(x) + model(torch.flip(x, dims=[3]))) / 2
    probs = torch.softmax(logits[0], dim=0)
    top = torch.topk(probs, 5)
    return {CLASSES[i]: float(p) for p, i in zip(top.values, top.indices)}


# --- Tu kiem tra duong suy luan tren anh test THAT truoc khi dung UI ---
sc_path, sc_true, sc_pred = None, None, None
for ti in test_idx[:20]:
    path, y = samples[ti]
    pred = predict(Image.open(path))
    top1 = max(pred, key=pred.get)
    ok = "DUNG" if top1 == CLASSES[y] else "SAI"
    log(f"self-check: {Path(path).name} | that: {CLASSES[y]} | doan: {top1} "
        f"(p={pred[top1]:.3f}) -> {ok}")
    if top1 == CLASSES[y] and sc_path is None:
        sc_path, sc_true, sc_pred = path, CLASSES[y], pred
if sc_path is None:                      # khong anh nao dung trong 20 anh dau (rat kho xay ra)
    sc_path, y = samples[test_idx[0]]
    sc_true, sc_pred = CLASSES[y], predict(Image.open(sc_path))
log(f"anh dung cho UI: {sc_path}")
log("top-5: " + ", ".join(f"{k}={v:.3f}" for k, v in sc_pred.items()))

# --- Dung UI Gradio ---
import gradio as gr

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Ảnh nội soi"),
    outputs=gr.Label(num_top_classes=5, label="Top-5 dự đoán (TTA lật ngang)"),
    title="GastroVision — DenseNet-121 (CPU)",
    description=(
        "Phân loại 22 lớp nội soi tiêu hoá. Checkpoint B0_densenet121_cpu_seed0 "
        "(macro-F1 test 0,6844 quy tắc best — hệ thống đầy đủ của báo cáo là P2+top3, "
        "mạnh hơn demo này; xem BAO_CAO.md mục 7.2). Suy luận 100% CPU, ~72 ms/ảnh."
    ),
    flagging_mode="never",
)
demo.launch(prevent_thread_lock=True, server_name="127.0.0.1", server_port=7861,
            quiet=True, share=False)
url = "http://127.0.0.1:7861"
log(f"gradio: UI phuc vu tai {url}")

# --- Playwright: upload anh test, bam Submit, doi ket qua, chup man hinh ---
from playwright.sync_api import sync_playwright

shot = OUT_DIR / "29b_demo_gradio_cpu.png"
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1440, "height": 1000})
    pg.goto(url, wait_until="networkidle")
    pg.set_input_files("input[type=file]", sc_path)
    pg.wait_for_timeout(1500)
    pg.get_by_role("button", name="Submit").click()
    top1 = max(sc_pred, key=sc_pred.get)
    pg.wait_for_selector(f"text={top1}", timeout=30000)
    pg.wait_for_timeout(1000)                      # cho thanh xac suat ve xong
    pg.screenshot(path=str(shot), full_page=True)
    b.close()
log(f"screenshot: {shot.name} ({shot.stat().st_size / 1024:.0f} KB)")

demo.close()
(OUT_DIR / "29b_demo_gradio_cpu.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
print("DONE")
