#!/usr/bin/env python
"""Trich hinh + bang tu output da luu trong notebook ra thu muc report/.

Chay lai sau MOI phien Colab:
    python report/extract.py

Cell duoc nhan dien theo DONG DAU cua source (notebook duoc sinh tu
build_notebook.py nen chuoi nay on dinh), khong theo chi so cell -- them cell
moi o giua se khong lam lech mapping. Script bao loi neu khong tim thay.
"""
import base64, io, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, os.pardir, "notebooks", "gastrovision_classification.ipynb")
CR, LF = chr(13), chr(10)

# ten file  ->  dong dau cua source cell (khop tien to)
TABLES = [
    ("00_gpu",                      "# Kiem tra GPU truoc khi lam bat cu viec gi khac."),
    ("01_moi_truong",               "import os, sys, io, json, math, time, random, hashlib"),
    ("02_ho_so_chay",               "# --------------------------- CAU HINH THUC NGHIEM"),
    ("03_quet_anh",                 "IMG_EXT = {"),
    ("04_loc_lop_22",               "# --- Quet thu muc lop theo DE QUY + loc theo luat cua bai bao"),
    ("05_chia_split",               "# --- Chia phan tang 60:20:20, CO DINH SPLIT_SEED cho ca nhom ---"),
    ("06_eda",                      "cnt_full = np.bincount(labels, minlength=NUM_CLASSES)"),
    ("07_audit_md5",                "# --- Lop 1: trung byte (MD5) ---"),
    ("08_audit_gan_trung",          "# --- Lop 2: gan trung (cosine tren embedding da pretrain) ---"),
    ("09_bo_danh_gia",              "ALL_LABELS = list(range(NUM_CLASSES))"),
    ("10_ba_kien_truc",             "def build_densenet121(nc):"),
    ("11_gate0a_tat_dinh",          "RUN_DETERMINISM_CHECK = True"),
    ("12_B0_densenet121",           "set_img_size(224)"),
    ("13_S0_swin_t",                "res_s0 = run_seeds(build_swin_t"),
    ("14_P0_coatnet0",              "res_p0 = run_seeds(build_coatnet0"),
    ("15_P1_coatnet0_288",          "RUN_P1_288 = True"),
    ("16_bang_6_quy_tac",           "rows = []"),
    ("17_bootstrap_ci",             "# Thanh sai so bootstrap tren seed dau tien"),
    ("18_per_class_va_confusion",   "FOCUS = max(RESULTS_STORE"),
    ("19_donbay_hieuchinh_logit",   'print(f"=== Don bay 2: hieu chinh logit'),
    ("20_donbay_ensemble_kientruc", 'print("=== Don bay 3: ensemble nhieu kien truc'),
    ("21_bang_tong_ket",            "summary = []"),
    ("22_do_ben_truoc_ro_ri",       "# --- (1) Bo cac anh test bi nghi ro ri roi tinh lai"),
    ("23_he_thong_de_xuat_3seed",   "# --- (2) HE THONG DE XUAT DAY DU tren ca 3 seed ---"),
    ("24_duong_hoc_val",            "# --- Duong hoc val: chan doan overfit"),
    ("25_ablation_tuy_chon",        "RUN_ABLATIONS = "),
    ("26_transfer_learning_log",    "RUN_TRANSFER = "),
    ("27_transfer_learning",        "# --- Bang so sanh: 4 dieu kien, cung backbone / split / seed / so epoch ---"),
    ("28_trien_khai_onnx_do_tre",   "# Tra ve ms cho MOI ANH."),
    ("29_demo_gradio",              "RUN_DEMO = "),
]


def find(cells, prefix):
    hits = [i for i, c in enumerate(cells)
            if c["cell_type"] == "code" and "".join(c["source"]).lstrip().startswith(prefix)]
    if len(hits) != 1:
        sys.exit("khong khop duy nhat mot cell cho prefix %r (tim thay %s)" % (prefix, hits))
    return hits[0]


def flatten_cr(text):
    """Dung nhu terminal: moi lan CR la ve lai dong -> chi giu doan cuoi.

    Khong lam viec nay thi cac thanh tien trinh (tqdm, tai model tu HF) bi noi
    thanh mot dong dai va che mat dong ket qua thuc su nam ngay sau chung.
    """
    return LF.join(line.split(CR)[-1] for line in text.split(LF))


def trim(text):
    """Bo dong trong o hai dau, GIU nguyen thut cua dong dau (header pandas)."""
    lines = [l.rstrip() for l in text.split(LF)]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return LF.join(lines)


def is_progress(s):
    """Thanh tien trinh cua tqdm / huggingface_hub -- chrome cua terminal, khong phai so lieu.

    Phai loc TRUOC khi noi chuoi: anh chup widget cua HF khong co newline o cuoi
    nen neu de lai thi no dinh vao dau dong ket qua ngay sau do.
    """
    return ("%|" in s) or ("reconstructing file" in s) or ("downloading bytes" in s)


def render_text(cell):
    """stdout truoc, roi stderr; bo qua repr cua Figure / object IPython."""
    out, err = [], []
    for o in cell.get("outputs", []) or []:
        t = o.get("output_type")
        if t == "stream":
            text = flatten_cr("".join(o["text"]))
            text = LF.join(l for l in text.split(LF) if not is_progress(l))
            (err if o.get("name") == "stderr" else out).append(text)
        elif t in ("execute_result", "display_data"):
            s = "".join(o.get("data", {}).get("text/plain", []))
            if s.startswith("<Figure") or s.startswith("<IPython") or is_progress(s):
                continue
            out.append(s)
        elif t == "error":
            err.append("%s: %s%s" % (o.get("ename"), o.get("evalue"), LF))
    return trim(flatten_cr("".join(out) + "".join(err))) + LF


def pngs(cell):
    for o in cell.get("outputs", []) or []:
        d = o.get("data", {}) if o.get("output_type") in ("display_data", "execute_result") else {}
        if "image/png" in d:
            yield base64.b64decode("".join(d["image/png"]))


def main():
    nb = json.load(io.open(NB, encoding="utf-8"))
    cells = nb["cells"]
    for d in ("tables", "figures"):
        p = os.path.join(HERE, d)
        os.makedirs(p, exist_ok=True)
        for f in os.listdir(p):                       # xoa file cu de doi ten khong de lai rac
            os.remove(os.path.join(p, f))

    n_fig = 0
    for name, prefix in TABLES:
        cell = cells[find(cells, prefix)]
        txt = render_text(cell)
        if txt.strip() == "":
            print("  (trong)  %s -- cell chua chay?" % name)
        with io.open(os.path.join(HERE, "tables", name + ".txt"), "w",
                     encoding="utf-8", newline="\n") as fh:
            fh.write(txt)
        for k, blob in enumerate(pngs(cell)):
            fn = "%s.png" % name if k == 0 else "%s_%d.png" % (name, k + 1)
            open(os.path.join(HERE, "figures", fn), "wb").write(blob)
            n_fig += 1
    print("da ghi %d bang + %d hinh" % (len(TABLES), n_fig))


if __name__ == "__main__":
    main()
