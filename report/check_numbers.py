#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Doi chieu moi con so quan trong trong BAO_CAO.md voi nguon that trong tables/.

    python report/check_numbers.py

Bao cao viet so kieu Viet ("0,6961") con output cua notebook kieu chuan ("0.6961"),
nen phai chuan hoa dau phay -> dau cham truoc khi so.

Muc dich: bat truong hop sua so o mot cho ma quen cho khac. Khong thay the viec doc,
chi bat sai lech may moc.
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "BAO_CAO.md")
TABLES = os.path.join(HERE, "tables")

# (con so, file nguon phai chua no, mo ta)
CHECKS = [
    # --- muc 1: baseline & tai lap ---
    ("0.6504", None, "baseline cong bo (paper Table 2)"),
    ("0.6491", "12_B0_densenet121.txt", "tai lap B0 duoi quy tac 'best'"),
    ("0.0124", "12_B0_densenet121.txt", "sigma cua B0 duoi 'best'"),
    # --- muc 2: EDA ---
    ("7930", "04_loc_lop_22.txt", "so anh sau khi loc > 25"),
    ("4758", "05_chia_split.txt", "so anh train"),
    ("1586", "05_chia_split.txt", "so anh test"),
    ("1467", "06_eda.txt", "lop lon nhat"),
    ("50.6", "06_eda.txt", "ti le mat can bang"),
    ("0.38%", "08_audit_gan_trung.txt", "ti le anh test bi anh huong ro ri"),
    ("0.428608", "11_gate0a_tat_dinh.txt", "duong val Gate 0a tren A100"),
    # --- muc 3: cac lever ---
    ("0.6676", "16_bang_6_quy_tac.txt", "B0 duoi top3_tta"),
    ("0.6851", "16_bang_6_quy_tac.txt", "S0 duoi top3_tta"),
    ("0.6818", "16_bang_6_quy_tac.txt", "P0 duoi top3_tta"),
    ("0.6961", "16_bang_6_quy_tac.txt", "P1 duoi top3_tta"),
    ("0.6645", "16_bang_6_quy_tac.txt", "P1 duoi 'best'"),
    ("0.6732", "16_bang_6_quy_tac.txt", "P1 duoi best_tta (con so cua demo)"),
    ("0.7055", "23_he_thong_de_xuat_3seed.txt", "hieu chinh logit: mean sau khi ap"),
    ("0.0139", "23_he_thong_de_xuat_3seed.txt", "hieu chinh logit: sigma phong"),
    ("0.6749", "22_do_ben_truoc_ro_ri.txt", "B0 sau khi loc anh nghi ro ri"),
    # --- muc 4: per-class ---
    ("0.849", "18_per_class_va_confusion.txt", "accuracy = micro-F1 cua P1"),
    ("0.733", "18_per_class_va_confusion.txt", "macro precision cua P1"),
    ("0.674", "18_per_class_va_confusion.txt", "macro recall cua P1"),
    ("0.242", "18_per_class_va_confusion.txt", "F1 lop Cecum"),
    # --- muc 5: kien truc + CI + ensemble ---
    ("0.6548", "17_bootstrap_ci.txt", "CI duoi cua P1"),
    ("0.7245", "17_bootstrap_ci.txt", "CI tren cua P1"),
    ("0.6513", "17_bootstrap_ci.txt", "CI duoi cua S0"),
    ("0.7221", "23_he_thong_de_xuat_3seed.txt", "ensemble 3 seed cua P1"),
    ("0.7242", "20_donbay_ensemble_kientruc.txt", "ensemble 4 kien truc"),
    ("0.6879", "24_duong_hoc_val.txt", "val cao nhat cua P1"),
    ("0.0021", "24_duong_hoc_val.txt", "do lech 5 epoch cuoi cua P1"),
    # --- muc 6: transfer learning ---
    ("0.5725", "27_transfer_learning.txt", "T1 linear probe"),
    ("0.6463", "27_transfer_learning.txt", "T2 dong bang nua duoi"),
    ("0.6472", "27_transfer_learning.txt", "T3 progressive + LR phan biet"),
    ("0.0132", "27_transfer_learning.txt", "nguong 2 sigma"),
    # --- muc 7: trien khai ---
    ("18.9", "28_trien_khai_onnx_do_tre.txt", "DenseNet ms @ batch 1"),
    ("0.58", "28_trien_khai_onnx_do_tre.txt", "DenseNet ms @ batch 32"),
    ("13.2", "28_trien_khai_onnx_do_tre.txt", "CoAtNet@288 ms @ batch 1"),
    ("0.99", "28_trien_khai_onnx_do_tre.txt", "CoAtNet@288 ms @ batch 32"),
    ("114.8", "28_trien_khai_onnx_do_tre.txt", "kich thuoc ONNX cua P1"),
]


def norm(text):
    """0,6961 -> 0.6961 (chi doi dau phay nam GIUA hai chu so)."""
    return re.sub(r"(?<=\d),(?=\d)", ".", text)


def strip_thousands(text):
    """4.758 -> 4758, de so voi output cua notebook (khong dung dau phan cach nghin).

    Chi bo dau cham khi phan nguyen KHAC '0', nen '0.6961' khong bi cham vao.
    """
    return re.sub(r"\b([1-9]\d{0,2})\.(\d{3})\b", r"\1\2", text)


def main():
    raw = norm(io.open(REPORT, encoding="utf-8").read())
    report = raw + chr(10) + strip_thousands(raw)     # so ca hai cach viet so
    cache = {}
    bad = []
    for value, source, what in CHECKS:
        in_report = value in report
        if source is None:
            in_source = True
        else:
            if source not in cache:
                path = os.path.join(TABLES, source)
                if not os.path.exists(path):
                    bad.append(("THIEU FILE", source, what))
                    cache[source] = ""
                else:
                    cache[source] = norm(io.open(path, encoding="utf-8").read())
            in_source = value in cache[source]
        if not in_report:
            bad.append(("THIEU TRONG BAO CAO", value, what))
        elif not in_source:
            bad.append(("KHONG CO TRONG NGUON", "%s (%s)" % (value, source), what))

    print("da doi chieu %d con so" % len(CHECKS))
    if bad:
        for kind, value, what in bad:
            print("  !! %-22s %-28s %s" % (kind, value, what))
        sys.exit(1)
    print("tat ca khop: moi con so trong bao cao deu co trong tables/ va nguoc lai")


if __name__ == "__main__":
    main()
