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
# `source` cua moi CHECK la duong dan tuong doi TRONG report/, nen no chi ro bang do thuoc
# vong chay nao: tables/ = vong T4 hien tai, tables-a100/ = vong A100 (chi con Gate 0a +
# demo + bang do tre), tables-offline/ = tinh lai tu .npz. Truoc day moi thu mac dinh la
# tables/, nen khong the neo mot con so vao ban A100 ma khong noi doi ve nguon cua no.
ROOT = HERE

# (con so, file nguon phai chua no, mo ta)
CHECKS = [
    # --- muc 1: baseline & tai lap (vong T4) ---
    ("0.6504", None, "baseline cong bo (paper Table 2)"),
    ("0.6686", "tables/16_bang_6_quy_tac.txt", "tai lap B0 duoi quy tac 'best'"),
    ("0.0234", "tables-offline/30_lap_lai_a100_vs_t4.txt", "sigma cua B0 duoi 'best'"),
    ("0.6491", None, "lan do A100 cu -- co tinh khong con nguon, chi neu trong van"),
    # --- muc 2: EDA ---
    ("7930", "tables/04_loc_lop_22.txt", "so anh sau khi loc > 25"),
    ("4758", "tables/05_chia_split.txt", "so anh train"),
    ("1586", "tables/05_chia_split.txt", "so anh test"),
    ("1467", "tables/06_eda.txt", "lop lon nhat"),
    ("50.6", "tables/06_eda.txt", "ti le mat can bang"),
    ("0.38%", "tables/08_audit_gan_trung.txt", "ti le anh test bi anh huong ro ri"),
    ("0.428608", "tables-a100/11_gate0a_tat_dinh.txt", "duong val Gate 0a tren A100"),
    # Duong val T4 cua Gate 0a do 27-08 va KHONG duoc trich ra bang nao (o Gate 0a cua vong do
    # chay tren A100). Nguon duy nhat la ../RESULTS.md muc 3, nen neo source=None nhu 0.6504.
    ("0.430615", None, "duong val Gate 0a tren T4 -- nguon: ../RESULTS.md muc 3"),
    # --- muc 2.5: lap lai A100 <-> T4 ---
    ("0.0046", "tables-offline/30_lap_lai_a100_vs_t4.txt", "lech phan cung cua top3"),
    ("0.0182", "tables-offline/30_lap_lai_a100_vs_t4.txt", "lech phan cung cua 'best'"),
    ("0.0717", "tables-offline/30_lap_lai_a100_vs_t4.txt", "lech phan cung lon nhat 1 seed"),
    # --- muc 3: cac lever, quy tac 'top3' ---
    ("0.6780", "tables/16_bang_6_quy_tac.txt", "B0 duoi top3"),
    ("0.6813", "tables/16_bang_6_quy_tac.txt", "S0 duoi top3"),
    ("0.6814", "tables/16_bang_6_quy_tac.txt", "P0 duoi top3"),
    ("0.6855", "tables/16_bang_6_quy_tac.txt", "P1 duoi top3"),
    ("0.7298", "tables/16_bang_6_quy_tac.txt", "P2 duoi top3"),
    ("0.6670", "tables/16_bang_6_quy_tac.txt", "P2b duoi top3, 3 seed"),
    ("0.7172", "tables/16_bang_6_quy_tac.txt", "P2c duoi top3"),
    ("0.0119", "tables/21_bang_tong_ket.txt", "sigma cua P2b tren 3 seed"),
    ("0.7199", "tables/16_bang_6_quy_tac.txt", "P2 duoi best_tta (con so cua demo)"),
    # --- muc 3.2: cong thuc hien dai + doi chung P2b ---
    ("0.0471", "tables-offline/35_bang_2x2_tuong_tac.txt", "don bay cong thuc @288 duoi top3_tta"),
    ("0.0110", "tables/17b_so_sanh_theo_cap.txt", "don bay cong thuc tren DenseNet @224 (am), 3 seed"),
    ("0.0086", "tables/17b_so_sanh_theo_cap.txt", "P2b - B0 seed 1"),
    ("0.0200", "tables/17b_so_sanh_theo_cap.txt", "P2b - B0 seed 2"),
    # --- muc 3.2: bang 2x2 va so hang tuong tac (SESSION = 4) ---
    ("0.0358", "tables-offline/35_bang_2x2_tuong_tac.txt", "don bay cong thuc tren CoAtNet @224"),
    ("0.0468", "tables-offline/35_bang_2x2_tuong_tac.txt", "so hang cong thuc x kien truc"),
    ("0.0085", "tables-offline/35_bang_2x2_tuong_tac.txt", "so hang cong thuc x do phan giai"),
    ("0.0497", "tables-offline/35_bang_2x2_tuong_tac.txt", "so hang tuong tac tren cung seed 0"),
    ("0.0454", "tables-offline/35_bang_2x2_tuong_tac.txt", "cong thuc tren hybrid @224, seed 0"),
    ("0.0529", "tables-offline/35_bang_2x2_tuong_tac.txt", "tuong tac doc lai duoi top3_tta"),
    ("0.0083", "tables-offline/35_bang_2x2_tuong_tac.txt", "do phan giai doc lai duoi top3_tta"),
    ("0.0160", "tables/17b_so_sanh_theo_cap.txt", "doi mot minh kien truc, seed 0 (am)"),
    ("0.0294", "tables/17b_so_sanh_theo_cap.txt", "P2c - B0 seed 0"),
    ("0.0018", "tables/17b_so_sanh_theo_cap.txt", "McNemar p cua P2c vs B0"),
    ("0.6664", "tables/17_bootstrap_ci.txt", "CI duoi cua P2c"),
    ("0.7568", "tables/17_bootstrap_ci.txt", "CI tren cua P2c"),
    # --- muc 3.4: do phan giai duoi cong thuc moi ---
    ("0.0006", "tables-offline/35_bang_2x2_tuong_tac.txt", "288 vs 224 duoi cong thuc moi (am)"),
    # --- muc 3.3 / 3.7 / 4.6: mat can bang va du lieu pretrain ---
    ("0.6831", "tables/25_ablation_tuy_chon.txt", "A2 balanced softmax"),
    ("0.0047", "tables/25_ablation_tuy_chon.txt", "A2 so voi B0 (am)"),
    ("0.7028", "tables/25_ablation_tuy_chon.txt", "A1 Swin-T pretrain IN-22k"),
    ("0.0254", "tables/25_ablation_tuy_chon.txt", "A1 so voi S0"),
    ("0.7091", "tables/19_donbay_hieuchinh_logit.txt", "B0 sau hieu chinh logit"),
    ("0.0213", "tables/19_donbay_hieuchinh_logit.txt", "don bay hieu chinh logit tren B0"),
    ("0.0117", "tables/19_donbay_hieuchinh_logit.txt", "don bay hieu chinh logit tren P0"),
    ("0.0176", "tables/19_donbay_hieuchinh_logit.txt", "don bay hieu chinh logit tren P1, seed 0"),
    ("0.0065", "tables/19_donbay_hieuchinh_logit.txt", "don bay hieu chinh logit tren P2b"),
    ("0.0276", "tables/19_donbay_hieuchinh_logit.txt", "don bay hieu chinh logit tren P2, seed 0"),
    # --- muc 2.1: phep quet de quy nhat nham thu muc output ---
    ("8006", "tables/04_loc_lop_22.txt", "so anh quet duoc o phien SESSION = 4"),
    ("0.6539", "tables/24_duong_hoc_val.txt", "dinh val cua P2b -- thap hon B0"),
    ("0.6600", "tables/24_duong_hoc_val.txt", "dinh val cua B0"),
    # --- muc 3.6: loc ro ri ---
    ("0.7149", "tables/22_do_ben_truoc_ro_ri.txt", "P2 sau khi loc anh nghi ro ri"),
    # --- muc 3.7 + 5.3: hieu chinh logit tren P2 ---
    ("0.7441", "tables-offline/31_he_thong_p2_hieu_chinh_logit.txt", "he thong de xuat"),
    ("0.0088", "tables-offline/31_he_thong_p2_hieu_chinh_logit.txt", "sigma cua he thong de xuat"),
    ("0.6986", "tables-offline/31_he_thong_p2_hieu_chinh_logit.txt", "CI duoi cua he thong de xuat"),
    ("0.7736", "tables-offline/31_he_thong_p2_hieu_chinh_logit.txt", "CI tren cua he thong de xuat"),
    ("0.7587", "tables-offline/31_he_thong_p2_hieu_chinh_logit.txt", "ensemble 3 seed cua P2"),
    ("0.7442", "tables/19_donbay_hieuchinh_logit.txt", "P2 sau hieu chinh logit, seed 0"),
    ("0.6907", "tables-offline/31_he_thong_p2_hieu_chinh_logit.txt", "P1 sau hieu chinh logit (top3)"),
    ("0.0095", "tables-offline/31_he_thong_p2_hieu_chinh_logit.txt", "sigma cua P1 sau hieu chinh"),
    ("0.6855", "tables-offline/31_he_thong_p2_hieu_chinh_logit.txt", "P1 raw duoi top3"),
    ("0.0139", None, "sigma cua P1 sau hieu chinh o vong A100 -- nguon: ../RESULTS.md muc 10.8"),
    # --- muc 4: per-class ---
    ("0.850", "tables/18_per_class_va_confusion.txt", "accuracy = micro-F1 cua P2"),
    ("0.810", "tables/18_per_class_va_confusion.txt", "macro precision cua P2"),
    ("0.690", "tables/18_per_class_va_confusion.txt", "macro recall cua P2"),
    ("0.364", "tables/18_per_class_va_confusion.txt", "F1 lop Cecum"),
    ("0.286", "tables/18_per_class_va_confusion.txt", "F1 lop Mucosal inflammation"),
    ("90.4%", "tables-offline/32_per_class_vs_paper_table3.txt", "phan gop cua lop hiem"),
    ("0.313", "tables-offline/32_per_class_vs_paper_table3.txt", "muc tang lon nhat: Resected polyps"),
    ("0.0097", "tables-offline/32_per_class_vs_paper_table3.txt", "phan keo cua Mucosal inflammation"),
    # --- muc 4.6: du dia con lai ---
    ("0.2834", "tables-offline/34_du_dia_con_lai.txt", "khoang cach den tran 1,0"),
    ("85.2%", "tables-offline/34_du_dia_con_lai.txt", "phan du dia o 15 lop hiem"),
    ("0.2415", "tables-offline/34_du_dia_con_lai.txt", "gop cua 15 lop hiem"),
    ("0.0420", "tables-offline/34_du_dia_con_lai.txt", "TRAN cua 7 lop pho bien -- sat nguong 0,035"),
    ("0.0552", "tables-offline/34_du_dia_con_lai.txt", "gop cua rieng 2 lop 6 anh"),
    ("73.5", "tables-offline/34_du_dia_con_lai.txt", "anh train TB cua lop hiem"),
    # --- muc 5: kien truc + CI ---
    ("0.6412", "tables/17_bootstrap_ci.txt", "CI duoi cua B0"),
    ("0.6660", "tables/17_bootstrap_ci.txt", "CI duoi cua P2 -- khong chong lan 0,6504"),
    ("0.7531", "tables/17_bootstrap_ci.txt", "CI tren cua P2"),
    ("0.0443", "tables/17b_so_sanh_theo_cap.txt", "P2 - P1 theo cap, TB 3 seed"),
    ("0.0272", "tables/17b_so_sanh_theo_cap.txt", "bien do S0-B0 giua cac seed"),
    ("0.0384", "tables-offline/33_vs_hai_baseline.txt", "P2 vs S0 (baseline 2), ens3seed"),
    ("0.0074", "tables-offline/33_vs_hai_baseline.txt", "CI duoi cua P2 - S0"),
    ("0.0391", "tables-offline/33_vs_hai_baseline.txt", "P2 vs S0 tai seed 0 -- co y nghia"),
    ("0.7130", "tables/20_donbay_ensemble_kientruc.txt", "ensemble kien truc chon tren val (am)"),
    ("0.7358", "tables/20_donbay_ensemble_kientruc.txt", "to hop cao nhat tren test (khong bao cao)"),
    ("0.0228", "tables/20_donbay_ensemble_kientruc.txt", "khoang cach val-test cua to hop"),
    ("0.0019", "tables/24_duong_hoc_val.txt", "do lech 5 epoch cuoi cua P2"),
    # --- muc 6: transfer learning ---
    ("0.5674", "tables/27_transfer_learning.txt", "T1 linear probe"),
    ("0.6596", "tables/27_transfer_learning.txt", "T2 dong bang nua duoi"),
    ("0.6394", "tables/27_transfer_learning.txt", "T3 progressive + LR phan biet"),
    ("0.0146", "tables/27_transfer_learning.txt", "nguong 2 sigma"),
    ("0.1106", "tables/27_transfer_learning.txt", "khoang cach T1 -> T4"),
    # --- muc 7: trien khai (T4) ---
    ("15.4", "tables/28_trien_khai_onnx_do_tre.txt", "DenseNet ms @ batch 1"),
    ("3.18", "tables/28_trien_khai_onnx_do_tre.txt", "DenseNet ms @ batch 32"),
    ("19.5", "tables/28_trien_khai_onnx_do_tre.txt", "Swin-T ms @ batch 1"),
    ("4.82", "tables/28_trien_khai_onnx_do_tre.txt", "Swin-T ms @ batch 32"),
    ("11.7", "tables/28_trien_khai_onnx_do_tre.txt", "CoAtNet@224 ms @ batch 1"),
    ("5.13", "tables/28_trien_khai_onnx_do_tre.txt", "CoAtNet@224 ms @ batch 32 -- cau hinh trien khai"),
    ("13.0", "tables/28_trien_khai_onnx_do_tre.txt", "CoAtNet@288 ms @ batch 1"),
    ("8.74", "tables/28_trien_khai_onnx_do_tre.txt", "CoAtNet@288 ms @ batch 32"),
    ("114.8", "tables/28_trien_khai_onnx_do_tre.txt", "kich thuoc ONNX cua P2"),
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
                path = os.path.join(ROOT, *source.split("/"))
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
    print("tat ca khop: moi con so trong bao cao deu co trong nguon va nguoc lai")


if __name__ == "__main__":
    main()
