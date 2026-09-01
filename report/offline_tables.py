#!/usr/bin/env python
"""BAY bang (30-36) tinh LAI o 0 GPU, chay tu may local.

    python report/offline_tables.py        # -> report/tables-offline/

Nguon: bang 30-34 doc `ckpt-t4/*_seed*.npz`; bang 35-36 doc lai chinh `report/tables/*.txt`
(phien SESSION = 4 sinh them P2c / P2b seed 1-2 / A1 / A2 ma may local chua tai `.npz` ve).

Vi sao khong dung `extract.py`: khong o nao trong notebook in ra duoc cac bang nay.
Bang 30 can ca hai lan chay (A100 + T4) mot luc, ma mot phien notebook chi thay mot lan chay.
Bang 31 la con so cua `PROPOSED_TAG = P2` DUOI QUY TAC DA CHOT (`top3`) — o 19b cua notebook
dang luu con ghim cung `top3_tta` nen `tables/23_*` in 0,7486 thay vi 0,7441. Code da sua
(`build_notebook.py`, `test_notebook.py` nhom 8); notebook bat kip o phien Kaggle sau. Den luc
do bao cao van phai lay so tu day, khong tu `tables/23_*`.

Ghi vao `tables-offline/` chu khong phai `tables/`: `extract.py` xoa sach `tables/` moi lan chay.

`f1_macro` o day duoc viet lai bang numpy (may local khong co sklearn). Ham `_self_check()`
doi chieu no voi con so notebook da in ra va NEM LOI neu lech — khong co bang nao duoc ghi ra
tu mot phep do chua doi chieu.
"""
import io, json, os, re, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir))
NPZ_DIR = os.path.join(ROOT, "ckpt-t4")
OUT_DIR = os.path.join(HERE, "tables-offline")
A100_DIR = os.path.join(HERE, "tables-a100")

LF = chr(10)

RULES = ["best", "smooth", "top3", "best_tta", "smooth_tta", "top3_tta"]
RULE_KEY = {"best": "logits_best", "smooth": "logits_smooth", "top3": "probs_top3",
            "best_tta": "logits_best_tta", "smooth_tta": "logits_smooth_tta",
            "top3_tta": "probs_top3_tta"}
BASE_TAGS = ["B0_densenet121", "S0_swin_t", "P0_coatnet0", "P1_coatnet0_288"]
SEEDS = [0, 1, 2]
PAPER = 0.6504
NC = 22

# So anh TRAIN moi lop, lay tu `tables/06_eda.txt`. Thu tu index nhan = thu tu ALPHABET cua ten
# thu muc lop (ImageFolder), khong phai thu tu giam dan cua bang EDA -> phai sort lai theo ten.
TRAIN_COUNTS_BY_NAME = {
    "Accessory tools": 760, "Barrett's esophagus": 57, "Blood in lumen": 103, "Cecum": 68,
    "Colon diverticula": 17, "Colon polyps": 492, "Colorectal cancer": 83, "Duodenal bulb": 123,
    "Dyed-lifted-polyps": 85, "Dyed-resection-margins": 148, "Esophagitis": 64,
    "Gastric polyps": 39, "Gastroesophageal_junction_normal z-line": 198, "Ileocecal valve": 120,
    "Mucosal inflammation large bowel": 17, "Normal esophagus": 84,
    "Normal mucosa and vascular pattern in the large bowel": 880, "Normal stomach": 581,
    "Pylorus": 236, "Resected polyps": 55, "Retroflex rectum": 40,
    "Small bowel_terminal ileum": 508,
}
CLASSES = sorted(TRAIN_COUNTS_BY_NAME)          # thu tu ImageFolder = thu tu index nhan
TRAIN_COUNTS = np.array([TRAIN_COUNTS_BY_NAME[k] for k in CLASSES], float)


# --------------------------------------------------------------------------- do luong
def f1_macro(y_true, y_pred, n_class=NC):
    '''Tuong duong sklearn f1_score(labels=range(22), average="macro", zero_division=0).'''
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = np.empty(n_class)
    for c in range(n_class):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        out[c] = 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    return float(out.mean())


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def as_logits(a):
    '''`probs_top3` la xac suat; muon tru log-prior thi phai lay log truoc (xem muc 16 notebook).'''
    a = np.asarray(a, dtype=np.float64)
    if a.min() >= 0 and np.allclose(a.sum(axis=1), 1.0, atol=1e-3):
        return np.log(a + 1e-12)
    return a


def logit_adjust(logits, tau):
    prior = TRAIN_COUNTS / TRAIN_COUNTS.sum()
    return logits - tau * np.log(prior + 1e-12)[None, :]


def tune_tau(val_logits, val_y):
    '''Do tau tren VAL, khong bao gio tren test.'''
    sc = [(round(float(t), 1), f1_macro(val_y, logit_adjust(val_logits, t).argmax(1)))
          for t in np.arange(0.0, 1.01, 0.1)]
    return max(sc, key=lambda x: x[1])[0]


def bootstrap_ci(y_true, y_pred, n_boot=1000, seed=0):
    '''Chi trung binh tren cac lop CO MAT trong mau lay lai — giong `bootstrap_ci` cua notebook.'''
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n, st = len(y_true), np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        present = np.unique(y_true[idx])
        yt, yp = y_true[idx], y_pred[idx]
        f = [(lambda tp, fp, fn: 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))(
            int(((yt == c) & (yp == c)).sum()), int(((yt != c) & (yp == c)).sum()),
            int(((yt == c) & (yp != c)).sum())) for c in present]
        st[i] = float(np.mean(f))
    lo, hi = np.percentile(st, [2.5, 97.5])
    return f1_macro(y_true, y_pred), float(lo), float(hi)


def load(tag, seed):
    p = os.path.join(NPZ_DIR, "%s_seed%d.npz" % (tag, seed))
    if not os.path.exists(p):
        sys.exit("khong thay %s -- tai Output cua phien Kaggle ve ckpt-t4/ truoc" % p)
    return np.load(p, allow_pickle=True)


def scores(tag, seed):
    return json.loads(str(load(tag, seed)["scores_json"]))


# --------------------------------------------------------------------------- doi chieu
def _self_check():
    '''`f1_macro` viet lai phai khop tung chu so voi con so notebook da ghi vao `.npz`.'''
    bad = []
    for tag in BASE_TAGS + ["P2_coatnet0_288_modern"]:
        for s in SEEDS:
            d, sc = load(tag, s), scores(tag, s)
            for rule in RULES:
                mine = f1_macro(d["y_true"], as_logits(d[RULE_KEY[rule]]).argmax(1))
                if abs(mine - sc[rule]) > 5e-5:
                    bad.append("%s seed%d %s: %.6f != %.6f" % (tag, s, rule, mine, sc[rule]))
    if bad:
        sys.exit("f1_macro viet lai KHONG khop notebook:\n  " + "\n  ".join(bad))
    print("  doi chieu f1_macro voi %d run x 6 quy tac: khop" % (5 * 3))


def read_a100():
    '''Doc lai so A100 tu `tables-a100/12..15` — khong go tay con so nao.'''
    out = {}
    for tag in BASE_TAGS:
        fn = [f for f in os.listdir(A100_DIR) if f.endswith(tag + ".txt")]
        if not fn:
            sys.exit("khong thay bang A100 cho %s trong %s" % (tag, A100_DIR))
        txt = io.open(os.path.join(A100_DIR, fn[0]), encoding="utf-8").read()
        rows = re.findall(r"^\s+best=([\d.]+)\s+best_tta=([\d.]+)\s+smooth=([\d.]+)"
                          r"\s+smooth_tta=([\d.]+)\s+top3=([\d.]+)\s+top3_tta=([\d.]+)",
                          txt, re.M)
        if len(rows) != 3:
            sys.exit("bang A100 cua %s co %d dong seed, can 3" % (tag, len(rows)))
        order = ["best", "best_tta", "smooth", "smooth_tta", "top3", "top3_tta"]
        out[tag] = [dict(zip(order, map(float, r))) for r in rows]
    return out


# --------------------------------------------------------------------------- bang 30
def table_hw():
    a100, L = read_a100(), []
    w = L.append
    w("Lap lai CUNG code / CUNG split (SPLIT_SEED=42) / CUNG seed tren PHAN CUNG KHAC NHAU")
    w("  A100-SXM4-40GB: trong so huan luyen 2026-08-26, trich bang o lan chay thu 4 2026-08-27")
    w("                  (nguon: report/tables-a100/12..15)")
    w("  Tesla T4 / Kaggle, 2026-08-30..31  (nguon: ckpt-t4/*_seed*.npz)")
    w("30 epoch, batch 32, seed [0,1,2], 22 lop, test 1.586 anh -- moi thu giong nhau tru GPU.")
    w("")
    w("%-18s %-11s %8s %8s %10s %9s %11s %9s"
      % ("cau hinh", "quy tac", "A100 TB", "T4 TB", "chenh TB", "|d| max", "sigma A100", "sigma T4"))
    w("%-18s %-11s %8s %8s %10s %9s %11s %9s"
      % ("", "", "", "", "", "1 seed", "", ""))
    agg = {r: [] for r in RULES}
    for tag in BASE_TAGS:
        for rule in RULES:
            a = np.array([a100[tag][s][rule] for s in range(3)])
            t = np.array([scores(tag, s)[rule] for s in SEEDS])
            agg[rule].append((a.mean(), t.mean(), np.abs(a - t).max()))
            w("%-18s %-11s %8.4f %8.4f %+10.4f %9.4f %11.4f %9.4f"
              % (tag, rule, a.mean(), t.mean(), t.mean() - a.mean(),
                 np.abs(a - t).max(), a.std(), t.std()))
        w("")
    w("Trung binh tren 4 cau hinh -- day la dong dang doc:")
    w("")
    w("%-13s %14s %14s" % ("quy tac", "|chenh TB| TB", "|d| max 1 seed"))
    for rule in sorted(RULES, key=lambda r: np.abs(np.array(agg[r])[:, 1]
                                                   - np.array(agg[r])[:, 0]).mean()):
        v = np.array(agg[rule])
        w("%-13s %14.4f %14.4f" % (rule, np.abs(v[:, 1] - v[:, 0]).mean(), v[:, 2].max()))
    w("")
    w("Doc the nao:")
    w(" - `top3` ben voi phan cung gap ~4x cac quy tac mot-checkpoint. Day la ly le THU BA cho")
    w("   `top3`, doc lap voi hai ly le cu ('mien phi' va 'giam phuong sai') o RESULTS.md muc 9.")
    w(" - Bien do 0,015-0,019 cua cac quy tac mot-checkpoint LON HON don bay kien truc ma vong 1")
    w("   do duoc (+0,0142..+0,0175) -> xep hang kien truc vong 1 khong song qua mot lan doi may.")
    w(" - Doi chieu Gate 0a: no do lech ~0,010 tren duong val 3 epoch. Bang nay cho thay o 30")
    w("   epoch + chon checkpoint thi lech tich luy den 0,04-0,07 o mot seed.")
    w(" - P2 - P1 = +0,0443 LON HON moi hieu ung phan cung o bang nay, va P2/P2b/B0 deu cung T4")
    w("   -> ket luan cua muc 15c KHONG vat qua hai loai may.")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- bang 31
def _logit_block(w, TAG, rule):
    '''Mot khoi "hieu chinh logit" cho mot tag duoi MOT quy tac. Tra ve (mean_raw, mean_adj, use).

    Tach ra thanh ham vi muc 3.7 cua bao cao doi chieu P1 voi P2, va hai nua cua phep doi chieu do
    BUOC phai o cung quy tac -- ban dau bao cao tron cot truoc cua top3 voi cot sau cua top3_tta,
    va dau hieu la mot dong co tau*=0,0 ma diem van doi (khong the dung).'''
    fr, fa, taus, praw, padj, yt = [], [], [], [], [], None
    for s in SEEDS:
        d = load(TAG, s)
        lg = as_logits(d[RULE_KEY[rule]])
        tau = tune_tau(d["val_logits"], d["val_y"])
        adj = logit_adjust(lg, tau)
        f0, f1_ = f1_macro(d["y_true"], lg.argmax(1)), f1_macro(d["y_true"], adj.argmax(1))
        fr.append(f0); fa.append(f1_); taus.append(tau)
        praw.append(softmax(lg)); padj.append(softmax(adj)); yt = d["y_true"]
        w("  seed %d: %-9s = %.4f   tau*=%.1f -> %.4f  (%+.4f)"
          % (s, rule, f0, tau, f1_, f1_ - f0))
    mr, sr, ma, sa = np.mean(fr), np.std(fr), np.mean(fa), np.std(fa)
    w("")
    w("  %18s %11s %9s" % ("bien the", "trung binh", "do lech"))
    w("  %18s %11.4f %9.4f" % ("khong hieu chinh", mr, sr))
    w("  %18s %11.4f %9.4f" % ("co hieu chinh", ma, sa))
    use = (ma > mr) and (sa <= sr * 1.5)
    w("  -> tieu chi o 19b (tang TB VA do lech khong phinh > 1,5x): %s"
      % ("GIU hieu chinh logit" if use else "BO hieu chinh logit"))
    return mr, sr, ma, sa, use, (padj if use else praw), yt


def table_p2_system():
    TAG, L = "P2_coatnet0_288_modern", []
    w = L.append
    w("HE THONG DE XUAT voi PROPOSED_TAG = %s" % TAG)
    w("(bang nay thay cho `tables/23_*` cho den phien Kaggle sau -- xem docstring cua script)")
    w("")
    w("=" * 78)
    w("DOI CHIEU: cung lever, cung quy tac 'top3', tren P1 va tren P2")
    w("Day la hai nua cua muc 3.7 trong bao cao -- PHAI o cung quy tac moi so duoc.")
    w("")
    for t in ("P1_coatnet0_288", TAG):
        w("--- %s ---" % t)
        _logit_block(w, t, "top3")
        w("")
    w("=" * 78)
    w("")
    for rule in ("top3", "top3_tta"):
        w("--- quy tac '%s' ---" % rule)
        fr, fa, taus, praw, padj, yt = [], [], [], [], [], None
        for s in SEEDS:
            d = load(TAG, s)
            lg = as_logits(d[RULE_KEY[rule]])
            tau = tune_tau(d["val_logits"], d["val_y"])
            adj = logit_adjust(lg, tau)
            f0, f1_ = f1_macro(d["y_true"], lg.argmax(1)), f1_macro(d["y_true"], adj.argmax(1))
            fr.append(f0); fa.append(f1_); taus.append(tau)
            praw.append(softmax(lg)); padj.append(softmax(adj)); yt = d["y_true"]
            w("  seed %d: %-9s = %.4f   tau*=%.1f -> %.4f  (%+.4f)"
              % (s, rule, f0, tau, f1_, f1_ - f0))
        mr, sr, ma, sa = np.mean(fr), np.std(fr), np.mean(fa), np.std(fa)
        w("")
        w("  %18s %11s %9s" % ("bien the", "trung binh", "do lech"))
        w("  %18s %11.4f %9.4f" % ("khong hieu chinh", mr, sr))
        w("  %18s %11.4f %9.4f" % ("co hieu chinh", ma, sa))
        # Dung DUNG tieu chi da dat truoc trong o 19b cua notebook, khong noi long ra.
        use = (ma > mr) and (sa <= sr * 1.5)
        w("  -> tieu chi o 19b (tang TB VA do lech khong phinh > 1,5x): %s"
          % ("GIU hieu chinh logit" if use else "BO hieu chinh logit"))
        m, sd, pp = (ma, sa, padj) if use else (mr, sr, praw)
        pt, lo, hi = bootstrap_ci(yt, pp[0].argmax(1))
        w("")
        w("  HE THONG DE XUAT (%s)" % ("co hieu chinh logit" if use else "khong hieu chinh logit"))
        w("                   macro-F1 = %.4f +/- %.4f  tren %d seed" % (m, sd, len(fr)))
        w("                   so voi paper %.4f: %+.4f" % (PAPER, m - PAPER))
        w("                   CI 95%% (seed dau) = [%.4f, %.4f]  -> %s"
          % (lo, hi, "VUOT (CI khong chong lan)" if lo > PAPER else "chua tach bach"))
        e = sum(softmax(as_logits(p)) for p in pp) / len(pp)
        p3, l3, h3 = bootstrap_ci(yt, e.argmax(1))
        w("  + ensemble ca 3 seed  macro-F1 = %.4f  CI 95%% [%.4f, %.4f]   (DONG RIENG: 3 lan train)"
          % (p3, l3, h3))
        w("     -> %s" % ("CI khong chong lan 0,6504: duoc quyen noi 'vuot baseline cong bo'."
                          if l3 > PAPER else "CI van chong lan 0,6504."))
        w("")
    w("Doi chieu voi P1 (RESULTS.md muc 10.8, phat hien 3): tren P1 thi tau* = 0,2/0,0/0,1 va")
    w("do lech PHINH 0,0121 -> 0,0134 nen hieu chinh logit bi bo. Tren P2 thi nguoc lai. Ket luan")
    w("'khong co gi de an o day' chi dung tren P1, va no khong con ha ky vong cua P5 nua.")
    return "\n".join(L) + "\n"



# --------------------------------------------------------------------------- bang 32
# F1 per-class cua DenseNet-121 trong Table 3 cua bai bao (arXiv 2307.08140, trang 12), lam tron
# 2 chu so nhu ban goc. 22 gia tri nay trung binh ra 0,6518 -- khop voi 0,6504 duoc cong bo, nen
# bang da duoc doc dung. Khoa = ten lop cua ImageFolder.
PAPER_F1 = {
    "Accessory tools": 0.95, "Barrett's esophagus": 0.40, "Blood in lumen": 0.89, "Cecum": 0.23,
    "Colon diverticula": 0.50, "Colon polyps": 0.82, "Colorectal cancer": 0.50,
    "Duodenal bulb": 0.74, "Dyed-lifted-polyps": 0.86, "Dyed-resection-margins": 0.93,
    "Esophagitis": 0.31, "Gastric polyps": 0.33,
    "Gastroesophageal_junction_normal z-line": 0.74, "Ileocecal valve": 0.72,
    "Mucosal inflammation large bowel": 0.50, "Normal esophagus": 0.77,
    "Normal mucosa and vascular pattern in the large bowel": 0.84, "Normal stomach": 0.88,
    "Pylorus": 0.86, "Resected polyps": 0.17, "Retroflex rectum": 0.55,
    "Small bowel_terminal ileum": 0.85,
}
TEST_COUNTS_BY_NAME = {
    "Accessory tools": 253, "Barrett's esophagus": 19, "Blood in lumen": 34, "Cecum": 23,
    "Colon diverticula": 6, "Colon polyps": 164, "Colorectal cancer": 28, "Duodenal bulb": 41,
    "Dyed-lifted-polyps": 28, "Dyed-resection-margins": 49, "Esophagitis": 21,
    "Gastric polyps": 13, "Gastroesophageal_junction_normal z-line": 66, "Ileocecal valve": 40,
    "Mucosal inflammation large bowel": 6, "Normal esophagus": 28,
    "Normal mucosa and vascular pattern in the large bowel": 294, "Normal stomach": 194,
    "Pylorus": 79, "Resected polyps": 18, "Retroflex rectum": 13,
    "Small bowel_terminal ileum": 169,
}


def per_class_f1(y_true, y_pred, n_class=NC):
    out = np.empty(n_class)
    for c in range(n_class):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        out[c] = 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    return out


def table_per_class():
    TAG, RULE, SEED = "P2_coatnet0_288_modern", "top3", 0
    d = load(TAG, SEED)
    y = np.asarray(d["y_true"])
    f1 = per_class_f1(y, as_logits(d[RULE_KEY[RULE]]).argmax(1))
    names = CLASSES                     # thu tu alphabet = thu tu index nhan

    # Doi chieu: 22 gia tri cua bai bao phai trung binh ra ~0,6518, khong thi da doc sai bang.
    pm = float(np.mean([PAPER_F1[n] for n in names]))
    if abs(pm - 0.6518) > 0.001:
        sys.exit("PAPER_F1 trung binh %.4f, ky vong 0,6518 -- da doc sai Table 3" % pm)
    # Va so anh test phai khop bang split da trich.
    if sum(TEST_COUNTS_BY_NAME.values()) != 1586:
        sys.exit("TEST_COUNTS tong %d, can 1586" % sum(TEST_COUNTS_BY_NAME.values()))

    L = []
    w = L.append
    w("F1 tung lop: %s seed %d, quy tac '%s'  vs  Table 3 cua bai bao (DenseNet-121)" % (TAG, SEED, RULE))
    w("Bai bao lam tron 2 chu so; 22 gia tri do trung binh ra %.4f (cong bo 0,6504)." % pm)
    w("Tap test cung thanh phan tung lop (muc 1.3 cua bao cao), nen quy duoc phan cai thien VE TUNG LOP.")
    w("")
    w("%-54s %8s %8s %9s %6s %6s" % ("lop", "paper", "P2", "delta", "test", "train"))
    rows = sorted(range(NC), key=lambda i: -(f1[i] - PAPER_F1[names[i]]))
    for i in rows:
        n = names[i]
        w("%-54s %8.2f %8.3f %+9.3f %6d %6d"
          % (n[:54], PAPER_F1[n], f1[i], f1[i] - PAPER_F1[n],
             TEST_COUNTS_BY_NAME[n], TRAIN_COUNTS_BY_NAME[n]))
    w("")
    w("--- Chia theo co tap test (nguong 50 anh, giong muc 4.4 ban A100) ---")
    w("")
    w("%-28s %8s %12s %26s" % ("nhom", "so lop", "delta F1 TB", "gop vao macro-F1"))
    tot = 0.0
    for label, keep in (("hiem (< 50 anh test)", lambda n: TEST_COUNTS_BY_NAME[n] < 50),
                        ("pho bien (>= 50 anh test)", lambda n: TEST_COUNTS_BY_NAME[n] >= 50)):
        idx = [i for i in range(NC) if keep(names[i])]
        ds = np.array([f1[i] - PAPER_F1[names[i]] for i in idx])
        contrib = ds.sum() / NC
        tot += contrib
        w("%-28s %8d %12.3f %+26.4f" % (label, len(idx), ds.mean(), contrib))
    ds_all = np.array([f1[i] - PAPER_F1[names[i]] for i in range(NC)])
    w("%-28s %8d %12.3f %+26.4f" % ("toan bo 22 lop", NC, ds_all.mean(), ds_all.sum() / NC))
    idx_rare = [i for i in range(NC) if TEST_COUNTS_BY_NAME[names[i]] < 50]
    c_rare = sum(f1[i] - PAPER_F1[names[i]] for i in idx_rare) / NC
    w("")
    w("-> lop hiem gop %.1f%% cua toan bo muc cai thien." % (100 * c_rare / (ds_all.sum() / NC)))
    w("")
    w("--- Cac lop van THUA bai bao (phai bao cao, nguyen tac 5 cua de bai) ---")
    w("")
    worse = [i for i in range(NC) if f1[i] < PAPER_F1[names[i]]]
    for i in sorted(worse, key=lambda i: f1[i] - PAPER_F1[names[i]]):
        n = names[i]
        drag = (f1[i] - PAPER_F1[n]) / NC
        w("%-54s %+.3f  -> keo macro-F1 %+.4f  (%d anh test)"
          % (n[:54], f1[i] - PAPER_F1[n], drag, TEST_COUNTS_BY_NAME[n]))
    w("")
    w("Vi macro-F1 can bang 22 lop nhu nhau, MOT lop 6 anh co the ngoam %.4f macro-F1." % (1.0 / NC))
    w("Do la co che khien bootstrap CI (+/-0,035) rong gap nhieu lan sigma giua cac seed.")
    return LF.join(L) + LF



# --------------------------------------------------------------------------- bang 33
def _mcnemar(y, pa, pb):
    from math import comb
    b = int(((pa == y) & (pb != y)).sum())
    c = int(((pa != y) & (pb == y)).sum())
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    return b, c, min(1.0, sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2)


def _paired(y, pa, pb, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    d0 = f1_macro(y, pa) - f1_macro(y, pb)
    ds = np.empty(n_boot)
    for i in range(n_boot):
        ix = rng.integers(0, n, n)
        ds[i] = f1_macro(y[ix], pa[ix]) - f1_macro(y[ix], pb[ix])
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return d0, float(lo), float(hi)


def table_vs_baselines():
    '''Bang 17b cua notebook so MOI THU voi B0, nen no khong tra loi cau hoi cua rubric:
    "mo hinh de xuat co vuot CA HAI baseline khong?". Baseline 2 (S0 Swin-T) chua tung duoc
    kiem theo cap. Bang nay lap cho du -- 0 GPU, doc tu logits da luu.'''
    RULE = "top3"
    PROP = "P2_coatnet0_288_modern"
    BASES = [("B0_densenet121", "baseline 1 -- CNN tham chieu, = mo hinh cua bai bao"),
             ("S0_swin_t", "baseline 2 -- Transformer, nhanh bai bao KHONG co")]

    def preds(tag, seed):
        d = load(tag, seed)
        return np.asarray(d["y_true"]), as_logits(d[RULE_KEY[RULE]]).argmax(1)

    def preds_ens(tag):
        ds = [load(tag, s) for s in SEEDS]
        p = sum(softmax(as_logits(d[RULE_KEY[RULE]])) for d in ds) / len(ds)
        return np.asarray(ds[0]["y_true"]), p.argmax(1)

    L = []
    w = L.append
    w("HE THONG DE XUAT vs CA HAI BASELINE -- ghep cap tren CUNG bo anh, quy tac '%s'" % RULE)
    w("(bang 17b cua notebook chi so voi B0; baseline 2 chua tung duoc kiem theo cap)")
    w("")
    w("%-26s %-10s %9s %21s %10s %8s  %s"
      % ("A vs B", "nguon", "d", "CI95 cua d", "McNemar", "p", "ket luan"))
    for base, note in BASES:
        n_sig = 0
        for sd in SEEDS:
            y, pa = preds(PROP, sd)
            _, pb = preds(base, sd)
            d, lo, hi = _paired(y, pa, pb)
            b, c, pv = _mcnemar(y, pa, pb)
            sig = lo > 0 or hi < 0
            n_sig += bool(sig and d > 0)
            w("%-26s %-10s %+9.4f [%+.4f, %+.4f] %5d/%-4d %8.4f  %s"
              % ("P2 vs " + base.split("_")[0], "seed %d" % sd, d, lo, hi, b, c, pv,
                 "A > B, CO Y NGHIA" if sig and d > 0 else "chua ket luan duoc"))
        y, pa = preds_ens(PROP)
        _, pb = preds_ens(base)
        d, lo, hi = _paired(y, pa, pb)
        b, c, pv = _mcnemar(y, pa, pb)
        sig = lo > 0 or hi < 0
        w("%-26s %-10s %+9.4f [%+.4f, %+.4f] %5d/%-4d %8.4f  %s"
          % ("", "ens3seed", d, lo, hi, b, c, pv,
             "A > B, CO Y NGHIA" if sig and d > 0 else "chua ket luan duoc"))
        w("    ^^^ %s   -> co y nghia o %d/%d seed + dong ens3seed"
          % (note, n_sig, len(SEEDS)))
        w("")
    w("Doc the nao:")
    w(" - CI cua d KHONG chua 0  -> khac biet co y nghia tren BO TEST NAY.")
    w(" - Vuot baseline 2 con SACH HON vuot baseline 1: 3/3 seed thay vi 2/3. Ly do la B0 co mot")
    w("   seed may (seed 0 duoi quy tac 'best' cho 0,7008, cao bat thuong) -- xem muc 1.4.")
    w(" - Van chi la MOT bo test co dinh (SPLIT_SEED = 42). Phep nay lam so sanh nhay hon, KHONG")
    w("   tra loi 'ket qua co giu tren bo chia khac khong'.")
    return LF.join(L) + LF



# --------------------------------------------------------------------------- bang 34
def table_headroom():
    '''Du dia con lai nam o dau, dinh luong. Muc 4.4 tra loi "phan CAI THIEN da qua den tu dau";
    bang nay tra loi cau khac va la cau quyet dinh co nen tieu them gio GPU: "phan CHUA DAT nam o
    dau". Phep tinh la mot thi nghiem tuong tuong -- dat F1 cua mot nhom lop = 1,0 roi xem macro-F1
    len bao nhieu. No khong phai du bao, chi la phan ra so hoc cua khoang cach den tran 1,0.'''
    TAG, RULE, SEED = "P2_coatnet0_288_modern", "top3", 0
    d = load(TAG, SEED)
    y = np.asarray(d["y_true"])
    f1 = per_class_f1(y, as_logits(d[RULE_KEY[RULE]]).argmax(1))
    names = CLASSES
    cur = float(f1.mean())

    def what_if(idx):
        g = f1.copy()
        g[idx] = 1.0
        return float(g.mean())

    rare = [i for i, n in enumerate(names) if TEST_COUNTS_BY_NAME[n] < 50]
    comm = [i for i, n in enumerate(names) if TEST_COUNTS_BY_NAME[n] >= 50]
    tiny = [i for i, n in enumerate(names) if TEST_COUNTS_BY_NAME[n] <= 6]
    worst5 = list(np.argsort(f1)[:5])

    L = []
    w = L.append
    w("DU DIA CON LAI NAM O DAU -- %s seed %d, quy tac '%s'" % (TAG, SEED, RULE))
    w("Phan ra so hoc cua khoang cach den tran 1,0. KHONG phai du bao.")
    w("")
    w("macro-F1 hien tai = %.4f   -> con %.4f den tran ly thuyet 1,0" % (cur, 1 - cur))
    w("")
    w("%-52s %9s %9s %9s" % ("Neu HOAN HAO o...", "macro-F1", "gop", "% du dia"))
    for lbl, idx in (("15 lop HIEM (< 50 anh test)", rare),
                     ("7 lop PHO BIEN (>= 50 anh test)", comm)):
        v = what_if(idx)
        w("%-52s %9.4f %+9.4f %8.1f%%" % (lbl, v, v - cur, (v - cur) / (1 - cur) * 100))
    w("")
    for lbl, idx in (("chi 2 lop 6 anh test", tiny), ("chi 5 lop yeu nhat", worst5)):
        v = what_if(idx)
        w("%-52s %9.4f %+9.4f" % (lbl, v, v - cur))
    w("")
    w("So anh TRAIN va F1 trung binh cua moi nhom:")
    w("  15 lop hiem    : %6.1f anh/lop   F1 TB %.3f" %
      (float(np.mean([TRAIN_COUNTS_BY_NAME[names[i]] for i in rare])), float(f1[rare].mean())))
    w("  7 lop pho bien : %6.1f anh/lop   F1 TB %.3f" %
      (float(np.mean([TRAIN_COUNTS_BY_NAME[names[i]] for i in comm])), float(f1[comm].mean())))
    w("")
    w("Doc the nao -- va day la bang quyet dinh CO NEN TIEU THEM GIO GPU:")
    w(" - 85% du dia nam o 15 lop hiem, trung binh 73,5 anh train moi lop. Rieng HAI lop 6 anh giu")
    w("   +0,0552, tuc NHIEU HON ca don bay cong thuc (+0,0443) -- don bay lon nhat da do duoc.")
    w(" - Nghia la moi lever nham vao MO HINH deu dang tranh nhau 15% du dia con lai o cac lop da")
    w("   bao hoa, con 85% thi nam o cho ma chi THEM DU LIEU (hoac few-shot) moi cham duoc.")
    w(" - Cong voi nguong phan giai +/-0,035 cua bo test nay: mot lever nham vao mo hinh phai mua")
    w("   duoc > 0,035 moi chung minh duoc, ma tran cua toan bo nhom lop pho bien chi la +0,0420.")
    w("   Vay KHONG con lever nao thuoc ho 'doi mo hinh' co the chung minh duoc tren bo test nay.")
    return LF.join(L) + LF


# --------------------------------------------------------------- doc lai output notebook
SCORE_ORDER = ["best", "best_tta", "smooth", "smooth_tta", "top3", "top3_tta"]
TAB_DIR = os.path.join(HERE, "tables")


def read_seed_scores():
    """Diem 6 quy tac cua TUNG seed, doc tu `report/tables/*.txt` (extract.py sinh tu notebook).

    Vi sao khong doc `ckpt-t4/*.npz` nhu cac bang 30-34: phien SESSION = 4 sinh them
    `P2c`, `P2b` seed 1-2, `A1`, `A2` ma may local chua tai `.npz` ve. Output van la nguon
    MAY DOC duoc, khong phai con so go tay -- va `_check_seed_scores()` doi chieu trung binh
    cua no voi bang pivot ma pandas da in o muc 16.
    """
    txt = LF.join(io.open(os.path.join(TAB_DIR, f), encoding="utf-8").read()
                  for f in sorted(os.listdir(TAB_DIR)) if f.endswith(".txt"))
    head = re.compile(r"^\[(\S+) seed (\d)\]")
    row = re.compile(r"^\s+best=([\d.]+)\s+best_tta=([\d.]+)\s+smooth=([\d.]+)"
                     r"\s+smooth_tta=([\d.]+)\s+top3=([\d.]+)\s+top3_tta=([\d.]+)\s*$")
    out, cur = {}, None
    for line in txt.split(LF):
        m = head.match(line)
        if m:
            cur = (m.group(1), int(m.group(2)))
            continue
        m = row.match(line)
        if m and cur is not None:
            out.setdefault(cur, dict(zip(SCORE_ORDER, [float(x) for x in m.groups()])))
            cur = None
    if not out:
        sys.exit("khong doc duoc dong diem nao trong %s -- chay report/extract.py truoc" % TAB_DIR)
    return out


def read_pivot():
    """Bang pivot 'macro-F1 theo tung quy tac' o muc 16 + quy tac da chot."""
    p = os.path.join(TAB_DIR, "16_bang_6_quy_tac.txt")
    txt = io.open(p, encoding="utf-8").read()
    m = re.search(r"CHOT SELECTION_RULE = '(\w+)'", txt)
    if not m:
        sys.exit("%s khong co dong CHOT SELECTION_RULE -- notebook chua chay het muc 16" % p)
    cols = ["best", "smooth", "top3", "best_tta", "smooth_tta", "top3_tta"]   # thu tu cot cua pandas
    piv = {}
    for tag, *vals in re.findall(
            r"^(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$",
            txt, re.M):
        piv[tag] = dict(zip(cols, [float(v) for v in vals]))
    if len(piv) < 4:
        sys.exit("chi doc duoc %d dong tu bang pivot muc 16" % len(piv))
    return m.group(1), piv


def _check_seed_scores(sc, piv):
    """Trung binh tren seed cua so doc tung dong phai khop bang pivot pandas da in (lam tron 4 so)."""
    bad = []
    for tag, ref in piv.items():
        rows = [v for (t, _), v in sc.items() if t == tag]
        if not rows:
            bad.append("%s: khong tim thay dong seed nao" % tag)
            continue
        for rule, want in ref.items():
            got = sum(r[rule] for r in rows) / len(rows)
            if abs(got - want) > 1e-4:
                bad.append("%s %s: %.4f != %.4f (pivot)" % (tag, rule, got, want))
    if bad:
        sys.exit("so doc tu dong seed KHONG khop bang pivot muc 16:" + LF + "  "
                 + (LF + "  ").join(bad))
    print("  doi chieu %d dong seed voi bang pivot muc 16: khop" % len(sc))


def read_logit_table():
    """Muc 19: tau* va test truoc/sau hieu chinh logit, tren seed dau tien."""
    p = os.path.join(TAB_DIR, "19_donbay_hieuchinh_logit.txt")
    txt = io.open(p, encoding="utf-8").read()
    out = {}
    for tag, tau, before, after in re.findall(
            r"^(\S+)\s+tau\*=([\d.]+)\s+test:\s+([\d.]+)\s+->\s+([\d.]+)", txt, re.M):
        out[tag] = (float(tau), float(before), float(after))
    if not out:
        sys.exit("khong doc duoc dong nao tu %s" % p)
    return out


# --------------------------------------------------------------------------- bang 35
CELLS_2x2 = [("CoAtNet-0 @288",    "P1_coatnet0_288", "P2_coatnet0_288_modern"),
             ("CoAtNet-0 @224",    "P0_coatnet0",     "P2c_coatnet0_224_modern"),
             ("DenseNet-121 @224", "B0_densenet121",  "P2b_densenet121_modern")]


def table_interaction():
    sc = read_seed_scores()
    rule, piv = read_pivot()
    _check_seed_scores(sc, piv)

    def mean(tag):
        rows = [v[rule] for (t, _), v in sc.items() if t == tag]
        return (sum(rows) / len(rows), len(rows)) if rows else (float("nan"), 0)

    def s0(tag):
        return sc[(tag, 0)][rule]

    L = []
    w = L.append
    w("Tach don bay CONG THUC HUAN LUYEN ra khoi KIEN TRUC va DO PHAN GIAI")
    w("  nguon : report/tables/12,14,15b,16 (phien Kaggle T4 2026-08-31, SESSION = 4)")
    w("  quy tac doc: '%s' -- muc 16 chot (xem CANH BAO cuoi bang)" % rule)
    w("")
    w("A. TRUNG BINH TREN CAC SEED")
    w("")
    w("  %-20s  %-16s %-16s %9s   %s"
      % ("cau hinh", "cong thuc cu", "cong thuc moi", "don bay", "seed"))
    lev = {}
    for label, old, new in CELLS_2x2:
        mo, no_ = mean(old)
        mn, nn = mean(new)
        lev[label] = mn - mo
        w("  %-20s  %-4s %-11.4f %-4s %-11.4f %+9.4f   %d vs %d"
          % (label, old.split("_")[0], mo, new.split("_")[0], mn, mn - mo, no_, nn))
    d288, d224h, d224c = (lev["CoAtNet-0 @288"], lev["CoAtNet-0 @224"],
                          lev["DenseNet-121 @224"])
    w("")
    w("  [1] cong thuc x KIEN TRUC     (co dinh 224)   : %+.4f - (%+.4f) = %+.4f"
      % (d224h, d224c, d224h - d224c))
    w("  [2] cong thuc x DO PHAN GIAI  (co dinh hybrid): %+.4f - (%+.4f) = %+.4f"
      % (d288, d224h, d288 - d224h))
    w("")
    w("B. CUNG SEED 0 -- phep so chat hon, vi P2c moi chay 1 seed")
    w("")
    b0, p0 = s0("B0_densenet121"), s0("P0_coatnet0")
    p2b, p2c = s0("P2b_densenet121_modern"), s0("P2c_coatnet0_224_modern")
    w("  %-22s %12s %12s %10s" % ("kien truc @224", "cong thuc cu", "cong thuc moi", "don bay"))
    w("  %-22s %12.4f %12.4f %+10.4f" % ("DenseNet-121 (CNN)", b0, p2b, p2b - b0))
    w("  %-22s %12.4f %12.4f %+10.4f" % ("CoAtNet-0 (hybrid)", p0, p2c, p2c - p0))
    inter = (p2c - p0) - (p2b - b0)
    w("")
    w("  so hang TUONG TAC = %+.4f - (%+.4f) = %+.4f" % (p2c - p0, p2b - b0, inter))
    w("")
    w("  Tung yeu to MOT MINH tai seed 0:")
    w("    chi doi kien truc  (giu cong thuc cu) : %.4f - %.4f = %+.4f" % (p0, b0, p0 - b0))
    w("    chi doi cong thuc  (giu CNN)          : %.4f - %.4f = %+.4f" % (p2b, b0, p2b - b0))
    w("    doi CA HAI                            : %.4f - %.4f = %+.4f" % (p2c, b0, p2c - b0))
    w("  -> ca hai don bay don le deu <= 0; chi to hop moi duong. Day la mot TUONG TAC,")
    w("     khong phai tong cua hai don bay doc lap.")
    w("")
    w("C. DO PHAN GIAI 288 vs 224, doc rieng duoi tung cong thuc (seed 0)")
    w("")
    p1, p2 = s0("P1_coatnet0_288"), s0("P2_coatnet0_288_modern")
    w("    duoi cong thuc CU  : P1  %.4f - P0  %.4f = %+.4f" % (p1, p0, p1 - p0))
    w("    duoi cong thuc MOI : P2  %.4f - P2c %.4f = %+.4f" % (p2, p2c, p2 - p2c))
    w("  -> 288 khong con mua duoc gi khi da co cong thuc moi.")
    w("")
    w("  Nguong phan giai cua bo test 1.586 anh: CI bootstrap ~ +/-0,035.")
    w("  Chi so hang TUONG TAC (%+.4f muc A / %+.4f muc B) vuot nguong;" % (d224h - d224c, inter))
    w("  thanh phan DO PHAN GIAI (%+.4f) va don bay 288 duoi cong thuc moi (%+.4f) thi khong."
      % (d288 - d224h, p2 - p2c))
    w("")
    w("D. DOC LAI CA BANG DUOI 'top3_tta' -- kiem ket luan co phu thuoc quy tac cham diem khong")
    w("")
    alt = "top3_tta"

    def mean_alt(tag):
        rows = [v[alt] for (t, _), v in sc.items() if t == tag]
        return sum(rows) / len(rows) if rows else float("nan")

    lev_alt = {}
    for label, old, new in CELLS_2x2:
        lev_alt[label] = mean_alt(new) - mean_alt(old)
        w("    %-20s %+.4f" % (label, lev_alt[label]))
    ia = lev_alt["CoAtNet-0 @224"] - lev_alt["DenseNet-121 @224"]
    ra = lev_alt["CoAtNet-0 @288"] - lev_alt["CoAtNet-0 @224"]
    w("")
    w("    cong thuc x KIEN TRUC     = %+.4f   (duoi '%s': %+.4f)" % (ia, rule, d224h - d224c))
    w("    cong thuc x DO PHAN GIAI  = %+.4f   (duoi '%s': %+.4f)" % (ra, rule, d288 - d224h))
    w("  -> Doi quy tac cham diem KHONG doi ket luan: thanh phan kien truc van vuot +/-0,035 con")
    w("     thanh phan do phan giai van khong. Muc 2.5 cho thay quy tac cham diem tung dao ca mot")
    w("     xep hang, nen phep kiem nay khong thua.")
    w("")
    w("CANH BAO -- vi sao bang nay o day chu khong lay tu o 15c cua notebook:")
    w("  Output luu trong .ipynb cua phien SESSION = 4 in bang 15c duoi quy tac 'best' chu khong")
    w("  phai '%s'. O 15c doc SELECTION_RULE, ma bien do chi duoc CHOT o muc 16 -- chay SAU no." % rule)
    w("  Duoi 'best' thanh phan do phan giai doc ra la +0,0526 (nguoc han voi %+.4f o tren) nen"
      % (d288 - d224h))
    w("  dong ket luan in kem trong o do la SAI. Notebook da duoc sua (ca 15c..15f nay goi chung")
    w("  ham vote_rule() voi muc 16), nhung output cu thi khong sua lai duoc ma khong chay lai GPU.")
    return LF.join(L) + LF


# --------------------------------------------------------------------------- bang 36
def table_imbalance_and_pretrain():
    sc = read_seed_scores()
    rule, piv = read_pivot()
    adj = read_logit_table()

    def s0(tag):
        return sc[(tag, 0)][rule]

    L = []
    w = L.append
    w("Hai don bay do o muc 19 va muc 25, xep canh nhau vi chung tra loi cung mot cau hoi")
    w("  nguon : report/tables/19, 25 | quy tac '%s' | tat ca deu tren SEED 0" % rule)
    w("")
    w("A. MAT CAN BANG LOP: sua trong HAM MAT MAT hay sua LUC SUY LUAN?")
    w("   cung DenseNet-121, cung split, cung seed, cung 30 epoch")
    w("")
    b0 = s0("B0_densenet121")
    a2 = s0("A2_balanced_softmax")
    tau_b0, before_b0, after_b0 = adj["B0_densenet121"]
    w("   %-42s %8.4f" % ("B0  cross-entropy (doi chung)", b0))
    w("   %-42s %8.4f  %+.4f" % ("A2  balanced softmax  -- sua HAM MAT MAT", a2, a2 - b0))
    w("   %-42s %8.4f  %+.4f   tau* = %.1f"
      % ("B0 + hieu chinh logit -- sua LUC SUY LUAN", after_b0, after_b0 - before_b0, tau_b0))
    w("")
    tau_p2, before_p2, after_p2 = adj["P2_coatnet0_288_modern"]
    w("   Tren cau hinh tot nhat (P2), cung don bay suy luan:")
    w("   %-42s %8.4f  %+.4f   tau* = %.1f"
      % ("P2 -> P2 + hieu chinh logit", after_p2, after_p2 - before_p2, tau_p2))
    w("")
    w("   -> Cung mot muc tieu, hai cho sua. Sua ham mat mat: %+.4f (am). Sua luc suy luan:"
      % (a2 - b0))
    w("      %+.4f tren B0 va %+.4f tren P2, ton 0 epoch. Doi voi bo du lieu nay, mat can bang"
      % (after_b0 - before_b0, after_p2 - before_p2))
    w("      nen duoc chua SAU khi huan luyen chu khong phai trong luc huan luyen.")
    w("")
    w("B. DU LIEU PRETRAIN: cung kien truc Swin-T, chi doi bo trong so khoi tao")
    w("")
    s0_ = s0("S0_swin_t")
    a1 = s0("A1_swin_in22k")
    w("   %-42s %8.4f" % ("S0  Swin-T, pretrain ImageNet-1k", s0_))
    w("   %-42s %8.4f  %+.4f" % ("A1  Swin-T, pretrain ImageNet-22k", a1, a1 - s0_))
    w("")
    p0, b0m = s0("P0_coatnet0"), b0
    w("   De so sanh, don bay KIEN TRUC do tren cung seed 0: %+.4f (P0 - B0)." % (p0 - b0m))
    w("   Chi doi DU LIEU pretrain mua duoc %+.4f -- lon hon, va khong doi mot dong kien truc nao."
      % (a1 - s0_))
    w("   Day la so do ung ho huong 'them du lieu' o muc 4.6, thay cho mot lap luan thuan ly thuyet.")
    return LF.join(L) + LF


def main():
    if not os.path.isdir(NPZ_DIR):
        sys.exit("khong thay %s" % NPZ_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    print("doi chieu truoc khi ghi bang:")
    _self_check()
    for name, fn in (("30_lap_lai_a100_vs_t4", table_hw),
                     ("31_he_thong_p2_hieu_chinh_logit", table_p2_system),
                     ("32_per_class_vs_paper_table3", table_per_class),
                     ("33_vs_hai_baseline", table_vs_baselines),
                     ("34_du_dia_con_lai", table_headroom),
                     ("35_bang_2x2_tuong_tac", table_interaction),
                     ("36_mat_can_bang_va_pretrain", table_imbalance_and_pretrain)):
        path = os.path.join(OUT_DIR, name + ".txt")
        with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(fn())
        print("  da ghi %s" % os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
