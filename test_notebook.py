#!/usr/bin/env python
"""Kiem thu O DON LE cho notebook -- khong can GPU, khong can du lieu.

    python test_notebook.py

(Truoc day ten la `test_session4.py`. Doi ten 2026-09-01: no khong con la kiem thu mot lan cho
phien 4 nua, ma la bo BAT BIEN cua notebook -- moi phien sau deu phai chay lai.)

Vi sao khong chay smoke ca notebook: mot luot cpu-smoke mat hang chuc phut, tai model that, va
ghi vao cac thu muc tuong doi theo cwd. Cac o duoc kiem duoi day lai la LOGIC THUAN (chon co,
chon seed, doc bang 2x2, chot quy tac), nen kiem duoc bang cach exec dung nhung o do voi mot
moi truong gia lap.

Cai duoc kiem:
  1. O 6b giai dung 7 gia tri SESSION -- va SESSION = 4 bat dung 6 co, khong bat thua co nao.
  2. Them mot co moi vao _ALL_FLAGS thi cac phien CU tu dong khong chay no (bat tat mac dinh).
  3. SEEDS_P2B doi theo RUN_P2B_FULL, va P2c chi chay khi RUN_P2C.
  4. O 15c in dung bang 2x2 o CA BA trang thai: chua co P2c, co P2c va hai don bay xap xi nhau,
     co P2c va hai don bay khac nhau ro -- ba nhanh ket luan khac nhau.
  5. O 15c goi locked_rule() chu khong ghim cung "top3_tta" VA khong doc SELECTION_RULE truc tiep
     (hai loi da mac, moi loi mot lan -- xem RESULTS.md muc 10.9 va 10.10).
  6. Moi tag moi (P2c) deu co builder trong o demo, khong thi doi PROPOSED_TAG se lam vo muc 20b.
  7. vote_rule() o muc 13 tra ve dung ket qua ma pandas o muc 16 cho, tren so THAT cua phien 4 --
     day la ca hai nua cua ban va: mot ham duy nhat, va no phai khop ban cu.
  8. O 19b (he thong de xuat) cung goi locked_rule() chu khong ghim cung ten quy tac -- lo hong
     ma nhom 5 khong phu, va no da lam `tables/23_*` lech khoi con so bao cao (xem docstring
     cua test_19b_rule).
"""
import io
import json
import re
import sys

NB = "notebooks/final-gastrovision-classification.ipynb"
LF = chr(10)


def cells():
    nb = json.loads(io.open(NB, encoding="utf-8").read())
    return [(i, c["cell_type"], "".join(c["source"])) for i, c in enumerate(nb["cells"])]


def cell_by_prefix(prefix):
    hits = [(i, src) for i, t, src in cells() if t == "code" and src.lstrip().startswith(prefix)]
    assert len(hits) == 1, "can dung 1 o cho %r, tim thay %d" % (prefix, len(hits))
    return hits[0][1]


def rule_helpers(selection_rule="best", store=None, is_smoke=False):
    """Nap vote_rule()/locked_rule() THAT tu o muc 13 vao mot namespace gia lap.

    Khong viet lai logic o day: neu test tu cai dat mot ban sao thi no se khop voi chinh no
    chu khong khop voi notebook -- dung cai bay da sinh ra loi o muc 10.10.
    """
    src = cell_by_prefix("SELECTION_RULE = ")
    marker = "# --- Chot quy tac chon checkpoint"
    assert marker in src, "o muc 13 khong con phan chot quy tac"
    body = src[src.index(marker):].replace(
        'print("run_seeds san sang | thu muc luu:", CKPT_DIR)', "")
    ns = {"SELECTION_RULE": selection_rule, "IS_SMOKE": is_smoke,
          "RESULTS_STORE": {} if store is None else store}
    exec(body, ns)
    return ns


FAIL = []
N_CHECK = [0]


def check(name, cond, detail=""):
    N_CHECK[0] += 1
    print("  %-62s %s" % (name, "OK" if cond else "SAI  " + detail))
    if not cond:
        FAIL.append(name)


# ------------------------------------------------------------------ 1, 2, 3: o 6b
def test_session_switch():
    print("1-3. O 6b: giai cong tac phien")
    src = cell_by_prefix("# --------------------------- CONG TAC PHIEN COLAB")
    # Cat bo phan phu thuoc moi truong that (in ra, do gio) -- chi giu phan khai bao ke hoach.
    head = src[:src.index("print(f\"SESSION =")]

    def run(session):
        # Thay dong gan THAT bang regex chu khong thay chuoi "SESSION = 1" cung: mac dinh cua o do
        # doi moi phien, va mot lan da lam bon phep kiem duoi day am tham chay tren sai SESSION.
        body, n = re.subn(r"^SESSION = .+$", "SESSION = %r" % (session,), head, count=1,
                          flags=re.M)
        assert n == 1, "khong tim thay dong gan SESSION trong o 6b"
        ns = {"IS_SMOKE": True, "DEVICE": "cpu", "GPU_NAME": "cpu", "time": __import__("time"),
              "SESSION_T0": 0.0}
        exec(body, ns)
        assert ns["SESSION"] == session, "thay SESSION khong an: %r" % (ns["SESSION"],)
        return ns

    for sess in (0, 1, 2, 3, 4, "all", "manual"):
        try:
            ns = run(sess)
            ok = "SESSION_FLAGS" in ns
        except Exception as e:                                   # noqa: BLE001
            ok, ns = False, {}
            print("     (%r nem loi: %s)" % (sess, e))
        check("SESSION = %r giai duoc" % (sess,), ok)

    ns4 = run(4)
    f4 = ns4["SESSION_FLAGS"]
    want_on = {"RUN_P1_288", "RUN_P2", "RUN_P2_RECIPE_CHECK", "RUN_P2C", "RUN_P2B_FULL",
               "RUN_ABLATIONS"}
    on = {k for k, v in f4.items() if v}
    check("SESSION 4 bat dung %d co" % len(want_on), on == want_on,
          "bat: %s | thieu: %s | thua: %s" % (sorted(on), sorted(want_on - on), sorted(on - want_on)))
    check("SESSION 4 KHONG bat P3/P4/P5/Gate0a",
          not any(f4[k] for k in ("RUN_P3", "RUN_P4", "RUN_P5", "RUN_DETERMINISM_CHECK")))

    # Phien cu phai TU DONG khong chay cac co moi -- day la ly do _plan() tat het roi bat lai.
    for sess in (0, 1, 2, 3):
        f = run(sess)["SESSION_FLAGS"]
        check("SESSION %r khong tu dai them co moi" % sess,
              not (f["RUN_P2C"] or f["RUN_P2B_FULL"] or f["RUN_ABLATIONS"]))

    # Uoc tinh gio phai co mat, va phien 4 phai vua ngan sach 9 gio cua Kaggle.
    plans = run(4)["SESSION_PLANS"]
    check("phien 4 co uoc tinh gio T4", plans[4]["t4"] == 5.0, str(plans[4]["t4"]))
    check("phien 4 vua ngan sach 9 gio", plans[4]["t4"] < 9.0)
    check("phien 'all' >= tong cac phien le",
          plans["all"]["t4"] >= plans[1]["t4"] + plans[4]["t4"])


# ------------------------------------------------------------------ 3: o P2
def test_p2_cell():
    print(LF + "3. O 15c-train: SEEDS_P2B va nhanh P2c")
    src = cell_by_prefix("RUN_P2               = SESSION_FLAGS.get")
    head = src[:src.index("res_p2 = res_p2b = res_p2c = None")]

    for full, want in ((False, [0]), (True, [0, 1, 2])):
        ns = {"SESSION_FLAGS": {"RUN_P2B_FULL": full}, "IS_SMOKE": False, "SEEDS": [0, 1, 2]}
        exec(head, ns)
        check("RUN_P2B_FULL=%s -> SEEDS_P2B=%s" % (full, want), ns["SEEDS_P2B"] == want,
              str(ns["SEEDS_P2B"]))

    ns = {"SESSION_FLAGS": {}, "IS_SMOKE": False, "SEEDS": [0, 1, 2]}
    exec(head, ns)
    check("mac dinh (khong co co) -> P2c TAT", ns["RUN_P2C"] is False)
    check("mac dinh -> P2b 1 seed", ns["SEEDS_P2B"] == [0])
    check("EPOCHS_P2 = 80 khi khong smoke", ns["EPOCHS_P2"] == 80)

    ns = {"SESSION_FLAGS": {}, "IS_SMOKE": True, "SEEDS": [0]}
    exec(head, ns)
    check("EPOCHS_P2 = 2 khi cpu-smoke", ns["EPOCHS_P2"] == 2)

    # Ba loi goi run_seeds phai dung dung tag/builder/seed ma bang 2x2 mong doi.
    check("P2c goi build_coatnet0 @224, 1 seed",
          'tag="P2c_coatnet0_224_modern", seeds=[0]' in src and "build_coatnet0" in src)
    check("P2b dung SEEDS_P2B", "seeds=SEEDS_P2B" in src)
    check("P2c dat set_img_size(224) truoc khi train",
          src.index("if RUN_P2C:") < src.index("res_p2c = run_seeds"))


# ------------------------------------------------------------------ 4, 5: o 15c doc bang
def test_2x2_cell():
    print(LF + "4-5. O 15c-doc: bang 2x2 va ba nhanh ket luan")
    src = cell_by_prefix("# --- Doc ket qua P2: tach don bay CONG THUC")
    check("khong ghim cung 'top3_tta'", 'rule = "top3_tta"' not in src)
    # Loi cua phien 4: o nay chay TRUOC muc 16 nen SELECTION_RULE khi do van la gia tri khoi dau.
    check("khong doc thang SELECTION_RULE", "rule = SELECTION_RULE" not in src)
    check("goi locked_rule() -- cung ham ma muc 16 dung", "rule = locked_rule()" in src)

    def run(store, rule="top3"):
        buf = []
        # locked_rule() phai tra ve dung `rule` du SELECTION_RULE co la gi: bat chuoc dung tinh
        # huong that (o nay chay truoc khi muc 16 gan bien do).
        ns = {"res_p2": object(), "res_p2b": object(), "RESULTS_STORE": store,
              "SELECTION_RULE": "best", "locked_rule": lambda: rule,
              "print": lambda *a, **k: buf.append(" ".join(map(str, a)))}
        exec(src, ns)
        return LF.join(buf)

    RULES = ("best", "smooth", "top3", "best_tta", "smooth_tta", "top3_tta")

    def store(**kw):
        # kw: tag -> (macro_f1, n_seed). "agg" cua notebook that luon co ca 6 quy tac.
        return {t: {"agg": {r: (v[0], 0.008) for r in RULES}, "seeds": list(range(v[1]))}
                for t, v in kw.items()}

    # (a) chua co P2c -> phai NOI RO la o con trong, va noi cach bat
    out = run(store(P1_coatnet0_288=(0.6855, 3), P2_coatnet0_288_modern=(0.7298, 3),
                    B0_densenet121=(0.6780, 3), P2b_densenet121_modern=(0.6835, 1)))
    check("(a) chua co P2c -> canh bao o con trong", "CHUA CHAY P2c" in out)
    check("(a) chi ro cach bat", "SESSION = 4" in out)
    check("(a) van in duoc hai dong da co", "CoAtNet-0 @288" in out and "DenseNet-121 @224" in out)
    check("(a) canh bao P2b 1 seed", "chi 1 seed" in out and "RUN_P2B_FULL" in out)
    check("(a) khong nem loi khi thieu P0", "--" in out)

    # (b) co P2c, hai don bay o 224 XAP XI nhau -> ket luan: khong phai 'cong thuc x kien truc'
    out = run(store(P1_coatnet0_288=(0.6855, 3), P2_coatnet0_288_modern=(0.7298, 3),
                    P0_coatnet0=(0.6814, 3), P2c_coatnet0_224_modern=(0.6800, 1),
                    B0_densenet121=(0.6780, 3), P2b_densenet121_modern=(0.6756, 3)))
    check("(b) xap xi -> ket luan KHONG co thanh phan kien truc", "XAP XI nhau" in out)
    check("(b) khong con canh bao o trong", "CHUA CHAY P2c" not in out)
    check("(b) khong con canh bao P2b 1 seed", "chi 1 seed" not in out)
    check("(b) in ca thanh phan do phan giai", "DO PHAN GIAI x cong thuc" in out)

    # (c) co P2c, hai don bay o 224 KHAC nhau ro -> ket luan nguoc lai
    out = run(store(P1_coatnet0_288=(0.6855, 3), P2_coatnet0_288_modern=(0.7298, 3),
                    P0_coatnet0=(0.6814, 3), P2c_coatnet0_224_modern=(0.7300, 1),
                    B0_densenet121=(0.6780, 3), P2b_densenet121_modern=(0.6756, 3)))
    check("(c) khac ro -> ket luan CO thanh phan kien truc", "KHAC nhau ro" in out)
    check("(c) khong noi xap xi", "XAP XI nhau" not in out)

    # Quy tac chấm phai di theo SELECTION_RULE, khong phai mot chuoi ghim cung
    out = run(store(P1_coatnet0_288=(0.6841, 3), P2_coatnet0_288_modern=(0.7312, 3),
                    B0_densenet121=(0.6863, 3)), rule="top3_tta")
    check("in ra dung quy tac dang dung", "'top3_tta'" in out)


# ------------------------------------------------------------------ 6: builder o demo
def test_demo_builders():
    print(LF + "6. O 20b: moi tag co builder")
    demo = cell_by_prefix("RUN_DEMO = ")
    spec = re.search(r"spec = \{(.*?)\}\n", demo, re.S)
    assert spec, "khong tim thay dict spec trong o demo"
    declared = set(re.findall(r'"([A-Za-z0-9_]+)":\s*\(build_', spec.group(1)))
    # Moi tag duoc run_seeds tao ra o cac o train
    trained = set()
    for _, t, s in cells():
        if t == "code":
            trained |= set(re.findall(r'tag="([A-Za-z0-9_]+)"', s))
    need = {t for t in trained if t.startswith(("B0", "S0", "P0", "P1", "P2"))}
    missing = need - declared
    check("moi tag chinh co builder trong o demo", not missing, "thieu: %s" % sorted(missing))
    check("P2c co builder", "P2c_coatnet0_224_modern" in declared)


# --------------------------------------------------- 7: vote_rule khop ban pandas cu
RULES_6 = ["best", "best_tta", "smooth", "smooth_tta", "top3", "top3_tta"]


def _seed_scores_from_outputs():
    """Diem 6 quy tac tung seed, doc tu OUTPUT that dang nam trong .ipynb."""
    head = re.compile(r"^\[(\S+) seed (\d)\]")
    row = re.compile(r"^\s+best=([\d.]+)\s+best_tta=([\d.]+)\s+smooth=([\d.]+)"
                     r"\s+smooth_tta=([\d.]+)\s+top3=([\d.]+)\s+top3_tta=([\d.]+)\s*$")
    nb = json.loads(io.open(NB, encoding="utf-8").read())
    out, cur = {}, None
    for c in nb["cells"]:
        for o in (c.get("outputs") or []):
            for line in "".join(o.get("text", [])).split(LF):
                m = head.match(line)
                if m:
                    cur = (m.group(1), int(m.group(2)))
                    continue
                m = row.match(line)
                if m and cur is not None:
                    out.setdefault(cur, dict(zip(RULES_6, [float(x) for x in m.groups()])))
                    cur = None
    return out


def test_vote_rule():
    print(LF + "7. O muc 13: vote_rule() khop ban pandas o muc 16")
    sc = _seed_scores_from_outputs()
    check("doc duoc diem tung seed tu output notebook", len(sc) >= 12, "chi %d dong" % len(sc))
    if len(sc) < 12:
        return
    per_tag = {}
    for (tag, _), v in sc.items():
        per_tag.setdefault(tag, []).append(v)
    store = {t: {"agg": {r: (sum(d[r] for d in rows) / len(rows), 0.0) for r in RULES_6},
                 "seeds": list(range(len(rows)))}
             for t, rows in per_tag.items()}

    ns = rule_helpers(store=store)
    winner, avg_rank, mean_f1, ranking, voters = ns["vote_rule"]()
    check("bo phieu dung 4 cau hinh goc", sorted(voters) == sorted(ns["RULE_VOTERS"]))

    # Ban pandas cu, viet lai nguyen van tai day de doi chieu (khong dung numpy/pandas that:
    # xep hang trung binh khi hoa la thu duy nhat can mo phong).
    def avg_rank_ref(tag):
        vals = {r: store[tag]["agg"][r][0] for r in RULES_6}
        out = {}
        for r in RULES_6:
            higher = sum(1 for x in RULES_6 if vals[x] > vals[r])
            ties = sum(1 for x in RULES_6 if vals[x] == vals[r])
            out[r] = higher + (ties + 1) / 2.0
        return out

    ref = {r: sum(avg_rank_ref(t)[r] for t in voters) / len(voters) for r in RULES_6}
    ref_mean = {r: sum(store[t]["agg"][r][0] for t in voters) / len(voters) for r in RULES_6}
    ref_order = sorted(RULES_6, key=lambda c: (ref[c], -ref_mean[c]))
    check("hang trung binh khop ban pandas",
          all(abs(ref[r] - avg_rank[r]) < 1e-9 for r in RULES_6))
    check("thu tu xep hang khop ban pandas", ref_order == ranking)
    check("nguoi thang khop ban pandas", ref_order[0] == winner)
    check("nguoi thang la 'top3' tren so that cua phien 4", winner == "top3")

    # locked_rule(): khong smoke thi theo phieu, smoke thi giu nguyen SELECTION_RULE
    check("locked_rule() = nguoi thang khi khong smoke",
          rule_helpers(selection_rule="best", store=store)["locked_rule"]() == "top3")
    check("locked_rule() giu SELECTION_RULE o che do cpu-smoke",
          rule_helpers(selection_rule="best", store=store, is_smoke=True)["locked_rule"]() == "best")
    check("khong co cau hinh nao -> locked_rule() tra ve SELECTION_RULE",
          rule_helpers(selection_rule="smooth")["locked_rule"]() == "smooth")

    # Moi o doc ket qua chay TRUOC muc 16 deu phai goi ham, khong doc bien
    n_locked = sum(s.count("rule = locked_rule()") for _, t, s in cells() if t == "code")
    n_direct = sum(s.count("    rule = SELECTION_RULE") for _, t, s in cells() if t == "code")
    check("ca 4 o 15c-15f goi locked_rule()", n_locked == 4, "thay %d" % n_locked)
    check("khong o nao con doc thang SELECTION_RULE", n_direct == 0, "con %d" % n_direct)


# ------------------------------------------------------------------ 8: o 19b
def test_19b_rule():
    """O 19b (he thong de xuat) phai lay quy tac tu locked_rule(), khong ghim cung ten quy tac.

    Nhom 5 chi dem 4 o 15c-15f nen o 19b lot luoi: no ghim cung "top3_tta" trong khi vong T4 da
    chot "top3". Hau qua la `report/tables/23_*` in 0,7486 con bao cao in 0,7441 -- cung mot "he
    thong de xuat", hai con so.

    O DAY DOC `build_notebook.py`, KHONG doc notebook. Notebook la ban ghi dong bang cua vong
    chay da sinh ra `report/tables/`, nen o 19b trong do van la ban truoc khi sua; ghep lai bang
    tay thi output se khong con tuong ung voi code. Hai file khop lai sau phien Kaggle ke tiep --
    den luc do bao cao van lay so tu `report/tables-offline/31_*`, von tinh o quy tac da chot.
    """
    print(LF + "8. O 19b: he thong de xuat khong ghim cung quy tac")
    src = io.open("build_notebook.py", encoding="utf-8").read()
    i = src.find('PROPOSED_TAG = "')
    assert i > 0 and src.count('PROPOSED_TAG = "') == 1, "khong tim thay dong gan PROPOSED_TAG"
    cell = src[i:src.index('""")', i)]
    check("tim dung o 19b", "HE THONG DE XUAT" in cell)
    check("o 19b goi locked_rule()", "rule = locked_rule()" in cell)
    check("o 19b khong ghim cung ten quy tac",
          not re.search(r'sel_scores\([^)]*[\'"](best|smooth|top3)', cell))

    # Va bat ky o nao khac cung khong duoc truyen ten quy tac cung vao sel_scores().
    hard = re.findall(r'sel_scores\([^)]*[\'"](?:best|smooth|top3)[^)]*\)', src)
    check("khong o nao truyen ten quy tac cung vao sel_scores()", not hard,
          "con %d: %s" % (len(hard), hard[:3]))

    # Notebook chua duoc chay lai thi noi ro, nhung KHONG bao sai: day la trang thai dung.
    nb_cell = cell_by_prefix("# --- (2) HE THONG DE XUAT DAY DU")
    if "rule = locked_rule()" not in nb_cell:
        print("     (notebook van la ban truoc khi sua -- dung nhu vay cho den phien Kaggle sau;")
        print("      bao cao lay so tu report/tables-offline/31_*, khong tu tables/23_*)")


if __name__ == "__main__":
    print("Kiem thu cac o logic cua notebook (khong GPU, khong du lieu)" + LF)
    test_session_switch()
    test_p2_cell()
    test_2x2_cell()
    test_demo_builders()
    test_vote_rule()
    test_19b_rule()
    if FAIL:
        print(LF + "%d PHEP KIEM SAI: %s" % (len(FAIL), FAIL))
    else:
        print(LF + "TAT CA %d phep kiem DAT" % N_CHECK[0])
    sys.exit(1 if FAIL else 0)
