#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dung lai notebook tu build_notebook.py MA KHONG mat output cua vong chay that.

    python rebuild_notebook.py

Van de: `python build_notebook.py` ghi ra mot notebook SACH -- xoa sach output, tuc xoa
mat ban ghi duy nhat cua vong chay A100 (va 3 hinh ma report/figures/ duoc trich ra tu do).
Script nay lam dung trinh tu an toan:

    1. sao luu ban .ipynb hien tai (dang co output)
    2. chay build_notebook.py
    3. GHEP: giu nguyen ban sao luu, chi thay source cua nhung o MARKDOWN da doi
    4. DUNG LAI voi loi ro rang neu mot o CODE doi -- luc do output khong con dung nua,
       phai chay lai tren Colab, khong duoc ghep

Nho buoc 3, dinh dang cua Colab (indent 2, cell id, metadata) cung duoc giu nguyen nen
diff chi con dung nhung dong prose that su doi.

Sau khi chay xong: `python report/extract.py` de trich lai report/ (khong doi neu chi sua prose).
"""
import io
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "notebooks", "final-gastrovision-classification.ipynb")
BUILDER = os.path.join(HERE, "build_notebook.py")


def src(cell):
    return "".join(cell["source"])


def bo_cuoc(backup, msg):
    """Phuc hoi ban sao luu roi XOA no, va thoat voi loi.

    Truoc day cac nhanh nay chi phuc hoi chu khong xoa, nen moi lan tu choi ghep lai de lai mot
    file `.ipynb.bak-<timestamp>` nam canh notebook that -- vai MB, trong y het notebook that,
    va git khong bat vi khong ai them no vao .gitignore. Sau khi da phuc hoi thi ban sao luu
    giong het NB, nen giu lai khong duoc gi.
    """
    shutil.copy2(backup, NB)
    os.remove(backup)
    sys.exit(msg)


def main():
    if not os.path.exists(NB):
        sys.exit("chua co notebook o " + NB)

    backup = NB + ".bak-" + time.strftime("%Y%m%d-%H%M%S")
    shutil.copy2(NB, backup)
    raw = io.open(backup, encoding="utf-8").read()
    base = json.loads(raw)
    n_out = sum(1 for c in base["cells"] if c.get("outputs"))
    print("1/3  da sao luu -> %s  (%d o dang co output)" % (os.path.basename(backup), n_out))

    r = subprocess.run([sys.executable, BUILDER], cwd=HERE)
    if r.returncode != 0:
        bo_cuoc(backup, "build_notebook.py loi -> da phuc hoi ban sao luu")
    print("2/3  da dung lai tu build_notebook.py")

    fresh = json.load(io.open(NB, encoding="utf-8"))

    if len(base["cells"]) != len(fresh["cells"]):
        bo_cuoc(backup,
                "SO O DOI (%d -> %d): khong ghep duoc tu dong. Ban dung lai da bi phuc hoi;\n"
                "hay ghep tay hoac chay lai notebook tren Colab."
                % (len(base["cells"]), len(fresh["cells"])))

    changed_md, changed_code = [], []
    for i, (a, b) in enumerate(zip(base["cells"], fresh["cells"])):
        if a["cell_type"] != b["cell_type"]:
            bo_cuoc(backup, "o %d doi loai (%s -> %s): khong ghep duoc. Da phuc hoi ban sao luu."
                            % (i, a["cell_type"], b["cell_type"]))
        if src(a) != src(b):
            (changed_code if a["cell_type"] == "code" else changed_md).append(i)

    if changed_code:
        bo_cuoc(backup,
                "O CODE da doi: %s\n"
                "Output dang luu khong con tuong ung voi code nay, nen GHEP LA SAI.\n"
                "Ban dung lai da bi phuc hoi (va notebook giu nguyen output cua vong chay that).\n"
                "Day la trang thai DUNG khi vua sua code: build_notebook.py di truoc, notebook\n"
                "bat kip o phien Kaggle/Colab ke tiep. Dung ghep tay."
                % changed_code)

    for i in changed_md:
        base["cells"][i]["source"] = fresh["cells"][i]["source"]

    out = json.dumps(base, ensure_ascii=False, indent=2)
    if raw.endswith("\n"):
        out += "\n"
    io.open(NB, "w", encoding="utf-8", newline="\n").write(out)

    kept = sum(1 for c in base["cells"] if c.get("outputs"))
    figs = sum(1 for c in base["cells"] for o in (c.get("outputs") or [])
               if "image/png" in (o.get("data") or {}))
    print("3/3  da ghep %d o markdown; giu %d o output + %d hinh" % (len(changed_md), kept, figs))
    if kept != n_out:
        sys.exit("!! so o co output doi (%d -> %d) -- kiem tra lai" % (n_out, kept))
    os.remove(backup)
    print("     xoa ban sao luu tam. Buoc cuoi: python report/extract.py")


if __name__ == "__main__":
    main()
