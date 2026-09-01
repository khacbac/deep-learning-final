# -*- coding: utf-8 -*-
"""Kiem tra cu phap MOI o code cua notebook -- khong o nao duoc mien.

Luat cu ("o nao co dong bat dau bang ! hoac % thi bo qua ca o") da GIAU mot loi that: mot chuoi
bi xuong dong giua chung lam dong sau bat dau bang '!!', the la ca o duoc mien va `ast.parse`
khong bao gio chay. Luat dung: THAY tung dong magic bang `pass` roi parse phan con lai. O magic
that van parse sach, con chuoi hong thi van hong -> khong the nup sau luat magic nua.
"""
import ast, io, json, os, re, sys

MAGIC = re.compile(r"^\s*(!|%%?[A-Za-z])")


def strip_magic(src):
    return "\n".join("pass" if MAGIC.match(l) else l for l in src.splitlines())


def check(nb_path):
    nb = json.load(io.open(nb_path, encoding="utf-8"))
    bad = n = 0
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        n += 1
        src = "".join(c["source"])
        try:
            ast.parse(strip_magic(src))
        except SyntaxError as e:
            bad += 1
            line = src.splitlines()[e.lineno - 1] if e.lineno and e.lineno <= len(src.splitlines()) else ""
            print(f"  [{i:02d}] LOI CU PHAP dong {e.lineno}: {e.msg}")
            print(f"        {line[:90]}")
    print(f"{n} o code | {bad} o hong")
    return bad


if __name__ == "__main__":
    # Chay duoc tu bat ky cwd nao: neo theo vi tri cua chinh script, khong ghim duong dan may.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # tu kiem tra: luat nay PHAI bat duoc dung cai loi vua roi
    broken = 'print(f"\n!! KHONG mount duoc Drive ({e})")\n'
    try:
        ast.parse(strip_magic(broken))
        print("!! TU KIEM TRA HONG: khong bat duoc chuoi bi xuong dong")
        sys.exit(2)
    except SyntaxError:
        print("tu kiem tra: bat duoc chuoi bi xuong dong sau dong '!!'  OK\n")

    # va PHAI cho o magic that di qua
    ast.parse(strip_magic('%%capture\n!pip install -q timm\nimport timm\n'))
    print("tu kiem tra: o magic that van parse sach                  OK\n")

    sys.exit(1 if check("notebooks/final-gastrovision-classification.ipynb") else 0)
