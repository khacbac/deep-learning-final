# -*- coding: utf-8 -*-
"""Sinh bo slide bao ve -> report/slides.html  (dan y v3.3 da duyet 05-09-2026).

Cau truc: MO DAU + 5 hoi, 19 slide, ngon ngu gian di (thuat ngu = keyword kem chu
thich 1 dong), du chu thich de NGUOI DOC TU HIEU khong can nguoi trinh bay.
  Mo dau  : de tai (anh mau that tu dataset)
  Hoi 1   : phuong phap tiep can (pipeline, vi sao chon 3 kien truc — co hinh,
            chien luoc bac thang + chu giai ky hieu, van hanh thuc te)
  Hoi 2   : do cho dung (nhieu, nhieu theo lop, tai lap baseline)
  Hoi 3   : cai thien bang gi (ket qua am, cong thuc moi la gi, tuong tac, thac nuoc)
  Hoi 4   : kiem chung & so sanh (CI, cac paper khac, lop hiem, CPU, san pham)
  Hoi 5   : nhin thang (du lieu con sai + huong xu ly, han che + ket)

Moi con so tren slide da co trong BAO_CAO.md va do check_numbers.py doi chieu.
Can data/ da giai nen (lay anh mau S1 + nen hinh S3).

Chay:  python report/build_slides.py  ->  report/slides.html (tu chua, anh data-URI)
Dieu huong: <- -> / Space / click | phim G = luoi tong quan | in duoc (Ctrl+P).
"""
import base64
import io
import os
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE / "slides.html"
DATA = REPO / "data"


def b64png(rel):
    with open(HERE / rel, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def _class_folder(class_dir):
    for r, dirs, _ in os.walk(DATA):
        if class_dir in dirs:
            return Path(r) / class_dir
    raise AssertionError(f"khong thay thu muc lop '{class_dir}' duoi data/")


def _encode(img, size, q):
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((size, size))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def sample_jpg(class_dir, idx=2, size=360, q=72):
    """Lay 1 anh that tu dataset: crop vuong giua, thu nho, nen JPEG -> data URI."""
    folder = _class_folder(class_dir)
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    return _encode(Image.open(files[min(idx, len(files) - 1)]).convert("RGB"), size, q)


def exact_jpg(class_dir, fname, size=300, q=72):
    """Lay DUNG mot file theo ten — dung cho bang chung tu bang audit (tables/08)."""
    return _encode(Image.open(_class_folder(class_dir) / fname).convert("RGB"), size, q)


IMG_EDA = b64png("figures/06_eda.png")
IMG_PERCLASS = b64png("figures/18_per_class_va_confusion.png")
IMG_DEMO = b64png("demo/29b_demo_gradio_cpu.png")

IMG_NORMAL_LB = sample_jpg("Normal mucosa and vascular pattern in the large bowel", idx=12)
SAMPLES = [
    ("Polyp đại tràng", sample_jpg("Colon polyps")),
    ("Ung thư đại trực tràng", sample_jpg("Colorectal cancer")),
    ("Niêm mạc bình thường", IMG_NORMAL_LB),
    ("Viêm thực quản", sample_jpg("Esophagitis")),
    ("Dụng cụ can thiệp", sample_jpg("Accessory tools")),
]
BG_POLYP = sample_jpg("Colon polyps", idx=5, size=340, q=70)

# Bang chung du lieu (ten file lay tu report/tables/08_audit_gan_trung.txt):
# cap anh GIONG HET NHAU (cosine = 1.0000) nhung mang 2 nhan khac nhau.
IMG_MIS_A = exact_jpg("Esophagitis", "N2DaTmFs.jpg")
IMG_MIS_B = exact_jpg("Normal esophagus", "WdSYgDiw.jpg")
# cap DE NHAM: hai nhom khac nhau nhung nhin rat giong nhau.
IMG_CECUM = sample_jpg("Cecum", size=300)

CSS = """
:root{
  --paper:#F6F8F7; --card:#FFFFFF; --ink:#1C2321; --muted:#56635E;
  --teal:#1F6E5E; --teal-mid:#4E8F80; --teal-soft:#E3EEEA; --red:#A8402F;
  --line:#D7DEDA; --flat:#C9D3CE;
}
*{box-sizing:border-box;margin:0}
html,body{height:100%}
body{background:var(--ink);font-family:'Be Vietnam Pro',system-ui,-apple-system,'Segoe UI',sans-serif;color:var(--ink)}
#stage{position:fixed;inset:0;display:flex;align-items:center;justify-content:center;background:var(--ink)}
.slide{width:1280px;height:720px;background:var(--paper);display:none;flex-direction:column;
  padding:40px 62px 30px;position:absolute;transform-origin:center center}
.slide.active{display:flex}
.slide header{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px}
.eyebrow{font-family:'IBM Plex Mono',monospace;font-size:13px;letter-spacing:.18em;color:var(--teal);text-transform:uppercase}
.pageno{font-family:'IBM Plex Mono',monospace;font-size:13px;color:var(--muted)}
h1{font-family:'Archivo',sans-serif;font-weight:700;font-size:52px;line-height:1.08;text-wrap:balance}
h2{font-family:'Archivo',sans-serif;font-weight:650;font-size:36px;line-height:1.14;margin-bottom:14px;text-wrap:balance}
.slide footer{margin-top:auto;padding-top:8px;border-top:1px solid var(--line);
  display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--muted)}
.body{flex:1;min-height:0;display:flex;flex-direction:column}
p,li{font-size:19px;line-height:1.5}
.lead{font-size:22px;line-height:1.5;max-width:66ch}
ul{padding-left:24px;display:flex;flex-direction:column;gap:9px}
.muted{color:var(--muted)}
.accent{color:var(--teal);font-weight:600}
.neg{color:var(--red);font-weight:600}
.mono,td.num,.num{font-family:'IBM Plex Mono',monospace;font-variant-numeric:tabular-nums}
code{font-family:'IBM Plex Mono',monospace;font-size:.9em;background:var(--teal-soft);padding:1px 6px;border-radius:4px}
.cols{display:grid;gap:32px;flex:1;min-height:0;align-items:start}
.cols.c2{grid-template-columns:1fr 1fr}
.cols.c38{grid-template-columns:5fr 4fr}
table{border-collapse:collapse;width:100%}
th{font-family:'IBM Plex Mono',monospace;font-size:13px;letter-spacing:.05em;text-transform:uppercase;
  color:var(--muted);text-align:left;padding:7px 12px;border-bottom:2px solid var(--ink);font-weight:500}
td{padding:8px 12px;border-bottom:1px solid var(--line);font-size:18px}
td.num,th.num{text-align:right}
tr.hl td{background:var(--teal-soft);font-weight:600}
.stat{display:flex;flex-direction:column;gap:2px}
.stat .v{font-family:'IBM Plex Mono',monospace;font-size:58px;font-weight:600;color:var(--teal);line-height:1.05}
.stat .v.plain{color:var(--ink)}
.stat .v.small{font-size:42px}
.stat .k{font-size:16px;color:var(--muted);max-width:36ch}
.statrow{display:flex;gap:52px;margin:20px 0}
figure{margin:0;display:flex;flex-direction:column;gap:8px;min-height:0}
figure img{max-width:100%;max-height:100%;object-fit:contain;border:1px solid var(--line);background:#fff}
figcaption{font-size:15px;color:var(--muted);line-height:1.45}
.note{font-size:16px;color:var(--muted);border-left:3px solid var(--teal);padding:5px 0 5px 14px;max-width:78ch;line-height:1.5}
.takeaway{font-size:20px;line-height:1.45;background:var(--teal-soft);border-left:4px solid var(--teal);
  padding:11px 18px;margin-top:12px}
.takeaway strong{color:var(--teal)}
.gloss{font-size:14px;color:var(--muted);margin-top:10px;line-height:1.5}
.gloss b{color:var(--ink);font-weight:600}
.titleslide{justify-content:center;gap:18px}
.titleslide .who{font-family:'IBM Plex Mono',monospace;font-size:15px;color:var(--muted);letter-spacing:.05em}
.strip{display:flex;gap:14px;margin:10px 0}
.strip figure{flex:1;min-width:0}
.strip img{width:100%;height:168px;object-fit:cover;border:1px solid var(--line)}
.strip figcaption{text-align:center;font-size:14px}
.pair{display:flex;gap:10px}
.pair figure{flex:1;min-width:0}
.pair img{width:100%;height:150px;object-fit:cover;border:1px solid var(--line)}
.pair figcaption{text-align:center;font-size:13.5px}
kbd{font-family:'IBM Plex Mono',monospace;background:var(--card);border:1px solid var(--line);border-radius:4px;padding:0 6px;font-size:12px}
#hint{position:fixed;right:16px;bottom:12px;color:#9FB0AA;font-size:12px;font-family:'IBM Plex Mono',monospace;opacity:.85}
body.grid #stage{position:static;display:grid;grid-template-columns:repeat(3,1fr);gap:14px;padding:14px;background:var(--ink);height:auto}
body.grid .slide{display:flex;position:static;transform:scale(1)!important;width:100%;height:auto;aspect-ratio:16/9;
  padding:18px 26px;cursor:pointer;overflow:hidden;zoom:.33}
body.grid #hint{display:none}
@media print{
  body{background:#fff}
  #stage{position:static;display:block}
  .slide{display:flex;position:static;transform:none!important;page-break-after:always;width:100%;height:97vh}
  #hint{display:none}
}
:focus-visible{outline:3px solid var(--teal);outline-offset:2px}
"""

JS = """
const slides=[...document.querySelectorAll('.slide')];let cur=0;
function show(i){cur=Math.max(0,Math.min(slides.length-1,i));
  slides.forEach((s,j)=>s.classList.toggle('active',j===cur));
  location.hash='s'+(cur+1);fit();}
function fit(){const s=Math.min(innerWidth/1300,innerHeight/740);
  slides.forEach(el=>el.style.transform=`scale(${s})`);}
addEventListener('resize',fit);
addEventListener('keydown',e=>{
  if(e.key==='ArrowRight'||e.key===' '||e.key==='PageDown')show(cur+1);
  else if(e.key==='ArrowLeft'||e.key==='PageUp')show(cur-1);
  else if(e.key==='Home')show(0);else if(e.key==='End')show(slides.length-1);
  else if(e.key==='g'||e.key==='G')document.body.classList.toggle('grid');});
addEventListener('click',e=>{
  if(document.body.classList.contains('grid')){
    const s=e.target.closest('.slide');
    if(s){document.body.classList.remove('grid');show(slides.indexOf(s));}return;}
  if(e.clientX>innerWidth*0.66)show(cur+1);else if(e.clientX<innerWidth*0.2)show(cur-1);});
const m=location.hash.match(/^#s(\\d+)$/);show(m?+m[1]-1:0);
"""


def vn(v, nd=4):
    return f"{v:.{nd}f}".replace(".", ",")


# ---------------- SVG: pipeline end-to-end (S2) ---------------- #
def pipeline_svg():
    blocks = [
        ("8.000 ảnh", "nội soi, 27 nhóm"),
        ("Làm sạch", "kiểm trùng lặp, rò rỉ"),
        ("Chia cố định", "60 · 20 · 20"),
        ("Học chuyển giao", "3 kiến trúc"),
        ("Huấn luyện", "lưu lại mọi dự đoán"),
        ("Một thước đo", "chung cho tất cả"),
        ("Phân tích", "0 GPU · demo"),
    ]
    bw, bh, gap, x0, y0 = 148, 84, 14, 10, 30
    p = ['<svg viewBox="0 0 1160 150" role="img" aria-label="So do giai phap">']
    for i, (t1, t2) in enumerate(blocks):
        x = x0 + i * (bw + gap)
        fill = "#1F6E5E" if i in (3, 6) else "#FFFFFF"
        tcol = "#FFFFFF" if i in (3, 6) else "#1C2321"
        scol = "#CFE3DD" if i in (3, 6) else "#56635E"
        p.append(f'<rect x="{x}" y="{y0}" width="{bw}" height="{bh}" rx="10" fill="{fill}" '
                 f'stroke="#1F6E5E" stroke-width="1.5"/>')
        p.append(f'<text x="{x + bw / 2:.0f}" y="{y0 + 36}" font-size="17" font-weight="700" fill="{tcol}" '
                 f'text-anchor="middle" font-family="Be Vietnam Pro,sans-serif">{t1}</text>')
        p.append(f'<text x="{x + bw / 2:.0f}" y="{y0 + 60}" font-size="13" fill="{scol}" '
                 f'text-anchor="middle" font-family="Be Vietnam Pro,sans-serif">{t2}</text>')
        if i < len(blocks) - 1:
            ax = x + bw + gap / 2
            p.append(f'<path d="M {ax - 5} {y0 + bh / 2} l 8 0 m -3 -5 l 5 5 l -5 5" stroke="#56635E" '
                     f'stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>')
    p.append("</svg>")
    return "".join(p)


# ---------------- SVG: vi sao 3 kien truc (S3) — 3 panel tren anh that ---------------- #
def arch_svg():
    pw, ph, gap, x0 = 340, 250, 44, 26
    lesion = (215, 130)          # vi tri ton thuong tren anh 340px (uoc luong vung polyp)
    p = ['<svg viewBox="0 0 1160 330" role="img" aria-label="Ba cach nhin cua ba kien truc">']

    def panel(i, overlays, title, sub):
        x = x0 + i * (pw + gap)
        p.append(f'<image href="{BG_POLYP}" x="{x}" y="0" width="{pw}" height="{ph}" preserveAspectRatio="xMidYMid slice"/>')
        p.append(f'<rect x="{x}" y="0" width="{pw}" height="{ph}" fill="none" stroke="#1C2321" stroke-width="2"/>')
        p.extend(overlays(x))
        p.append(f'<text x="{x + pw / 2}" y="{ph + 34}" font-size="19" font-weight="700" fill="#1C2321" '
                 f'text-anchor="middle" font-family="Archivo,sans-serif">{title}</text>')
        p.append(f'<text x="{x + pw / 2}" y="{ph + 58}" font-size="14.5" fill="#56635E" '
                 f'text-anchor="middle" font-family="Be Vietnam Pro,sans-serif">{sub}</text>')

    def cnn(x):
        lx, ly = x + lesion[0], lesion[1]
        o = [f'<circle cx="{lx}" cy="{ly}" r="62" fill="none" stroke="#FFD34D" stroke-width="4"/>']
        for k in range(1, 3):
            o.append(f'<line x1="{lx - 62}" y1="{ly - 62 + k * 41.3:.0f}" x2="{lx + 62}" y2="{ly - 62 + k * 41.3:.0f}" stroke="#FFD34D" stroke-width="1.6" opacity="0.9"/>')
            o.append(f'<line x1="{lx - 62 + k * 41.3:.0f}" y1="{ly - 62}" x2="{lx - 62 + k * 41.3:.0f}" y2="{ly + 62}" stroke="#FFD34D" stroke-width="1.6" opacity="0.9"/>')
        o.append(f'<line x1="{lx + 46}" y1="{ly + 46}" x2="{lx + 86}" y2="{ly + 88}" stroke="#FFD34D" stroke-width="7" stroke-linecap="round"/>')
        o.append(f'<rect x="{x + 10}" y="12" width="150" height="26" rx="13" fill="#1C2321" opacity="0.75"/>')
        o.append(f'<text x="{x + 85}" y="30" font-size="14" fill="#FFFFFF" text-anchor="middle" font-family="Be Vietnam Pro,sans-serif">soi gần từng mảng</text>')
        return o

    def transformer(x):
        o = []
        for k in range(1, 4):
            o.append(f'<line x1="{x}" y1="{k * ph / 4:.0f}" x2="{x + pw}" y2="{k * ph / 4:.0f}" stroke="#8FD3C7" stroke-width="1" stroke-dasharray="5 5" opacity="0.9"/>')
            o.append(f'<line x1="{x + k * pw / 4:.0f}" y1="0" x2="{x + k * pw / 4:.0f}" y2="{ph}" stroke="#8FD3C7" stroke-width="1" stroke-dasharray="5 5" opacity="0.9"/>')
        lx, ly = x + lesion[0], lesion[1]
        for sx, sy in ((x + 42, 32), (x + 42, ph - 36), (x + pw - 46, 34), (x + 90, ph - 60)):
            o.append(f'<line x1="{sx}" y1="{sy}" x2="{lx}" y2="{ly}" stroke="#8FD3C7" stroke-width="2.4" opacity="0.95"/>')
            o.append(f'<circle cx="{sx}" cy="{sy}" r="7" fill="#8FD3C7"/>')
        o.append(f'<circle cx="{lx}" cy="{ly}" r="11" fill="#FFD34D" stroke="#1C2321" stroke-width="2"/>')
        o.append(f'<rect x="{x + 10}" y="12" width="176" height="26" rx="13" fill="#1C2321" opacity="0.75"/>')
        o.append(f'<text x="{x + 98}" y="30" font-size="14" fill="#FFFFFF" text-anchor="middle" font-family="Be Vietnam Pro,sans-serif">lùi ra nhìn toàn khung</text>')
        return o

    def hybrid(x):
        o = []
        for k in range(1, 8):
            o.append(f'<line x1="{x + k * pw / 8:.0f}" y1="{ph - 62}" x2="{x + k * pw / 8:.0f}" y2="{ph}" stroke="#FFD34D" stroke-width="1.4" opacity="0.9"/>')
        o.append(f'<line x1="{x}" y1="{ph - 62}" x2="{x + pw}" y2="{ph - 62}" stroke="#FFD34D" stroke-width="1.6"/>')
        lx, ly = x + lesion[0], lesion[1]
        for sx in (x + 46, x + 130, x + pw - 40):
            o.append(f'<path d="M {sx} {ph - 66} Q {(sx + lx) / 2:.0f} {ly - 88} {lx} {ly}" fill="none" stroke="#8FD3C7" stroke-width="2.4"/>')
        o.append(f'<circle cx="{lx}" cy="{ly}" r="11" fill="#FFD34D" stroke="#1C2321" stroke-width="2"/>')
        o.append(f'<rect x="{x + 10}" y="12" width="120" height="26" rx="13" fill="#1C2321" opacity="0.75"/>')
        o.append(f'<text x="{x + 70}" y="30" font-size="14" fill="#FFFFFF" text-anchor="middle" font-family="Be Vietnam Pro,sans-serif">làm cả hai</text>')
        return o

    panel(0, cnn, "CNN — DenseNet-121 (B0)", "giỏi kết cấu bề mặt: mảng niêm mạc, bề mặt polyp")
    panel(1, transformer, "Transformer — Swin-T (S0)", "giỏi bối cảnh: vị trí giải phẫu, các vùng xa liên hệ nhau")
    panel(2, hybrid, "Hybrid — CoAtNet-0 (P·)", "tầng dưới soi kết cấu, tầng trên nối bối cảnh")
    p.append("</svg>")
    return "".join(p)


# ---------------- SVG: thac nuoc (S12) ---------------- #
def waterfall_svg():
    steps = [
        ("Tái lập mốc cũ", "DenseNet-121", None, 0.6686, "start"),
        ("Đổi kiến trúc", "sang hybrid", 0.0034, None, "flat"),
        ("Ảnh nét hơn", "224 → 288px", 0.0041, None, "flat"),
        ("Công thức mới", "học kỹ hơn (S10)", 0.0443, None, "big"),
        ("Gộp 3 checkpoint", "không tốn GPU", 0.0094, None, "meas"),
        ("Hiệu chỉnh tần suất", "không tốn GPU", 0.0143, None, "meas"),
        ("Hệ thống đề xuất", "P2", None, 0.7441, "end"),
    ]
    v0, v1, top, phh = 0.635, 0.760, 34, 280
    Y = lambda v: top + (v1 - v) / (v1 - v0) * phh
    colw, gap, x0 = 118, 34, 60
    color = {"start": "#56635E", "end": "#1F6E5E", "big": "#1F6E5E", "meas": "#4E8F80", "flat": "#C9D3CE"}
    p = ['<svg viewBox="0 0 1130 405" role="img" aria-label="Cong don muc tang tu 0,6686 len 0,7441">']
    for t in (0.65, 0.70, 0.75):
        p.append(f'<line x1="42" y1="{Y(t):.0f}" x2="1110" y2="{Y(t):.0f}" stroke="#D7DEDA" stroke-width="1"/>')
        p.append(f'<text x="36" y="{Y(t) + 5:.0f}" font-size="14" fill="#56635E" text-anchor="end" font-family="IBM Plex Mono,monospace">{vn(t, 2)}</text>')
    cum, prev_top = 0.6686, None
    for i, (name, sub, d, absv, kind) in enumerate(steps):
        x = x0 + i * (colw + gap)
        c = color[kind]
        if kind in ("start", "end"):
            y_hi = Y(absv)
            p.append(f'<rect x="{x}" y="{y_hi:.0f}" width="{colw}" height="{Y(v0) - y_hi:.0f}" fill="{c}" rx="3"/>')
            p.append(f'<text x="{x + colw / 2:.0f}" y="{y_hi - 10:.0f}" font-size="19" font-weight="700" '
                     f'fill="{"#1F6E5E" if kind == "end" else "#1C2321"}" text-anchor="middle" font-family="IBM Plex Mono,monospace">{vn(absv)}</text>')
            cum = absv
        else:
            y_hi = Y(cum + d)
            h = max(Y(cum) - y_hi, 4)
            p.append(f'<rect x="{x}" y="{y_hi:.0f}" width="{colw}" height="{h:.0f}" fill="{c}" rx="3"/>')
            p.append(f'<text x="{x + colw / 2:.0f}" y="{y_hi - 8:.0f}" font-size="16" font-weight="600" '
                     f'fill="{"#56635E" if kind == "flat" else "#1F6E5E"}" text-anchor="middle" font-family="IBM Plex Mono,monospace">+{vn(d)}</text>')
            cum += d
        if prev_top is not None:
            p.append(f'<line x1="{x - gap}" y1="{prev_top:.0f}" x2="{x}" y2="{Y(cum):.0f}" stroke="#9FB0AA" stroke-width="1" stroke-dasharray="3 3"/>')
        prev_top = Y(cum)
        p.append(f'<text x="{x + colw / 2:.0f}" y="{top + phh + 26}" font-size="15" fill="#1C2321" text-anchor="middle" font-family="Be Vietnam Pro,sans-serif" font-weight="600">{name}</text>')
        p.append(f'<text x="{x + colw / 2:.0f}" y="{top + phh + 46}" font-size="13" fill="#56635E" text-anchor="middle" font-family="Be Vietnam Pro,sans-serif">{sub}</text>')
    # vach moc ve SAU cac cot de khong bi che; nhan dat o khoang trong giua bieu do
    p.append(f'<line x1="42" y1="{Y(0.6504):.0f}" x2="1110" y2="{Y(0.6504):.0f}" stroke="#A8402F" stroke-width="2" stroke-dasharray="7 5"/>')
    p.append(f'<text x="530" y="{Y(0.6504) + 22:.0f}" font-size="14.5" fill="#A8402F" text-anchor="middle" font-weight="600" font-family="IBM Plex Mono,monospace">mốc phải vượt: 0,6504</text>')
    p.append("</svg>")
    return "".join(p)


# ---------------- SVG: khoang tin cay (S13) ---------------- #
def ci_svg():
    x0, x1, w = 0.62, 0.80, 980
    X = lambda v: 90 + (v - x0) / (x1 - x0) * w
    rows = [
        ("Mốc cũ tái lập (B0)", 0.6686, 0.6452, 0.6920, "#56635E", "vùng dao động ±σ, 3 lần chạy"),
        ("Hệ thống đề xuất (P2)", 0.7441, 0.6986, 0.7736, "#1F6E5E", "khoảng tin cậy 95%"),
        ("P2 · gộp 3 lần chạy", 0.7587, 0.7110, 0.7924, "#4E8F80", "khoảng tin cậy 95%"),
    ]
    p = ['<svg viewBox="0 0 1120 316" role="img" font-family="IBM Plex Mono,monospace" aria-label="Khoang tin cay so voi 0,6504">']
    for t in (0.65, 0.70, 0.75):
        p.append(f'<line x1="{X(t):.0f}" y1="36" x2="{X(t):.0f}" y2="264" stroke="#D7DEDA" stroke-width="1"/>')
        p.append(f'<text x="{X(t):.0f}" y="288" font-size="15" fill="#56635E" text-anchor="middle">{vn(t, 2)}</text>')
    bx = X(0.6504)
    p.append(f'<line x1="{bx:.0f}" y1="24" x2="{bx:.0f}" y2="264" stroke="#A8402F" stroke-width="2" stroke-dasharray="6 5"/>')
    p.append(f'<text x="{bx:.0f}" y="16" font-size="15" fill="#A8402F" text-anchor="middle" font-weight="600">mốc phải vượt 0,6504</text>')
    y = 76
    for name, mid, lo, hi, color, note in rows:
        p.append(f'<line x1="{X(lo):.0f}" y1="{y}" x2="{X(hi):.0f}" y2="{y}" stroke="{color}" stroke-width="6" stroke-linecap="round"/>')
        p.append(f'<circle cx="{X(mid):.0f}" cy="{y}" r="10" fill="{color}" stroke="#F6F8F7" stroke-width="2"/>')
        p.append(f'<text x="{X(lo):.0f}" y="{y - 20}" font-size="17" fill="#1C2321" font-family="Be Vietnam Pro,sans-serif" font-weight="600">{name}</text>')
        p.append(f'<text x="{X(lo):.0f}" y="{y + 28}" font-size="15" fill="#56635E">{vn(mid)} · [{vn(lo)}; {vn(hi)}] {note}</text>')
        y += 86
    p.append("</svg>")
    return "".join(p)


def slide(no, total, section, body, cls=""):
    return f"""<section class="slide {cls}" id="s{no}">
<header><span class="eyebrow">{section}</span><span class="pageno">{no:02d} / {total}</span></header>
<div class="body">{body}</div>
<footer><span>GastroVision · AIN501 Deep Learning · nhóm 2 thành viên</span><span>05-09-2026</span></footer>
</section>"""


T = 19
S = []

# ============ MO DAU ============ #
strip = "".join(f'<figure><img src="{u}" alt="{lb}"/><figcaption>{lb}</figcaption></figure>'
                for lb, u in SAMPLES)
S.append(slide(1, T, "Mở đầu · Đề tài", f"""
<h1 style="font-size:42px">Nhận diện 22 loại hình ảnh nội soi tiêu hoá bằng học sâu</h1>
<p class="lead">Máy nhìn một khung hình nội soi và gọi tên nó: bộ phận bình thường (dạ dày, thực quản,
manh tràng…) hay tổn thương (polyp, ung thư, viêm…). Đây là bước nền cho hệ thống hỗ trợ bác sĩ
trong lúc soi.</p>
<div class="strip">{strip}</div>
<div class="cols c2" style="flex:0 0 auto">
<ul>
<li><strong>Dữ liệu</strong>: bộ GastroVision — 8.000 ảnh từ 2 bệnh viện Na Uy. Sau lọc còn 22 nhóm / 7.930 ảnh.</li>
<li><strong>Vì sao khó</strong>: nhiều nhóm nhìn rất giống nhau. Nhóm đông nhất có 1.467 ảnh; nhóm ít nhất chỉ 29.</li>
</ul>
<ul>
<li><strong>Nhiệm vụ</strong>: làm lại được mốc đã công bố (<span class="mono">0,6504</span>).
Sau đó <strong>vượt mốc này một cách có bằng chứng</strong>.</li>
</ul>
</div>
<p class="gloss"><b>macro-F1</b>: điểm trung bình của từng nhóm, nhóm hiếm được tính công bằng như nhóm lớn (thang 0→1). Năm ảnh trên là mẫu thật từ bộ dữ liệu.</p>"""))

# ============ HOI 1 — PHUONG PHAP TIEP CAN ============ #
S.append(slide(2, T, "Hồi 1 · Phương pháp tiếp cận", f"""
<h2>Toàn cảnh giải pháp: một đường ống, mọi mô hình đi chung</h2>
{pipeline_svg()}
<ul style="margin-top:8px">
<li><strong>Làm sạch trước khi học</strong>: quét ảnh trùng lặp và ảnh "lọt" giữa tập học – tập thi.
Đề thi mà lẫn trong vở học thì điểm sẽ ảo.</li>
<li><strong>Học chuyển giao</strong>: bắt đầu từ mô hình đã học 1,3 triệu ảnh đời thường (ImageNet).
Sau đó dạy tiếp bằng ảnh nội soi, thay vì học từ con số 0.</li>
<li><strong>Một thước đo chung</strong>: mọi kiến trúc được chấm bằng đúng một bộ đề và một cách chấm.
Nhờ vậy so sánh mới công bằng.</li>
<li><strong>Lưu lại mọi dự đoán</strong> (logits): phân tích về sau chạy lại từ file đã lưu.
Không tốn thêm giờ GPU nào.</li>
</ul>
<p class="gloss"><b>logits</b>: điểm thô mô hình chấm cho từng nhóm trước khi chọn đáp án — lưu lại được thì mọi cách chấm khác đều tính lại được sau.</p>"""))

S.append(slide(3, T, "Hồi 1 · Phương pháp tiếp cận", f"""
<h2>Vì sao chọn 3 kiến trúc — ba cách "nhìn" một tổn thương</h2>
{arch_svg()}
<p class="takeaway" style="margin-top:10px">Ba kiến trúc = <strong>ba giả thuyết về nơi thông tin nằm</strong>:
trong kết cấu gần, trong bối cảnh xa, hay cả hai. Nhóm đặt cược vào hybrid —
<em>và số liệu ở Hồi 3 sẽ cho câu trả lời bất ngờ: kiến trúc đứng một mình gần như không ăn thua.</em></p>"""))

S.append(slide(4, T, "Hồi 1 · Phương pháp tiếp cận", """
<h2>Chiến lược: bậc thang — mỗi bậc chỉ đổi đúng MỘT thứ</h2>
<div class="cols c2">
<div>
<table>
<tr><th>Bậc</th><th>Đổi gì so với bậc trước</th></tr>
<tr><td><code>B0</code> DenseNet-121</td><td>mốc cũ — phải khớp 0,6504 trước đã</td></tr>
<tr><td><code>S0</code> Swin-T · <code>P0</code> CoAtNet</td><td>chỉ đổi <strong>kiến trúc</strong></td></tr>
<tr><td><code>P1</code> CoAtNet @288</td><td>chỉ đổi <strong>độ nét ảnh</strong></td></tr>
<tr class="hl"><td><code>P2</code> + công thức mới</td><td>chỉ đổi <strong>cách huấn luyện</strong></td></tr>
<tr><td><code>P2b</code> · <code>P2c</code></td><td>các <strong>đối chứng</strong> để tách bạch nguyên nhân</td></tr>
</table>
</div>
<div>
<p><strong>Chú giải ký hiệu</strong> (dùng suốt bài):</p>
<ul>
<li><code>B</code> = Baseline CNN của bài báo · <code>S</code> = Swin (baseline 2)</li>
<li><code>P</code> = các bậc dẫn tới mô hình đề xuất (P0, P1 là <em>bậc trung gian</em>, không phải đề xuất)</li>
<li><code>M</code> = bản chạy máy thường (CPU) · <code>T</code> = thí nghiệm học chuyển giao · <code>A</code> = thử nghiệm phụ</li>
</ul>
<p class="note">Hai luật chơi: mọi thay đổi phải có <strong>phép thử đối chứng</strong>, và
<strong>kết quả xấu cũng công bố</strong> — vì "không ăn thua" cũng là thông tin.</p>
</div>
</div>"""))

S.append(slide(5, T, "Hồi 1 · Phương pháp tiếp cận", """
<h2>Vận hành thực tế — mọi con số đều từ các lần chạy thật</h2>
<div class="cols c2">
<div>
<table>
<tr><th>Vòng chạy</th><th>Ở đâu · bao lâu</th></tr>
<tr><td>19 lượt huấn luyện chính</td><td>Kaggle GPU T4 — 21→118 phút/lượt</td></tr>
<tr><td>Vòng kiểm chứng độc lập</td><td>Máy tính thường (CPU 12 nhân) — 3 đêm × ~230 phút, <strong>0 đồng</strong></td></tr>
<tr><td>Demo sản phẩm</td><td>Chạy thật trên CPU, có ảnh chụp màn hình</td></tr>
</table>
</div>
<ul>
<li><strong>Tự phục hồi</strong>: rớt mạng giữa chừng → chạy lại là tiếp tục từ chỗ dở, không mất giờ GPU.</li>
<li><strong>Không số nào gõ tay</strong>: <span class="mono">122 con số</span> trong báo cáo được
một script đối chiếu tự động với file kết quả gốc — lệch là báo lỗi.</li>
<li><strong>Tái chạy được</strong>: toàn bộ phân tích dựng lại từ dự đoán đã lưu bằng 2 lệnh, không cần GPU.</li>
</ul>
</div>
<p class="takeaway">Đây không phải kết quả "trên giấy": <strong>mọi bảng, mọi hình trong bài đều sinh ra
từ code đã chạy hết, kèm nhật ký</strong> — người khác tải repo về là kiểm tra lại được.</p>"""))

# ============ HOI 2 — DO CHO DUNG ============ #
S.append(slide(6, T, "Hồi 2 · Đo cho đúng", f"""
<h2>Kẻ thù đầu tiên không phải mô hình — là nhiễu của phép đo</h2>
<div class="cols c38">
<div>
<ul>
<li>Chạy lại <em>y hệt</em> một thí nghiệm, điểm đã lệch <strong class="mono">±0,02–0,05</strong>.
Mức lệch này lớn hơn tác dụng của hầu hết "cải tiến". Tin một lần chạy đơn lẻ là tự lừa mình.</li>
<li>Đổi card đồ hoạ (GPU), giữ nguyên code và dữ liệu — vẫn <strong>ra một mô hình khác</strong>.
Vì vậy không trộn kết quả từ hai loại máy vào một phép so.</li>
</ul>
<p class="takeaway"><strong>4 kỷ luật cho mọi con số</strong>: ① chạy 3 lần, lấy trung bình ±σ ·
② kèm khoảng tin cậy 95% · ③ một cách chọn checkpoint duy nhất cho tất cả ·
④ lưu mọi dự đoán để phân tích lại không tốn GPU.</p>
</div>
<figure><img src="{IMG_EDA}" alt="Phan bo 22 nhom"/>
<figcaption>Số ảnh mỗi nhóm (cột phải thấp dần): chênh nhau tới <b>50,6 lần</b>.
Kiểm tra trùng lặp: chỉ 0,38% ảnh thi bị nghi "lọt đề"; bỏ chúng đi, kết quả gần như
không đổi (−0,0016) — kết luận không đến từ rò rỉ.</figcaption></figure>
</div>
<p class="gloss"><b>seed</b>: một lần chạy lặp độc lập (đổi cách xáo trộn ngẫu nhiên) · <b>σ (sigma)</b>: độ dao động giữa các lần chạy · <b>khoảng tin cậy</b>: vùng mà điểm thật nhiều khả năng nằm trong.</p>"""))

S.append(slide(7, T, "Hồi 2 · Đo cho đúng", f"""
<h2>Nhiễu nằm ở nhóm nào — nhìn tận mắt bằng ảnh thật</h2>
<div class="cols c2">
<div>
<table>
<tr><th>Nhóm "nóng"</th><th class="num">Ảnh thi</th><th class="num">F1</th><th>Hay nhầm với</th></tr>
<tr><td>Viêm niêm mạc đại tràng</td><td class="num">6</td><td class="num neg">0,286</td><td>niêm mạc thường</td></tr>
<tr><td>Túi thừa đại tràng</td><td class="num">6</td><td class="num muted">dao động mạnh</td><td>niêm mạc thường</td></tr>
<tr><td>Manh tràng</td><td class="num">23</td><td class="num neg">0,364</td><td>niêm mạc thường</td></tr>
<tr><td>Polyp đại tràng</td><td class="num">164</td><td class="num">khá</td><td>ung thư đại trực tràng</td></tr>
</table>
<p class="note" style="margin-top:10px">Nhóm chỉ có 6 ảnh thi: đoán sai thêm 1 ảnh là điểm nhóm
rơi ~0,15. Điểm trung bình 22 nhóm lắc ~0,007 chỉ vì một tấm ảnh. Đó là gốc rễ cơ học của
nhiễu ±0,02–0,05.</p>
</div>
<div>
<p style="font-weight:600;margin-bottom:6px">Bằng chứng 1 — hai nhóm khác nhau, nhìn gần như một:</p>
<div class="pair">
<figure><img src="{IMG_CECUM}" alt="Manh trang"/><figcaption>nhãn: <b>Manh tràng</b></figcaption></figure>
<figure><img src="{IMG_NORMAL_LB}" alt="Niem mac binh thuong"/><figcaption>nhãn: <b>Niêm mạc bình thường</b></figcaption></figure>
</div>
<p style="font-weight:600;margin:12px 0 6px">Bằng chứng 2 — <span class="neg">cùng một khung hình, hai nhãn khác nhau</span> (độ giống 1,0000, từ bảng audit):</p>
<div class="pair">
<figure><img src="{IMG_MIS_A}" alt="Nhan Viem thuc quan"/><figcaption>nhãn: <b>Viêm thực quản</b></figcaption></figure>
<figure><img src="{IMG_MIS_B}" alt="Nhan Thuc quan binh thuong"/><figcaption>nhãn: <b>Thực quản bình thường</b></figcaption></figure>
</div>
<p class="note" style="margin-top:10px">Không mô hình nào đúng được cả hai nhãn. Nhãn nghi sai
tự đặt <strong>trần</strong> cho độ chính xác. Hồi 5 bàn hướng xử lý.</p>
</div>
</div>"""))

S.append(slide(8, T, "Hồi 2 · Đo cho đúng", """
<h2>Bước đầu tiên phải đạt: làm lại được con số của bài báo</h2>
<div class="statrow">
<div class="stat"><span class="v plain small">0,6686 ± 0,0234</span><span class="k">mốc cũ làm lại (DenseNet-121, đúng cách chấm của bài báo, 3 lần chạy)</span></div>
<div class="stat"><span class="v plain small">+0,78 σ</span><span class="k">chênh +0,0182 so với 0,6504 — trong vùng dao động → <strong>tái lập thành công</strong></span></div>
<div class="stat"><span class="v plain small">0,6813</span><span class="k">Swin-T (S0) — baseline thứ hai nhóm tự thêm, bài báo không có</span></div>
</div>
<ul>
<li>Đạt được chỉ với <strong>30 vòng học thay vì 150</strong> của bài báo. Mô hình này đạt đỉnh sớm;
học thêm không giúp.</li>
<li>Thí nghiệm học chuyển giao: <strong>mở toàn bộ mạng cho học lại</strong> thắng mọi mức "đóng băng"
(chỉ học lớp cuối: 0,5674 → mở hết: 0,6686). Ảnh nội soi khác ảnh đời thường đủ nhiều để phải học lại sâu.</li>
</ul>
<p class="takeaway">Đã có mốc đáng tin. Giờ mới được phép hỏi: <strong>cái gì nâng được điểm?</strong> (Hồi 3)</p>"""))

# ============ HOI 3 — CAI THIEN BANG GI ============ #
S.append(slide(9, T, "Hồi 3 · Cải thiện bằng gì", """
<h2>Nói điều dở trước: bốn hướng "hiển nhiên" đều KHÔNG ăn thua</h2>
<table>
<tr><th>Hướng thử (ai cũng nghĩ tới trước)</th><th class="num">Tác dụng đo được</th><th>Kết luận</th></tr>
<tr><td>Đổi sang kiến trúc "xịn" hơn (một mình)</td><td class="num">+0,0034</td><td>≈ 0 — trong vùng nhiễu</td></tr>
<tr><td>Ảnh nét hơn 224 → 288px (tốn 1,7× tính toán)</td><td class="num">+0,0041</td><td>≈ 0; với công thức mới còn <span class="neg">−0,0006</span></td></tr>
<tr><td>3 kỹ thuật "cân bằng nhóm hiếm" lúc huấn luyện</td><td class="num">≤ 0</td><td>cả ba đều phẳng hoặc âm</td></tr>
<tr><td>Gộp nhiều kiến trúc lại (ensemble)</td><td class="num">0,7130</td><td><span class="neg">thua</span> mô hình đơn tốt nhất</td></tr>
</table>
<p class="takeaway">Ngưỡng để một thay đổi được coi là "thật": phải vượt vùng nhiễu <strong>±0,035</strong>.
Không hướng nào ở trên vượt — việc <strong>loại trừ có hệ thống</strong> này dồn nghi ngờ về đúng một chỗ:
<em>cách huấn luyện</em>. Slide sau.</p>
<p class="gloss"><b>ensemble</b>: lấy trung bình dự đoán của nhiều mô hình — thường được kỳ vọng tăng điểm, ở đây thì không.</p>"""))

S.append(slide(10, T, "Hồi 3 · Cải thiện bằng gì", """
<h2>Thay đổi ăn tiền nhất: dạy mô hình học KỸ thay vì học NHANH</h2>
<div class="cols c2">
<div>
<table>
<tr><th>Thành phần của "công thức mới"</th><th>Nôm na là gì</th></tr>
<tr><td>Học dài hơi — 80 vòng</td><td>đủ thời gian ngấm, thay vì 15–30 vòng</td></tr>
<tr><td>Khởi động chậm, hạ nhiệt dần</td><td>đầu nhẹ nhàng, cuối tinh chỉnh (warmup–cosine)</td></tr>
<tr><td>Tầng sâu học nhanh, tầng gốc học chậm</td><td>giữ kiến thức nền, chỉ tinh chỉnh phần trên (LLRD)</td></tr>
<tr><td>Lấy trung bình theo thời gian</td><td>bản "điềm tĩnh" của mô hình, đỡ nhiễu (EMA)</td></tr>
<tr><td>Trộn hai ảnh khi học</td><td>bắt mô hình học ranh giới mềm giữa các nhóm (mixup)</td></tr>
</table>
</div>
<div>
<div class="stat"><span class="v">+0,0443</span><span class="k">mức tăng của riêng công thức này — thay đổi <strong>duy nhất</strong> vượt ngưỡng nhiễu ±0,035 trong toàn bộ dự án</span></div>
<p class="note" style="margin-top:14px">Cộng thêm hai chỉnh sửa <strong>không tốn GPU</strong> ở khâu chấm.
<b>Gộp 3 checkpoint tốt nhất</b>: +0,0094, bớt phụ thuộc vận may của một thời điểm lưu.
<b>Hiệu chỉnh theo tần suất nhóm</b>: +0,0143, bù thiên vị nghiêng về nhóm đông ảnh.</p>
</div>
</div>
<p class="gloss"><b>checkpoint</b>: bản lưu mô hình tại một thời điểm học · các tên trong ngoặc (warmup, LLRD, EMA, mixup) là thuật ngữ gốc để tra cứu.</p>"""))

S.append(slide(11, T, "Hồi 3 · Cải thiện bằng gì", """
<h2>Nhưng công thức chỉ ăn tiền khi ghép ĐÚNG kiến trúc</h2>
<div class="cols c2">
<div>
<table>
<tr><th>Cùng một công thức mới, áp lên…</th><th class="num">Tác dụng</th></tr>
<tr><td>…kiến trúc cũ (DenseNet-121)</td><td class="num neg">−0,0110 · tệ đi cả 3/3 lần chạy</td></tr>
<tr class="hl"><td>…kiến trúc hybrid (CoAtNet-0)</td><td class="num">+0,0468 ✅</td></tr>
</table>
<p class="note" style="margin-top:12px">Tách riêng từng thứ trên cùng một lần chạy: đổi kiến trúc
một mình <span class="neg">−0,0160</span>, đổi công thức một mình cũng âm — <strong>ghép lại mới
+0,0294</strong> (kiểm định thống kê p = 0,0018).</p>
</div>
<ul>
<li>Đây là <strong>một cặp đôi</strong>, không phải phép cộng hai thứ tốt. Công thức "học kỹ"
cần phần attention đủ dẻo để phát huy. Kiến trúc cũ bị nó làm quá tay.</li>
<li>Phát hiện này <strong>ngược với kỳ vọng ban đầu</strong> ("cứ đổi mô hình xịn là hơn").
Đây là đóng góp chính của nhóm.</li>
<li>Hệ quả thực dụng: bản <em>nên triển khai</em> là hybrid @224 — nhanh hơn 1,7×, điểm tương đương.</li>
</ul>
</div>
<p class="takeaway"><strong>Cách cải thiện độ chính xác, gói trong một câu:</strong> giữ nguyên dữ liệu;
ghép <strong>công thức học kỹ</strong> với <strong>kiến trúc hybrid</strong>; thu nốt phần tăng
miễn phí từ <strong>cách chấm</strong>.</p>"""))

S.append(slide(12, T, "Hồi 3 · Cải thiện bằng gì", f"""
<h2>Cộng dồn lại: 0,6686 → 0,7441 — từng bậc đã được đo riêng</h2>
{waterfall_svg()}
<p class="note" style="margin-top:4px">Hai cột xám (+0,0034 · +0,0041) là hai hướng "hiển nhiên" nhưng
gần bằng 0. Cột đậm nhất là công thức huấn luyện. Hai cột xanh nhạt không tốn GPU.
Vạch đỏ 0,6504 — mốc phải vượt — bị bỏ lại ngay từ bậc đầu tiên.</p>"""))

# ============ HOI 4 — KIEM CHUNG & SO SANH ============ #
S.append(slide(13, T, "Hồi 4 · Kiểm chứng & So sánh", f"""
<h2>Bằng chứng thống kê: cả khoảng tin cậy nằm trên mốc cũ</h2>
{ci_svg()}
<p class="note">Cách đọc hình: chấm tròn là điểm ước lượng; thanh ngang là vùng điểm thật nhiều khả năng
nằm trong. Hai thanh của hệ thống đề xuất <strong>không chạm</strong> vạch đỏ 0,6504 — mức vượt không
thể chỉ là may mắn. Accuracy cũng vượt: <span class="mono">0,850</span> so với 0,8203.</p>"""))

S.append(slide(14, T, "Hồi 4 · Kiểm chứng & So sánh", """
<h2>Đặt cạnh các công bố khác trên cùng bộ dữ liệu</h2>
<table>
<tr><th>Công bố</th><th>Cách chia dữ liệu · thước đo</th><th class="num">Kết quả</th><th>So với nhóm</th></tr>
<tr><td>Bài báo gốc GastroVision (2023)</td><td>60:20:20 · macro-F1</td><td class="num">0,6504</td><td>✅ cùng luật chơi — nhóm đạt <strong>0,7441</strong></td></tr>
<tr><td>CNN-Transformer lai (arXiv 2408.10733, 2024)</td><td><strong>80:20, không tập val</strong> · F1 nhiều khả năng weighted</td><td class="num">acc 0,8386</td><td>⚠️ chỉ so được accuracy: nhóm <strong>0,850</strong> — cao hơn dù dùng ít dữ liệu học hơn</td></tr>
<tr><td>CNN + Explainable AI (BSPC 2024)</td><td>tách riêng tiêu hoá trên / dưới (2 bài toán con dễ hơn)</td><td class="num">macro-F1 0,664 · 0,681</td><td>⚠️ xác nhận vùng 0,65–0,68 là mức của DenseNet-based</td></tr>
<tr><td>GastroViT — ensemble ViT (arXiv 2509.26502, 2025)</td><td>dataset chị em HyperKvasir, 23 nhóm</td><td class="num">F1 0,64 · acc 0,9198</td><td>⚠️ khác dataset. Nhưng minh hoạ rõ: accuracy 92% mà F1 chỉ 0,64 — <strong>F1 công bằng thấp là bản chất bài toán GI nhiều nhóm lệch</strong></td></tr>
<tr><td>Hướng pretrain chuyên ngành (EndoExtend24, GastroNet-5M)</td><td>không công bố số trên split này</td><td class="num">—</td><td>❌ không so được — chính là hướng phát triển nhóm đề xuất</td></tr>
</table>
<p class="takeaway">Trong phạm vi khảo sát: <strong>chưa công bố nào đạt macro-F1 trên 0,6504 ở đúng
luật chơi 22 nhóm / 60:20:20</strong>. Vì vậy 0,7441 là con số so-được tốt nhất nhóm biết.
Mỗi khác biệt luật chơi đều ghi rõ, không so bừa.</p>
<p class="gloss"><b>weighted F1</b>: trung bình có trọng số theo cỡ nhóm — nhóm đông ảnh lấn át nên số thường cao hơn macro-F1, không so trực tiếp được.</p>"""))

S.append(slide(15, T, "Hồi 4 · Kiểm chứng & So sánh", f"""
<h2>Điểm tăng rơi đúng chỗ khó nhất: các nhóm hiếm</h2>
<div class="cols c38">
<figure><img src="{IMG_PERCLASS}" alt="Diem tung nhom va ma tran nham lan"/>
<figcaption>Trái: điểm F1 từng nhóm. Phải: ma trận nhầm lẫn — ô sáng ngoài đường chéo là các cặp hay nhầm.</figcaption></figure>
<ul>
<li><strong class="mono">90,4%</strong> phần điểm tăng (so với bảng chi tiết của bài báo) nằm ở
<strong>các nhóm hiếm</strong> — nơi mô hình cũ yếu nhất.</li>
<li>Nhóm tăng mạnh nhất: <em>polyp đã cắt</em> <span class="accent">+0,313</span>.</li>
<li>Phần điểm còn thiếu: <span class="mono">85,2%</span> vẫn nằm ở 15 nhóm hiếm — muốn tiến tiếp
phải <strong>thêm dữ liệu</strong>, không phải thêm mô hình (Hồi 5).</li>
</ul>
</div>"""))

S.append(slide(16, T, "Hồi 4 · Kiểm chứng & So sánh", """
<h2>Kiểm chứng độc lập: lặp lại toàn bộ trên máy tính thường</h2>
<div class="cols c2">
<div>
<table>
<tr><th>DenseNet-121, cùng cách chấm</th><th class="num">Điểm</th></tr>
<tr><td>Trên GPU T4 (3 lần chạy)</td><td class="num">0,6780 ± 0,0073</td></tr>
<tr><td>Trên CPU 12 nhân (3 lần chạy)</td><td class="num">0,6919 ± 0,0060</td></tr>
</table>
<p class="note" style="margin-top:12px">Hai bên dùng luật chơi khác nhau (số vòng học, cỡ mẻ) nên
không so điểm trực tiếp. Nhưng <strong>hai phát hiện phương pháp lặp lại nguyên vẹn</strong>:
"gộp 3 checkpoint" vẫn ổn định nhất; hiệu chỉnh tần suất vẫn mất tác dụng khi thiếu công thức mới.</p>
</div>
<div>
<div class="stat"><span class="v">0,7059</span>
<span class="k">hệ thống bản CPU (gộp 3 lần chạy) — khoảng tin cậy [0,6560; 0,7447]
<strong>không chứa 0,6504</strong> · chi phí <strong>0 đồng</strong>, 3 đêm máy bàn</span></div>
<p class="takeaway" style="margin-top:14px">Kể cả <strong>không có GPU</strong>, quy trình này vẫn vượt
mốc đã công bố theo đúng chuẩn bằng chứng — phương pháp đứng vững trên phần cứng thứ ba.</p>
</div>
</div>"""))

S.append(slide(17, T, "Hồi 4 · Kiểm chứng & So sánh", f"""
<h2>Sản phẩm chạy thật: web demo + tốc độ đủ dùng lâm sàng</h2>
<div class="cols c38">
<figure><img src="{IMG_DEMO}" alt="Demo web chay that"/>
<figcaption>Demo web (Gradio) chạy thật trên CPU: tải ảnh lên → trả 5 khả năng kèm xác suất.
Ảnh chụp từ phiên chạy tự động: đoán đúng "Dụng cụ can thiệp" với 96%.</figcaption></figure>
<ul>
<li>Xuất mô hình chuẩn ONNX; trên GPU T4: bản nên triển khai xử lý
<span class="mono">11,7 ms/ảnh</span> — ~85 khung hình/giây.</li>
<li>Máy không GPU: bản gọn MobileNetV3 <span class="mono">21,7 ms/ảnh</span> (16,9 MB) — chạy nổi trên máy phòng khám.</li>
<li>Minh bạch: demo dùng 1 bản lưu mô hình (điểm ~0,7199) — <strong>thấp hơn</strong> hệ thống đầy đủ
được báo cáo (0,7441); hai con số không được lẫn.</li>
</ul>
</div>"""))

# ============ HOI 5 — NHIN THANG ============ #
S.append(slide(18, T, "Hồi 5 · Nhìn thẳng", """
<h2>Dữ liệu còn lỗi ở đâu — và xử lý tiếp thế nào</h2>
<table>
<tr><th>Vấn đề còn lại của dữ liệu</th><th>Ảnh hưởng đã đánh giá</th><th>Hướng xử lý đề xuất</th></tr>
<tr><td>9 cặp ảnh "lọt" giữa các tập (và đó là <em>chặn dưới</em> — không có mã bệnh nhân để soát hết)</td><td>đã đo: bỏ đi chỉ đổi −0,0016 → kết luận an toàn</td><td>xin tác giả bổ sung mã bệnh nhân/ca soi; chia theo ca thay vì theo ảnh</td></tr>
<tr><td>Cặp ảnh gần giống hệt nhưng mang 2 nhãn khác nhau (nhãn nghi sai)</td><td>đặt <strong>trần</strong> độ chính xác — không mô hình nào đúng cả hai</td><td>nhờ bác sĩ nội soi rà lại các cặp đã khoanh vùng</td></tr>
<tr><td>2 nhóm chỉ có 6 ảnh thi; 15 nhóm hiếm giữ 85,2% phần điểm còn thiếu</td><td>nguồn nhiễu chính; giới hạn là <em>dữ liệu</em> chứ không phải mô hình</td><td>bổ sung ảnh nhóm hiếm: học trước trên kho nội soi lớn hơn (HyperKvasir — kèm kiểm rò rỉ chéo bắt buộc); đầu phân loại kiểu cosine cho nhóm hiếm</td></tr>
</table>
<p class="note">Nguyên tắc giữ nguyên: mọi hướng trên phải qua đúng bộ kỷ luật đo ở Hồi 2.
Con số trông đẹp đến đâu cũng chưa được tin trước khi qua cửa đó. Bằng chứng ảnh của
"nhãn nghi sai": xem slide 07.</p>"""))

S.append(slide(19, T, "Hồi 5 · Nhìn thẳng", """
<h2>Hạn chế của giải pháp — và ba điều mang về</h2>
<div class="cols c2">
<div>
<p><strong>Hạn chế (của giải pháp, tách khỏi lỗi dữ liệu):</strong></p>
<ul>
<li>Công thức "học kỹ" <strong>đắt</strong>: 80 vòng học, ~118 phút/lượt so với ~21 của mốc cũ.</li>
<li>Và <strong>không phổ quát</strong> — chỉ ăn tiền với kiến trúc hybrid; áp bừa lên CNN cũ là tệ đi.</li>
<li>Một đối chứng (@224) mới chạy 1 lần; điểm số phụ thuộc loại GPU; demo yếu hơn hệ thống đầy đủ.</li>
</ul>
</div>
<div>
<p><strong>Ba điều mang về:</strong></p>
<ul>
<li><strong>1.</strong> Làm lại được mốc công bố, rồi vượt nó <strong>bằng bằng chứng</strong>:
0,7441 ± 0,0088, khoảng tin cậy không chứa 0,6504 — kiểm chứng thêm trên máy thường.</li>
<li><strong>2.</strong> Điểm tăng đến từ <strong>cặp đôi công thức × kiến trúc</strong> + cách chấm —
không phải từ "đổi mô hình xịn".</li>
<li><strong>3.</strong> <strong>Trung thực là một tính năng</strong>: kết quả xấu công bố đủ,
122 con số máy tự đối chiếu, ai cũng tái chạy được.</li>
</ul>
</div>
</div>
<p class="takeaway" style="text-align:center"><strong>0,6504 → 0,7441</strong> — và một quy trình mà
người sau có thể tin, kiểm và xây tiếp.</p>"""))

HTML = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vượt baseline GastroVision</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;650;700&family=Be+Vietnam+Pro:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>{CSS}</style>
</head>
<body>
<div id="stage">
{chr(10).join(S)}
</div>
<div id="hint"><kbd>←</kbd><kbd>→</kbd> chuyển slide · <kbd>G</kbd> lưới tổng quan · Ctrl+P in</div>
<script>{JS}</script>
</body>
</html>"""

with io.open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML)
print(f"da ghi {OUT} ({os.path.getsize(OUT) // 1024} KB, {T} slide)")
