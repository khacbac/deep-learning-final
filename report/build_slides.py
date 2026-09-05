# -*- coding: utf-8 -*-
"""Sinh bo slide bao ve -> report/slides.html.

Cau truc KHONG theo thu tu tai lieu ma theo MOT truc truyen duy nhat:
  "0,6504 -> 0,7441 den tu dau?"  (cach thuc hien -> ket qua)
Diem nhan trung tam: slide THAC NUOC phan ra tung bac tang diem — hai lever
tra tien that (cong thuc hien dai x hybrid, cach do) duoc to dam; hai lever
~0 (kien truc, do phan giai) hien thi xam va noi thang.

Moi con so tren slide deu co trong BAO_CAO.md va do check_numbers.py doi chieu —
KHONG dua so moi vao day ma khong them vao CHECKS.

Chay:  python report/build_slides.py   ->  report/slides.html (tu chua, anh data-URI)
Dieu huong: <- -> / Space / click | phim G = luoi tong quan | in duoc (Ctrl+P).
"""
import base64
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "slides.html")


def b64(rel):
    with open(os.path.join(HERE, rel), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


IMG_EDA = b64("figures/06_eda.png")
IMG_PERCLASS = b64("figures/18_per_class_va_confusion.png")
IMG_DEMO = b64("demo/29b_demo_gradio_cpu.png")

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
  padding:44px 64px 34px;position:absolute;transform-origin:center center}
.slide.active{display:flex}
.slide header{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px}
.eyebrow{font-family:'IBM Plex Mono',monospace;font-size:13px;letter-spacing:.18em;color:var(--teal);text-transform:uppercase}
.pageno{font-family:'IBM Plex Mono',monospace;font-size:13px;color:var(--muted)}
h1{font-family:'Archivo',sans-serif;font-weight:700;font-size:56px;line-height:1.08;text-wrap:balance}
h2{font-family:'Archivo',sans-serif;font-weight:650;font-size:38px;line-height:1.14;margin-bottom:16px;text-wrap:balance}
.slide footer{margin-top:auto;padding-top:10px;border-top:1px solid var(--line);
  display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--muted)}
.body{flex:1;min-height:0;display:flex;flex-direction:column}
p,li{font-size:20px;line-height:1.5}
.lead{font-size:23px;line-height:1.5;max-width:64ch}
ul{padding-left:24px;display:flex;flex-direction:column;gap:10px}
.muted{color:var(--muted)}
.accent{color:var(--teal);font-weight:600}
.neg{color:var(--red);font-weight:600}
.mono,td.num,.num{font-family:'IBM Plex Mono',monospace;font-variant-numeric:tabular-nums}
code{font-family:'IBM Plex Mono',monospace;font-size:.92em;background:var(--teal-soft);padding:1px 6px;border-radius:4px}
.cols{display:grid;gap:36px;flex:1;min-height:0;align-items:start}
.cols.c2{grid-template-columns:1fr 1fr}
.cols.c38{grid-template-columns:5fr 4fr}
table{border-collapse:collapse;width:100%}
th{font-family:'IBM Plex Mono',monospace;font-size:14px;letter-spacing:.06em;text-transform:uppercase;
  color:var(--muted);text-align:left;padding:8px 14px;border-bottom:2px solid var(--ink);font-weight:500}
td{padding:9px 14px;border-bottom:1px solid var(--line);font-size:19px}
td.num,th.num{text-align:right}
tr.hl td{background:var(--teal-soft);font-weight:600}
.stat{display:flex;flex-direction:column;gap:2px}
.stat .v{font-family:'IBM Plex Mono',monospace;font-size:62px;font-weight:600;color:var(--teal);line-height:1.05}
.stat .v.plain{color:var(--ink)}
.stat .v.small{font-size:44px}
.stat .k{font-size:17px;color:var(--muted);max-width:34ch}
.statrow{display:flex;gap:56px;margin:24px 0}
figure{margin:0;display:flex;flex-direction:column;gap:8px;min-height:0}
figure img{max-width:100%;max-height:100%;object-fit:contain;border:1px solid var(--line);background:#fff}
figcaption{font-size:15px;color:var(--muted)}
.note{font-size:16px;color:var(--muted);border-left:3px solid var(--teal);padding:6px 0 6px 14px;max-width:74ch}
.takeaway{font-size:21px;line-height:1.45;background:var(--teal-soft);border-left:4px solid var(--teal);
  padding:12px 18px;margin-top:14px;max-width:100%}
.takeaway strong{color:var(--teal)}
.titleslide{justify-content:center;gap:20px}
.titleslide .who{font-family:'IBM Plex Mono',monospace;font-size:16px;color:var(--muted);letter-spacing:.06em}
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


# ------------- SVG 1: THAC NUOC — phan ra 0,6686 -> 0,7441 (diem nhan) ------------- #
def waterfall_svg():
    # (nhan 2 dong, delta, kieu)  |  start/end mang gia tri tuyet doi
    steps = [
        ("Tái lập B0", "DenseNet-121, best", None, 0.6686, "start"),
        ("Đổi kiến trúc", "→ CoAtNet-0", 0.0034, None, "flat"),
        ("Độ phân giải", "224 → 288", 0.0041, None, "flat"),
        ("Công thức hiện đại", "80ep·LLRD·EMA·mixup", 0.0443, None, "big"),
        ("Quy tắc top3", "0 epoch", 0.0094, None, "meas"),
        ("Hiệu chỉnh logit", "0 epoch", 0.0143, None, "meas"),
        ("P2 — hệ thống", "đề xuất", None, 0.7441, "end"),
    ]
    v0, v1 = 0.635, 0.760
    top, ph = 34, 300

    def Y(v):
        return top + (v1 - v) / (v1 - v0) * ph

    colw, gap, x0 = 118, 34, 60
    color = {"start": "#56635E", "end": "#1F6E5E", "big": "#1F6E5E",
             "meas": "#4E8F80", "flat": "#C9D3CE"}
    p = ['<svg viewBox="0 0 1130 430" role="img" aria-label="Phan ra muc tang tu 0,6686 len 0,7441">']
    # moc truc + luoi
    for t in (0.65, 0.70, 0.75):
        p.append(f'<line x1="42" y1="{Y(t):.0f}" x2="1110" y2="{Y(t):.0f}" stroke="#D7DEDA" stroke-width="1"/>')
        p.append(f'<text x="36" y="{Y(t) + 5:.0f}" font-size="14" fill="#56635E" text-anchor="end" '
                 f'font-family="IBM Plex Mono,monospace">{vn(t, 2)}</text>')
    # baseline cong bo
    p.append(f'<line x1="42" y1="{Y(0.6504):.0f}" x2="1110" y2="{Y(0.6504):.0f}" '
             f'stroke="#A8402F" stroke-width="2" stroke-dasharray="7 5"/>')
    p.append(f'<text x="52" y="{Y(0.6504) + 20:.0f}" font-size="14" fill="#A8402F" text-anchor="start" '
             f'font-weight="600" font-family="IBM Plex Mono,monospace">baseline công bố 0,6504</text>')

    cum = 0.6686
    prev_top = None
    for i, (name, sub, d, absv, kind) in enumerate(steps):
        x = x0 + i * (colw + gap)
        c = color[kind]
        if kind in ("start", "end"):
            val = absv
            y_hi, y_lo = Y(val), Y(v0)
            p.append(f'<rect x="{x}" y="{y_hi:.0f}" width="{colw}" height="{y_lo - y_hi:.0f}" '
                     f'fill="{c}" rx="3"/>')
            p.append(f'<text x="{x + colw / 2:.0f}" y="{y_hi - 10:.0f}" font-size="19" font-weight="700" '
                     f'fill="{"#1F6E5E" if kind == "end" else "#1C2321"}" text-anchor="middle" '
                     f'font-family="IBM Plex Mono,monospace">{vn(val)}</text>')
            cum = val
        else:
            lo, hi = cum, cum + d
            y_hi, y_lo = Y(hi), Y(lo)
            h = max(y_lo - y_hi, 4)
            p.append(f'<rect x="{x}" y="{y_hi:.0f}" width="{colw}" height="{h:.0f}" fill="{c}" rx="3"/>')
            lbl_fill = "#56635E" if kind == "flat" else "#1F6E5E"
            p.append(f'<text x="{x + colw / 2:.0f}" y="{y_hi - 8:.0f}" font-size="16" font-weight="600" '
                     f'fill="{lbl_fill}" text-anchor="middle" '
                     f'font-family="IBM Plex Mono,monospace">+{vn(d)}</text>')
            cum = hi
        # duong noi
        if prev_top is not None:
            p.append(f'<line x1="{x - gap}" y1="{prev_top:.0f}" x2="{x}" y2="{Y(cum):.0f}" '
                     f'stroke="#9FB0AA" stroke-width="1" stroke-dasharray="3 3"/>')
        prev_top = Y(cum)
        # nhan cot
        p.append(f'<text x="{x + colw / 2:.0f}" y="{top + ph + 26}" font-size="15" fill="#1C2321" '
                 f'text-anchor="middle" font-family="Be Vietnam Pro,sans-serif" font-weight="600">{name}</text>')
        p.append(f'<text x="{x + colw / 2:.0f}" y="{top + ph + 46}" font-size="13" fill="#56635E" '
                 f'text-anchor="middle" font-family="Be Vietnam Pro,sans-serif">{sub}</text>')
    p.append("</svg>")
    return "".join(p)


# ------------- SVG 2: khoang tin cay tren MOT truc ------------- #
def ci_svg():
    x0, x1, w = 0.62, 0.80, 980

    def X(v):
        return 90 + (v - x0) / (x1 - x0) * w

    rows = [
        ("B0 · DenseNet-121 (tái lập)", 0.6686, 0.6452, 0.6920, "#56635E", "±σ, 3 seed"),
        ("P2 · hệ thống đề xuất", 0.7441, 0.6986, 0.7736, "#1F6E5E", "CI 95% bootstrap"),
        ("P2 · ensemble 3 seed", 0.7587, 0.7110, 0.7924, "#4E8F80", "CI 95% bootstrap"),
    ]
    p = ['<svg viewBox="0 0 1120 320" role="img" font-family="IBM Plex Mono,monospace" '
         'aria-label="Khoang tin cay so voi baseline 0,6504">']
    for t in (0.65, 0.70, 0.75):
        p.append(f'<line x1="{X(t):.0f}" y1="36" x2="{X(t):.0f}" y2="268" stroke="#D7DEDA" stroke-width="1"/>')
        p.append(f'<text x="{X(t):.0f}" y="292" font-size="15" fill="#56635E" text-anchor="middle">{vn(t, 2)}</text>')
    bx = X(0.6504)
    p.append(f'<line x1="{bx:.0f}" y1="24" x2="{bx:.0f}" y2="268" stroke="#A8402F" stroke-width="2" stroke-dasharray="6 5"/>')
    p.append(f'<text x="{bx:.0f}" y="16" font-size="15" fill="#A8402F" text-anchor="middle" font-weight="600">baseline công bố 0,6504</text>')
    y = 76
    for name, mid, lo, hi, color, note in rows:
        p.append(f'<line x1="{X(lo):.0f}" y1="{y}" x2="{X(hi):.0f}" y2="{y}" stroke="{color}" stroke-width="6" stroke-linecap="round"/>')
        p.append(f'<circle cx="{X(mid):.0f}" cy="{y}" r="10" fill="{color}" stroke="#F6F8F7" stroke-width="2"/>')
        p.append(f'<text x="{X(lo):.0f}" y="{y - 20}" font-size="17" fill="#1C2321" font-family="Be Vietnam Pro,sans-serif" font-weight="600">{name}</text>')
        p.append(f'<text x="{X(lo):.0f}" y="{y + 28}" font-size="15" fill="#56635E">{vn(mid)} · [{vn(lo)}; {vn(hi)}] {note}</text>')
        y += 88
    p.append("</svg>")
    return "".join(p)


def slide(no, total, section, body, cls=""):
    return f"""<section class="slide {cls}" id="s{no}">
<header><span class="eyebrow">{section}</span><span class="pageno">{no:02d} / {total}</span></header>
<div class="body">{body}</div>
<footer><span>GastroVision · AIN501 Deep Learning</span><span>05-09-2026</span></footer>
</section>"""


T = 11
S = []

# 1 — LUAN DE: ket qua truoc, ly do sau
S.append(slide(1, T, "AIN501 · Final Project", """
<div class="body titleslide">
<h1>Vượt baseline công bố trên GastroVision</h1>
<div class="statrow">
<div class="stat"><span class="v plain">0,6504</span><span class="k">baseline công bố — DenseNet-121, arXiv 2307.08140 (1 lần chạy, không sai số)</span></div>
<div class="stat"><span class="v">0,7441 ± 0,0088</span><span class="k">hệ thống đề xuất P2, 3 seed — CI 95% [0,6986; 0,7736] <strong>không chứa baseline</strong></span></div>
</div>
<p class="lead">Điểm không đến từ một kiến trúc mới. Nó đến từ <strong>công thức huấn luyện × kiến trúc</strong>
(+0,0443) và <strong>cách đo</strong> (+0,0237, không tốn epoch nào) — cả hai đều được phân rã, đo riêng
và kiểm bằng khoảng tin cậy.</p>
<p class="who">Nhóm 2 thành viên · khacbac + quang3447 · 122 con số của báo cáo được đối chiếu máy với nguồn</p>
</div>""", cls="titleslide"))

# 2 — BAI TOAN & LUAT CHOI
S.append(slide(2, T, "Bài toán", """
<h2>Một con số phải vượt — và tiêu chuẩn tự đặt để vượt nó tử tế</h2>
<div class="cols c2">
<ul>
<li><strong>GastroVision</strong>: 8.000 ảnh nội soi tiêu hoá, 27 lớp → lọc theo luật bài báo
(&gt; 25 ảnh) → <span class="mono">22 lớp / 7.930 ảnh</span>.</li>
<li>Chia phân tầng 60:20:20, <code>SPLIT_SEED = 42</code> — <strong>kiểm chứng bằng chính Table 3
của bài báo</strong>: 16/22 lớp khớp chính xác, tổng test trùng khít 1.586 ảnh.</li>
<li>Baseline mạnh nhất họ công bố: DenseNet-121, macro-F1 <strong>0,6504</strong>.</li>
</ul>
<ul>
<li>Metric chính: <strong>macro-F1</strong> — bắt buộc khi dữ liệu mất cân bằng.</li>
<li>Hai baseline của nhóm: <code>B0</code> DenseNet-121 (khớp số bài báo) và <code>S0</code> Swin-T
(nhánh Transformer bài báo không có).</li>
</ul>
</div>
<p class="takeaway">Tiêu chuẩn tự đặt: <strong>"vượt baseline" chỉ được tuyên bố khi khoảng tin cậy 95%
không chứa 0,6504, đo trên ≥ 3 seed</strong> — không bao giờ bằng phép trừ hai điểm ước lượng.</p>"""))

# 3 — THACH THUC -> KY LUAT DO (cach thuc hien, nen tang)
S.append(slide(3, T, "Cách thực hiện · Nền móng", f"""
<h2>Nhiễu lớn hơn hiệu ứng — nên phải sửa phép đo trước khi sửa mô hình</h2>
<div class="cols c38">
<div>
<ul>
<li>Long-tail <strong class="mono">50,6×</strong>; 2 lớp chỉ còn <strong>6 ảnh test</strong> —
đoán đúng thêm 1 ảnh lớp hiếm, macro-F1 nhảy ~0,15/22.</li>
<li>Nhiễu giữa các lần chạy <strong>±0,02–0,05</strong> — lớn hơn hầu hết các "cải tiến" định thử.</li>
<li>Gate 0a: pipeline tất định trong một GPU, nhưng <strong>đổi loại GPU = đổi mô hình</strong>
ở cùng seed → không bao giờ trộn phần cứng trong một σ.</li>
</ul>
<p class="takeaway">4 kỷ luật cho <em>mọi</em> con số: <strong>3 seed + σ</strong> ·
<strong>bootstrap CI</strong> · <strong>một quy tắc checkpoint (top3) áp cho tất cả</strong> ·
<strong>lưu logits</strong> để mọi phân tích sau đều 0 epoch.</p>
</div>
<figure><img src="{IMG_EDA}" alt="Phan bo 22 lop, truc log"/>
<figcaption>Phân bố lớp (trục log): 1.467 → 29 ảnh. Dữ liệu cũng được audit rò rỉ 2 lớp
(MD5 + cosine): 0,38% test bị ảnh hưởng, bỏ đi thì Δ = −0,0016 — kết luận không đổi.</figcaption></figure>
</div>"""))

# 4 — TAI LAP BASELINE (diem xuat phat cua ladder)
S.append(slide(4, T, "Cách thực hiện · Bước 1", """
<h2>Tái lập baseline — điểm xuất phát trung thực: 0,6686 ± 0,0234</h2>
<div class="statrow">
<div class="stat"><span class="v plain small">0,6686 ± 0,0234</span><span class="k">B0 DenseNet-121, đúng quy tắc bài báo (best), 3 seed, 30 epoch</span></div>
<div class="stat"><span class="v plain small">+0,78 σ</span><span class="k">chênh +0,0182 so với 0,6504 — trong một độ lệch chuẩn → tái lập được</span></div>
<div class="stat"><span class="v plain small">0,6813</span><span class="k">S0 Swin-T (top3) — baseline 2, nhóm tự thêm</span></div>
</div>
<ul>
<li>Đạt trong <strong>30 epoch</strong> thay vì 150: DenseNet đạt đỉnh val ở epoch 6/30 — thêm epoch không cứu nó.</li>
<li>±0,023 do <strong>một seed may</strong> (0,7008 vs 0,6461/0,6589) — chính là bệnh mà quy tắc <code>top3</code> chữa.</li>
<li>Điều kiện nền đã kiểm: <strong>full fine-tune thắng mọi mức đóng băng ≥ 2σ</strong>
(linear probe 0,5674 → full 0,6686, cách biệt 0,1106) — đặc trưng ImageNet không đủ cho ảnh nội soi.</li>
</ul>
<p class="takeaway">Từ đây, mọi cải tiến được đo <strong>từng bậc một, cùng giao thức</strong> —
slide sau là toàn bộ con đường.</p>"""))

# 5 — MONEY SLIDE: THAC NUOC
S.append(slide(5, T, "Kết quả · Con đường tăng điểm", f"""
<h2>0,6686 → 0,7441: từng bậc được đo riêng — chỉ hai thứ thật sự trả tiền</h2>
{waterfall_svg()}
<p class="note" style="margin-top:6px">Ngưỡng nhiễu của bài toán: <strong>±0,035</strong>. Chỉ
<strong>công thức huấn luyện (+0,0443)</strong> vượt ngưỡng; <strong>cách đo</strong> (top3 + hiệu chỉnh
logit, +0,0237) miễn phí về GPU; còn <span class="neg">đổi kiến trúc (+0,0034)</span> và
<span class="neg">độ phân giải (+0,0041)</span> — hai lever nhóm đặt ra để kiểm định — không phân biệt
được với 0. Kết luận này <em>trái với kỳ vọng ban đầu</em> "nút thắt là backbone".</p>"""))

# 6 — VI SAO LEVER CHINH HOAT DONG: TUONG TAC
S.append(slide(6, T, "Kết quả · Cơ chế của lever chính", """
<h2>Công thức hiện đại không phổ quát — nó chỉ trả tiền khi ghép đúng kiến trúc</h2>
<div class="cols c2">
<div>
<table>
<tr><th>Cùng công thức (80ep · LLRD · EMA · mixup)</th><th class="num">Δ top3</th></tr>
<tr><td>… trên DenseNet-121 @224 (<code>P2b</code>, 3 seed)</td><td class="num neg">−0,0110</td></tr>
<tr class="hl"><td>… trên CoAtNet-0 @224 (<code>P2c</code>) — <strong>tương tác công thức × kiến trúc</strong></td><td class="num">+0,0468 ✅</td></tr>
<tr><td>công thức × độ phân giải (cùng backbone)</td><td class="num">+0,0085</td></tr>
<tr><td>288 vs 224 dưới công thức mới (tốn 1,70× compute)</td><td class="num neg">−0,0006</td></tr>
</table>
</div>
<ul>
<li>Trên cùng seed 0: kiến trúc một mình <span class="neg">−0,0160</span>, công thức trên DenseNet
âm cả 3/3 seed — <strong>ghép lại mới +0,0294</strong> (McNemar p = 0,0018).</li>
<li>→ Đây là một <strong>tương tác</strong>, không phải tổng hai lever độc lập — số hạng duy nhất
vượt ngưỡng ±0,035 trong toàn bộ phân rã.</li>
<li>Hệ quả triển khai: cấu hình <em>nên dùng</em> là <code>P2c</code> @224 (nhanh hơn 1,7×, điểm
tương đương); con số <em>được báo cáo</em> vẫn là <code>P2</code> vì đủ 3 seed.</li>
</ul>
</div>
<p class="takeaway"><strong>Cách improve accuracy, tóm trong một câu:</strong> giữ nguyên dữ liệu,
nâng <em>cách huấn luyện</em> cho đúng backbone hybrid, rồi thu nốt phần tăng miễn phí từ
<em>cách đo</em> — không phải đi tìm kiến trúc to hơn.</p>"""))

# 7 — BANG CHUNG THONG KE
S.append(slide(7, T, "Kết quả · Bằng chứng", f"""
<h2>Bằng chứng thống kê: khoảng tin cậy không chứa baseline</h2>
{ci_svg()}
<p class="note">micro-F1 (accuracy) = <span class="mono">0,850</span> so với 0,8203 của bài báo ·
Kết quả âm cũng báo cáo đủ: ensemble nhiều kiến trúc <em>thua</em> mô hình đơn (0,7130) —
đa dạng hoá không thay được một mô hình tốt.</p>"""))

# 8 — CAI THIEN NAM O DAU (per-class)
S.append(slide(8, T, "Kết quả · Cải thiện nằm ở đâu", f"""
<h2>90,4% phần tăng nằm ở các lớp hiếm — đúng chỗ khó nhất của bài toán</h2>
<div class="cols c38">
<figure><img src="{IMG_PERCLASS}" alt="F1 tung lop va ma tran nham lan cua P2"/>
<figcaption>F1 từng lớp + ma trận nhầm lẫn của P2 (test 1.586 ảnh), đối chiếu Table 3 của bài báo.</figcaption></figure>
<ul>
<li>Mức tăng lớn nhất: <em>Resected polyps</em> <span class="accent">+0,313</span> so với số công bố.</li>
<li>Dư địa còn lại: <span class="mono">85,2%</span> nằm ở 15 lớp hiếm; trần của 7 lớp phổ biến chỉ còn
0,0420 — sát ngưỡng nhiễu.</li>
<li>Nửa trung thực: 2 lớp 6-ảnh-test giữ 0,0552 dư địa — <strong>chỉ thêm dữ liệu mới cứu được</strong>,
không phải thêm mô hình.</li>
</ul>
</div>"""))

# 9 — KIEM CHUNG DOC LAP TREN CPU
S.append(slide(9, T, "Kiểm chứng độc lập", """
<h2>Lặp lại toàn pipeline trên CPU — phương pháp đứng vững trên phần cứng thứ ba</h2>
<div class="cols c2">
<div>
<table>
<tr><th>DenseNet-121, top3</th><th class="num">macro-F1</th></tr>
<tr><td>T4 · 30 ep · batch 32 · fp16 · 3 seed</td><td class="num">0,6780 ± 0,0073</td></tr>
<tr><td>CPU · 15 ep · batch 16 · fp32 · 3 seed</td><td class="num">0,6919 ± 0,0060</td></tr>
</table>
<p class="note" style="margin-top:12px">Hai giao thức khác nhau — chênh lệch là "giao thức + nhiễu",
không phải "CPU tốt hơn". Nhưng hai phát hiện phương pháp <strong>lặp lại nguyên vẹn</strong>:
<code>top3</code> bền nhất (σ nhỏ nhất), và hiệu chỉnh logit <em>phẳng</em> khi thiếu công thức
hiện đại — đúng như trên GPU.</p>
</div>
<div>
<div class="stat"><span class="v">0,7059</span>
<span class="k">hệ thống CPU tuyên bố trước (top3 + ensemble 3 seed) · CI 95% [0,6560; 0,7447]
— <strong>không chứa 0,6504</strong> · accuracy 0,8386 · chi phí <strong>0 đồng</strong>, 3 đêm × ~230 phút</span></div>
<p class="takeaway" style="margin-top:16px">Kể cả <strong>không có GPU</strong>, pipeline này vẫn vượt
số công bố của bài báo theo đúng chuẩn CI — một claim riêng, độc lập với thang T4.</p>
</div>
</div>"""))

# 10 — SAN PHAM
S.append(slide(10, T, "Sản phẩm", f"""
<h2>Triển khai được thật: ONNX + demo Gradio chạy đầu-cuối</h2>
<div class="cols c38">
<figure><img src="{IMG_DEMO}" alt="Demo Gradio chay that tren CPU"/>
<figcaption>Demo chạy thật trên CPU (Playwright tự upload ảnh test) — top-1 đúng, p = 0,961.</figcaption></figure>
<ul>
<li>ONNX trên T4: cấu hình nên triển khai <code>P2c</code> @224 — <span class="mono">11,7 ms/ảnh</span>
@ batch 1, <span class="mono">5,13 ms</span> @ batch 32.</li>
<li>Máy không GPU: MobileNetV3-L <span class="mono">21,7 ms/ảnh</span> (16,9 MB) ·
DenseNet-121 <span class="mono">72,1 ms/ảnh</span>.</li>
<li>Trung thực về sản phẩm: demo = 1 checkpoint + TTA (<span class="mono">0,7199</span>) —
<strong>yếu hơn</strong> hệ thống được báo cáo (0,7441); hai con số không được lẫn.</li>
</ul>
</div>"""))

# 11 — KET LUAN
S.append(slide(11, T, "Kết luận", """
<h2>Ba câu mang về — và những gì được nói thẳng</h2>
<ul>
<li><strong>1 · Tái lập rồi mới vượt:</strong> 0,6686 ± 0,0234 khớp 0,6504; hệ thống đề xuất
<strong>0,7441 ± 0,0088</strong>, CI [0,6986; 0,7736] không chứa baseline — kiểm chứng thêm một lần
độc lập trên CPU (0,7059, CI loại baseline, 0 đồng GPU).</li>
<li><strong>2 · Accuracy tăng nhờ đâu:</strong> không phải backbone (+0,0034) hay độ phân giải
(+0,0041) — mà là <strong>công thức huấn luyện × kiến trúc hybrid (+0,0468)</strong> cộng
<strong>cách đo</strong> (top3 + hiệu chỉnh logit, +0,0237 với 0 epoch).</li>
<li><strong>3 · Trung thực là một tính năng:</strong> mọi kết quả âm đều in trong báo cáo
(công thức làm DenseNet tệ đi −0,0110; 288px ≈ 0; ensemble kiến trúc thua mô hình đơn);
hạn chế nêu rõ: P2c 1 seed · không ID bệnh nhân · demo yếu hơn hệ thống.</li>
</ul>
<p class="takeaway">Toàn bộ 122 con số được đối chiếu máy với nguồn (<code>check_numbers.py</code>);
mọi phân tích hậu kỳ chạy lại từ logits đã lưu — <strong>tái lập không cần GPU</strong>.</p>"""))

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
