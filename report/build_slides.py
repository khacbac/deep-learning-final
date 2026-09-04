# -*- coding: utf-8 -*-
"""Sinh bo slide bao ve tu noi dung BAO_CAO.md -> report/slides.html.

Thu tu lap luan cua slide = thu tu cua BAO_CAO.md (dung nhu ghi chu trong RESULTS.md).
Moi con so tren slide deu da co trong BAO_CAO.md va do check_numbers.py doi chieu —
KHONG dua so moi vao day ma khong them vao CHECKS.

Chay:  python report/build_slides.py   ->  report/slides.html (tu chua, nhung anh data-URI)
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
IMG_CURVES = b64("figures/24_duong_hoc_val.png")
IMG_DEMO = b64("demo/29b_demo_gradio_cpu.png")

CSS = """
:root{
  --paper:#F6F8F7; --card:#FFFFFF; --ink:#1C2321; --muted:#56635E;
  --teal:#1F6E5E; --teal-soft:#E3EEEA; --red:#A8402F; --line:#D7DEDA;
}
*{box-sizing:border-box;margin:0}
html,body{height:100%}
body{background:var(--ink);font-family:'Be Vietnam Pro',system-ui,-apple-system,'Segoe UI',sans-serif;color:var(--ink)}
#stage{position:fixed;inset:0;display:flex;align-items:center;justify-content:center;background:var(--ink)}
.slide{width:1280px;height:720px;background:var(--paper);display:none;flex-direction:column;
  padding:44px 64px 36px;position:absolute;transform-origin:center center}
.slide.active{display:flex}
.slide header{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px}
.eyebrow{font-family:'IBM Plex Mono',monospace;font-size:13px;letter-spacing:.18em;color:var(--teal);text-transform:uppercase}
.pageno{font-family:'IBM Plex Mono',monospace;font-size:13px;color:var(--muted)}
h1{font-family:'Archivo',sans-serif;font-weight:700;font-size:58px;line-height:1.08;text-wrap:balance}
h2{font-family:'Archivo',sans-serif;font-weight:650;font-size:40px;line-height:1.12;margin-bottom:18px;text-wrap:balance}
.slide footer{margin-top:auto;padding-top:12px;border-top:1px solid var(--line);
  display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--muted)}
.body{flex:1;min-height:0;display:flex;flex-direction:column}
p,li{font-size:20px;line-height:1.5}
.lead{font-size:24px;line-height:1.5;max-width:62ch}
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
.stat .v{font-family:'IBM Plex Mono',monospace;font-size:64px;font-weight:600;color:var(--teal);line-height:1.05}
.stat .v.plain{color:var(--ink)}
.stat .k{font-size:17px;color:var(--muted);max-width:30ch}
.statrow{display:flex;gap:56px;margin:26px 0}
figure{margin:0;display:flex;flex-direction:column;gap:8px;min-height:0}
figure img{max-width:100%;max-height:100%;object-fit:contain;border:1px solid var(--line);background:#fff}
figcaption{font-size:15px;color:var(--muted)}
.note{font-size:16px;color:var(--muted);border-left:3px solid var(--teal);padding:6px 0 6px 14px;max-width:70ch}
.titleslide{justify-content:center;gap:22px}
.titleslide .who{font-family:'IBM Plex Mono',monospace;font-size:16px;color:var(--muted);letter-spacing:.06em}
kbd{font-family:'IBM Plex Mono',monospace;background:var(--card);border:1px solid var(--line);border-radius:4px;padding:0 6px;font-size:12px}
#hint{position:fixed;right:16px;bottom:12px;color:#9FB0AA;font-size:12px;font-family:'IBM Plex Mono',monospace;opacity:.85}
/* luoi tong quan */
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

# ---- SVG: khoang tin cay tren MOT truc (0.62 -> 0.78) --------------------- #
def ci_svg():
    x0, x1, w = 0.62, 0.78, 980          # mien gia tri -> pixel
    def X(v):
        return 90 + (v - x0) / (x1 - x0) * w
    rows = [
        # (nhan, tam, thap, cao, mau, chu thich khoang)
        ("B0 · DenseNet-121 (tái lập)", 0.6686, 0.6452, 0.6920, "#56635E", "±σ, 3 seed"),
        ("P2 · hệ thống đề xuất", 0.7441, 0.6986, 0.7736, "#1F6E5E", "CI 95% bootstrap"),
    ]
    parts = [f'<svg viewBox="0 0 1120 240" role="img" font-family="IBM Plex Mono,monospace" '
             f'aria-label="Khoang tin cay so voi baseline 0,6504">']
    # truc + moc
    for t in (0.65, 0.70, 0.75):
        parts.append(f'<line x1="{X(t):.0f}" y1="36" x2="{X(t):.0f}" y2="188" stroke="#D7DEDA" stroke-width="1"/>')
        parts.append(f'<text x="{X(t):.0f}" y="212" font-size="15" fill="#56635E" text-anchor="middle">{str(t).replace(".", ",")}</text>')
    # duong baseline cong bo
    bx = X(0.6504)
    parts.append(f'<line x1="{bx:.0f}" y1="24" x2="{bx:.0f}" y2="188" stroke="#A8402F" stroke-width="2" stroke-dasharray="6 5"/>')
    parts.append(f'<text x="{bx:.0f}" y="16" font-size="15" fill="#A8402F" text-anchor="middle" font-weight="600">baseline công bố 0,6504</text>')
    def vn(v):
        return f"{v:.4f}".replace(".", ",")
    y = 70
    for name, mid, lo, hi, color, note in rows:
        parts.append(f'<line x1="{X(lo):.0f}" y1="{y}" x2="{X(hi):.0f}" y2="{y}" stroke="{color}" stroke-width="6" stroke-linecap="round"/>')
        parts.append(f'<circle cx="{X(mid):.0f}" cy="{y}" r="10" fill="{color}" stroke="#F6F8F7" stroke-width="2"/>')
        parts.append(f'<text x="{X(lo):.0f}" y="{y - 20}" font-size="17" fill="#1C2321" font-family="Be Vietnam Pro,sans-serif" font-weight="600">{name}</text>')
        parts.append(f'<text x="{X(lo):.0f}" y="{y + 28}" font-size="15" fill="#56635E">'
                     f'{vn(mid)} · [{vn(lo)}; {vn(hi)}] {note}</text>')
        y += 84
    parts.append("</svg>")
    return "".join(parts)


def slide(no, total, section, body, footer_l="GastroVision · AIN501 Deep Learning", cls=""):
    return f"""<section class="slide {cls}" id="s{no}">
<header><span class="eyebrow">{section}</span><span class="pageno">{no:02d} / {total}</span></header>
<div class="body">{body}</div>
<footer><span>{footer_l}</span><span>04-09-2026</span></footer>
</section>"""


T = 15
S = []

S.append(slide(1, T, "AIN501 · Final Project", f"""
<div class="body titleslide">
<h1>Vượt baseline công bố trên GastroVision</h1>
<p class="lead">Phân loại 22 lớp ảnh nội soi tiêu hoá — tái lập DenseNet-121 của bài báo
(arXiv 2307.08140), rồi vượt nó <strong>với bằng chứng thống kê</strong>, không phải một con số may mắn.</p>
<div class="statrow">
<div class="stat"><span class="v plain">0,6504</span><span class="k">baseline công bố · DenseNet-121, Table 2 (1 lần chạy, không sai số)</span></div>
<div class="stat"><span class="v">0,7441 ± 0,0088</span><span class="k">hệ thống đề xuất P2 · 3 seed, CI 95% không chứa baseline</span></div>
</div>
<p class="who">Nhóm 2 thành viên · khacbac + quang3447 · toàn bộ số trích tự động từ notebook (108 con số được đối chiếu máy)</p>
</div>""", cls="titleslide"))

S.append(slide(2, T, "Bài toán", """
<h2>Dữ liệu công khai, một con số phải vượt</h2>
<div class="cols c2">
<ul>
<li><strong>GastroVision</strong>: 8.000 ảnh nội soi, 27 thư mục lớp → giữ lớp &gt; 25 ảnh
theo luật của bài báo → <span class="mono">22 lớp / 7.930 ảnh</span>.</li>
<li>Chia phân tầng 60:20:20, <code>SPLIT_SEED = 42</code> — kiểm chứng lại bằng Table 3 của
chính bài báo: 16/22 lớp khớp chính xác, 6 lớp lệch ±1, tổng test trùng khít 1.586.</li>
<li>Baseline mạnh nhất họ công bố: <strong>DenseNet-121 pretrained, macro-F1 0,6504</strong> —
một lần chạy, 150 epoch, không có sai số đi kèm.</li>
</ul>
<ul>
<li>Nhiệm vụ đề bài: <em>tái lập baseline đã công bố</em>, rồi <em>đề xuất cải tiến vượt nó</em>.</li>
<li>Metric chính: <strong>macro-F1</strong> (bắt buộc với dữ liệu mất cân bằng); accuracy chỉ để đối chiếu.</li>
<li>Hai baseline của nhóm: <code>B0</code> DenseNet-121 (khớp số bài báo) và <code>S0</code> Swin-T
(nhánh Transformer bài báo không có).</li>
</ul>
</div>"""))

S.append(slide(3, T, "Dữ liệu · EDA", f"""
<h2>Long-tail 50,6× — và cái giá của nó lên phép đo</h2>
<div class="cols c38">
<figure><img src="{IMG_EDA}" alt="Phan bo 22 lop, truc log"/><figcaption>Phân bố lớp (trục log): lớp lớn nhất 1.467 ảnh, nhỏ nhất 29.</figcaption></figure>
<ul>
<li>Tỷ lệ mất cân bằng <strong class="mono">50,6×</strong>; 2 lớp chỉ còn <strong>6 ảnh test</strong>.</li>
<li>Đoán đúng thêm 1 ảnh của lớp hiếm → F1 lớp đó nhảy ~0,15 → kéo cả trung bình 22 lớp.</li>
<li>Hệ quả thiết kế: <strong>không con số 1-seed nào được trích</strong> — mọi thứ là
trung bình ± σ trên 3 seed, kèm bootstrap CI.</li>
</ul>
</div>"""))

S.append(slide(4, T, "Dữ liệu · Audit", """
<h2>Audit rò rỉ hai lớp — kết quả không đến từ rò rỉ</h2>
<ul>
<li><strong>Lớp 1 — trùng byte</strong>: MD5 toàn bộ 7.930 ảnh.
<strong>Lớp 2 — gần trùng</strong>: cosine ≥ 0,98 trên embedding pretrained.</li>
<li>Tìm được <strong>9 cặp vắt qua các tập</strong> → 6/1.586 ảnh test bị ảnh hưởng
= <span class="mono">0,38%</span>.</li>
<li>Ablation bắt buộc: bỏ các ảnh nghi rò rỉ rồi tính lại → Δ macro-F1 = <span class="mono">−0,0016</span>
trên mô hình đề xuất — <strong>kết luận không đổi</strong>.</li>
<li>Phát hiện phụ giá trị hơn: cặp gần-trùng <em>khác nhãn</em> = <strong>nhãn nghi ngờ</strong> của
chính bộ dữ liệu — một đoạn riêng trong báo cáo.</li>
</ul>
<p class="note">Không có ID bệnh nhân trong bản phát hành công khai → không chia theo bệnh nhân được;
audit cosine là chặn dưới, và hạn chế này được nêu thẳng trong báo cáo (mục 9).</p>"""))

S.append(slide(5, T, "Phương pháp · Kỷ luật đo", """
<h2>Đo trước khi tin: nhiễu lớn hơn hiệu ứng</h2>
<ul>
<li><strong>Gate 0a</strong> — chạy cùng cấu hình 2 lần, so cả đường val: pipeline <strong>tất định
trong một loại GPU</strong>, nhưng <strong>A100 ≠ T4 ở cùng seed</strong> → không bao giờ trộn
phần cứng trong một σ.</li>
<li>Một phép lặp lại ngoài kế hoạch (4 cấu hình × 3 seed × 2 loại GPU) cho thấy:
<strong>xếp hạng kiến trúc không sống nổi qua một lần đổi phần cứng</strong>;
quy tắc <code>top3</code> bền gấp ~4 lần quy tắc một-checkpoint.</li>
<li><strong>6 quy tắc chọn checkpoint từ một lần huấn luyện</strong> (3 quy tắc × 2 chế độ TTA),
chốt <code>top3</code> <em>một lần, áp cho tất cả</em>.</li>
<li>Mọi phân tích hậu kỳ (CI, ensemble, hiệu chỉnh logit) đọc lại từ <strong>logits đã lưu — 0 epoch</strong>.</li>
</ul>"""))

S.append(slide(6, T, "Kết quả 1 · Tái lập", """
<h2>Tái lập baseline — trung thực, kèm dải nhiễu</h2>
<div class="statrow">
<div class="stat"><span class="v plain">0,6686 ± 0,0234</span><span class="k">B0 DenseNet-121, quy tắc của bài báo (best), 3 seed, 30 epoch</span></div>
<div class="stat"><span class="v plain">+0,78 σ</span><span class="k">chênh +0,0182 so với 0,6504 — trong một độ lệch chuẩn → tái lập được</span></div>
</div>
<ul>
<li>Đạt trong <strong>30 epoch</strong> thay vì 150 của bài báo — DenseNet đạt đỉnh sớm, thêm epoch không cứu nó.</li>
<li>±0,023 là dải rộng: báo cáo nói rõ đây <em>không</em> phải "trùng khít", và 0,6504 tự nó
cũng mang ~±0,012 nhiễu seed không được công bố.</li>
<li>Baseline 2 — <code>S0</code> Swin-T: <span class="mono">0,6813</span> (top3) — nhánh Transformer nhóm tự thêm.</li>
</ul>"""))

S.append(slide(7, T, "Kết quả 2 · Vượt baseline", f"""
<h2>Hệ thống đề xuất P2 — CI không chứa 0,6504</h2>
<p class="lead">CoAtNet-0 @288 + công thức huấn luyện hiện đại (80 epoch, cosine + LLRD + EMA + mixup)
+ ensemble top-3 checkpoint + hiệu chỉnh logit:</p>
{ci_svg()}
<p class="note">Micro-F1 (accuracy) = <span class="mono">0,850</span> so với 0,8203 của bài báo ·
ensemble 3 seed đạt <span class="mono">0,7587</span> (một dòng riêng vì tiêu 3 lần huấn luyện).</p>"""))

S.append(slide(8, T, "Vì sao thắng · Phân rã", """
<h2>Phân rã mức tăng — lever nào thật sự trả tiền</h2>
<table>
<tr><th>Lever (đo riêng, cùng một quy tắc chấm)</th><th class="num">Δ macro-F1</th><th>Vượt ngưỡng nhiễu ±0,035?</th></tr>
<tr class="hl"><td>Công thức huấn luyện hiện đại (P1 → P2)</td><td class="num">+0,0443</td><td>✅ có</td></tr>
<tr><td>Hiệu chỉnh logit lúc suy luận (0 epoch)</td><td class="num">+0,0143</td><td>— không</td></tr>
<tr><td>Quy tắc checkpoint best → top3 (0 epoch)</td><td class="num">+0,0094</td><td>— không</td></tr>
<tr><td>Độ phân giải 224 → 288 (công thức cũ)</td><td class="num">+0,0041</td><td>— không</td></tr>
<tr><td>Đổi kiến trúc DenseNet → CoAtNet (một mình)</td><td class="num">+0,0034</td><td>— không</td></tr>
</table>
<p class="note">Hai lever nhóm đặt ra để kiểm định — <strong>kiến trúc</strong> và <strong>độ phân giải</strong> —
không phân biệt được với 0. Thứ trả tiền là <strong>công thức</strong> (+0,0443) và <strong>cách đo</strong>
(+0,0237 với 0 epoch phụ). Kết luận này trái với kỳ vọng ban đầu "nút thắt là backbone".</p>"""))

S.append(slide(9, T, "Vì sao thắng · Tương tác", """
<h2>Không phải tổng hai lever — là một tương tác</h2>
<div class="cols c2">
<ul>
<li>Cùng công thức hiện đại áp lên DenseNet-121 @224 (<code>P2b</code>):
<span class="neg">−0,0110</span>, cả 3/3 seed đều âm — công thức <strong>không phổ quát</strong>.</li>
<li>Ô thứ tư của bảng 2×2 (<code>P2c</code> = CoAtNet @224 + công thức):
tách được <strong>công thức × kiến trúc = +0,0468</strong> ✅ — số hạng duy nhất vượt ngưỡng.</li>
<li>Trên cùng seed 0: kiến trúc một mình <span class="neg">−0,0160</span>, công thức trên DenseNet âm,
ghép lại <span class="accent">+0,0294</span> (McNemar p = 0,0018).</li>
</ul>
<ul>
<li><strong>288 px không làm nên chuyện</strong>: dưới công thức mới, 288 vs 224 =
<span class="neg">−0,0006</span> với 1,70× chi phí.</li>
<li>→ Cấu hình <em>nên triển khai</em> là <code>P2c</code> @224; con số <em>được báo cáo</em>
vẫn là <code>P2</code> (đủ 3 seed) — hai vai trò không lẫn.</li>
<li>Số hạng công thức × độ phân giải: +0,0085 — không vượt ngưỡng.</li>
</ul>
</div>"""))

S.append(slide(10, T, "Kết quả · Per-class", f"""
<h2>Cải thiện dồn vào đúng chỗ khó: lớp hiếm</h2>
<div class="cols c38">
<figure><img src="{IMG_PERCLASS}" alt="F1 tung lop va ma tran nham lan"/><figcaption>F1 từng lớp + ma trận nhầm lẫn của P2 (test 1.586 ảnh).</figcaption></figure>
<ul>
<li><strong class="mono">90,4%</strong> phần tăng so với Table 3 của bài báo nằm ở <strong>các lớp hiếm</strong>;
mức tăng lớn nhất: <em>Resected polyps</em> <span class="accent">+0,313</span>.</li>
<li>Dư địa còn lại: <span class="mono">85,2%</span> nằm ở 15 lớp hiếm; trần của 7 lớp phổ biến
chỉ còn 0,0420 — sát ngưỡng nhiễu.</li>
<li>Nửa khó chịu: 2 lớp 6-ảnh-test đóng góp 0,0552 dư địa — chỉ dữ liệu mới cứu được, không phải mô hình.</li>
</ul>
</div>"""))

S.append(slide(11, T, "Kiến trúc & đường học", f"""
<h2>CNN vs Transformer vs Hybrid — ở 3 seed, chưa được phép kết luận</h2>
<div class="cols c38">
<figure><img src="{IMG_CURVES}" alt="Duong hoc val cua cac cau hinh"/><figcaption>Đường học val: 30 epoch đủ cho công thức cũ; công thức hiện đại cần 80.</figcaption></figure>
<ul>
<li>Cùng giao thức @224, quy tắc top3: <code>B0</code> <span class="mono">0,6780</span> ·
<code>S0</code> <span class="mono">0,6813</span> · <code>P0</code> <span class="mono">0,6814</span> —
chênh nhau nhỏ hơn nhiễu.</li>
<li>Bằng chứng độc lập (phép lặp A100↔T4): xếp hạng kiến trúc đổi chỗ khi đổi GPU.</li>
<li>P2 ổn định cuối huấn luyện: độ lệch 5 epoch cuối chỉ <span class="mono">0,0019</span>.</li>
</ul>
</div>"""))

S.append(slide(12, T, "Transfer learning", """
<h2>Freeze vs trainable — full fine-tune thắng, cách biệt 2σ</h2>
<table>
<tr><th>Điều kiện (cùng DenseNet-121, cùng giao thức)</th><th class="num">macro-F1</th></tr>
<tr><td>T1 · Linear probe (đóng băng toàn bộ backbone)</td><td class="num">0,5674</td></tr>
<tr><td>T3 · Progressive unfreezing + LR phân biệt</td><td class="num">0,6394</td></tr>
<tr><td>T2 · Đóng băng nửa dưới backbone</td><td class="num">0,6596</td></tr>
<tr class="hl"><td>T4 · Fine-tune toàn mạng (= B0)</td><td class="num">0,6686</td></tr>
</table>
<p class="note">Khoảng cách T1 → T4 = <span class="mono">0,1106</span>; ngưỡng 2σ = 0,0146 —
cả ba điều kiện đóng băng đều thua quá ngưỡng. Đặc trưng ImageNet không đủ cho ảnh nội soi:
phải cho backbone học lại.</p>"""))

S.append(slide(13, T, "Triển khai & demo", f"""
<h2>Sản phẩm: ONNX + demo Gradio chạy thật</h2>
<div class="cols c38">
<figure><img src="{IMG_DEMO}" alt="Demo Gradio chay that tren CPU"/><figcaption>Demo Gradio chạy đầu-cuối trên CPU (02-09), Playwright tự upload ảnh test — top-1 đúng, p = 0,961.</figcaption></figure>
<ul>
<li>ONNX trên T4: CoAtNet@288 <span class="mono">8,74 ms/ảnh</span> @ batch 32, mô hình <span class="mono">114,8 MB</span>.</li>
<li>Máy không GPU (12 threads, fp32): DenseNet-121 <span class="mono">72,1 ms/ảnh</span>,
MobileNetV3-L <span class="mono">21,7 ms/ảnh</span>.</li>
<li>Demo = 1 checkpoint + TTA (<span class="mono">0,7199</span> với P2) — <strong>yếu hơn</strong> hệ thống
được báo cáo (0,7441); hai con số không được lẫn.</li>
</ul>
</div>"""))

S.append(slide(14, T, "Triển khai · Vòng CPU", """
<h2>Không có GPU vẫn vượt baseline — vòng CPU 3 seed</h2>
<div class="cols c2">
<div>
<table>
<tr><th>DenseNet-121 (top3)</th><th class="num">macro-F1</th></tr>
<tr><td>T4 · 30 ep · batch 32 · fp16</td><td class="num">0,6780 ± 0,0073</td></tr>
<tr><td>CPU · 15 ep · batch 16 · fp32</td><td class="num">0,6919 ± 0,0060</td></tr>
</table>
<p class="note" style="margin-top:14px">Hai giao thức <strong>khác nhau</strong> — chênh lệch là
"giao thức + nhiễu", không phải "CPU tốt hơn" (dưới quy tắc best hai phép đo chồng hẳn:
0,6709 ± 0,0163 vs 0,6686 ± 0,0234). Chi tiết: BAO_CAO mục 7.3.</p>
</div>
<div>
<div class="stat"><span class="v">0,7059</span>
<span class="k">hệ thống tuyên bố trước: top3 + ensemble 3 seed · CI 95% <strong>[0,6560; 0,7447]</strong>
— <strong>không chứa 0,6504</strong> · accuracy 0,8386 &gt; 0,8203 của bài báo</span></div>
<ul style="margin-top:18px">
<li>Chi phí: <strong>0 đồng</strong> — 3 × ~230 phút CPU chạy đêm, 12 threads.</li>
<li><code>top3</code> bền nhất (σ 0,0060) và hiệu chỉnh logit <em>phẳng</em> khi thiếu công thức
hiện đại — <strong>tái xác nhận hai phát hiện phương pháp trên phần cứng thứ ba</strong>.</li>
<li>Claim <strong>độc lập</strong> với thang T4 (Gate 0a) — không thay thế con số P2 0,7441.</li>
</ul>
</div>
</div>"""))

S.append(slide(15, T, "Kết luận", """
<h2>Kết luận — và những gì được nói thẳng</h2>
<ul>
<li><strong>Tái lập được</strong> 0,6504 (0,6686 ± 0,0234, +0,78σ) và <strong>vượt nó</strong>:
0,7441 ± 0,0088, CI 95% [0,6986; 0,7736] — <strong>không chứa baseline</strong>.</li>
<li>Luận điểm chính (ngoài dự kiến): nút thắt <strong>không phải backbone</strong> —
là <strong>công thức huấn luyện × kiến trúc</strong> (+0,0468) cộng <strong>cách đo</strong> (+0,0237, 0 epoch).</li>
<li>Kết quả âm cũng được báo cáo: công thức làm DenseNet tệ đi (−0,0110); 288 px ≈ 0;
ensemble nhiều kiến trúc <em>thua</em> mô hình đơn (0,7130).</li>
<li>Hạn chế nêu rõ: <code>P2c</code> 1 seed · không ID bệnh nhân · demo yếu hơn hệ thống.</li>
</ul>
<p class="note">Toàn bộ 108 con số của báo cáo được đối chiếu máy với nguồn (<code>check_numbers.py</code>);
mọi phân tích hậu kỳ đọc từ logits đã lưu — tái chạy không cần GPU.</p>"""))

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
