#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dung BAO_CAO.md thanh mot trang HTML doc lap (hinh nhung thanh data URI).

    python report/build_html.py

Nguon duy nhat la BAO_CAO.md -- sua o day roi chay lai script, khong sua .html bang tay.
Trang khong goi mot tai nguyen ngoai nao ngoai Google Fonts.
"""
import base64
import io
import os
import re
import sys

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "BAO_CAO.md")
OUT = os.path.join(HERE, "bao_cao.html")

PAGE_TITLE = "Vượt baseline GastroVision"


# ------------------------------------------------------------------ hinh anh
def data_uri(rel):
    path = os.path.join(HERE, rel.replace("/", os.sep))
    if not os.path.exists(path):
        sys.exit("khong tim thay hinh: " + path)
    b64 = base64.b64encode(open(path, "rb").read()).decode("ascii")
    return "data:image/png;base64," + b64


# ------------------------------------------------------------------ masthead
def split_head(md_text):
    """Tach phan dau (title / subtitle / meta) khoi than bai o dau '---' dau tien."""
    lines = md_text.split("\n")
    cut = next(i for i, l in enumerate(lines) if l.strip() == "---")
    head, body = lines[:cut], lines[cut + 1:]

    title = subtitle = ""
    meta = []
    for l in head:
        s = l.strip()
        if not s:
            continue
        if s.startswith("# "):
            title = s[2:].strip()
        elif s.startswith("**") and not subtitle:
            subtitle = s.strip("*").strip()
        else:
            meta.append(s)
    return title, subtitle, meta, "\n".join(body)


# ------------------------------------------------------------------ h2 -> header co so muc + trong so
H2 = re.compile(r"<h2>(.*?)</h2>", re.S)
WEIGHT = re.compile(r"^<strong>(\d+)%</strong>$")


def slug(n, text):
    base = re.sub(r"<[^>]+>", "", text)
    base = re.sub(r"[^0-9A-Za-z]+", "-", base).strip("-").lower()
    return ("m%s-" % n if n else "") + (base[:40] or "muc")


SECTIONS = []          # (id, so muc, ten, trong so) -- muc luc duoc sinh tu chinh day


def section_header(inner):
    num = ""
    m = re.match(r"^(\d+)\s+·\s+(.*)$", inner, re.S)
    if m:
        num, inner = m.group(1), m.group(2)
    elif " · " in inner:                       # 'Phu luc · ...'
        inner = inner.replace(" · ", " — ", 1)

    weight, kicker = "", ""
    if " — " in inner:
        title, tail = inner.split(" — ", 1)
        wm = WEIGHT.match(tail.strip())
        if wm:
            weight = wm.group(1)
        else:
            kicker = tail.strip()
    else:
        title = inner

    sid = slug(num, title)
    SECTIONS.append((sid, num, title, weight))
    meta = []
    if num:
        meta.append('<span class="s-num">%s</span>' % num)
    if weight:
        meta.append('<span class="s-w">%s%% của rubric</span>' % weight)
    bar = ('<div class="s-bar" role="img" aria-label="trọng số %s phần trăm">'
           '<span style="width:%s%%"></span></div>' % (weight, weight)) if weight else ""

    return (
        '<header class="sec" id="%s">'
        '%s'
        '<h2>%s</h2>'
        '%s'
        '%s'
        '</header>'
    ) % (
        sid,
        ('<div class="s-meta">%s</div>' % "".join(meta)) if meta else "",
        title,
        ('<p class="s-kick">%s</p>' % kicker) if kicker else "",
        bar,
    )


# ------------------------------------------------------------------ muc luc
# Ten ngan cho muc luc: tieu de day du qua dai cho o luoi hep.
SHORT = {
    "1": "Bài toán &amp; baseline",
    "2": "Phân tích dữ liệu (EDA)",
    "3": "Xử lý dữ liệu",
    "4": "Nhãn &amp; kiểm định",
    "5": "Kiến trúc CNN / TF / hybrid",
    "6": "Transfer learning",
    "7": "Deployment",
    "8": "Năm nguyên tắc",
    "9": "Hạn chế",
    "10": "Kết luận",
}


def toc_html():
    items = []
    for sid, num, title, w in SECTIONS:
        if not num:
            continue
        items.append(
            '<a class="toc-i" href="#%s"><span class="toc-n">%s</span>'
            '<span class="toc-t">%s</span>'
            '<span class="toc-w">%s</span></a>'
            % (sid, num, SHORT.get(num, title), (w + "%") if w else "·")
        )
    return '<nav class="toc wide" aria-label="Mục lục">%s</nav>' % "".join(items)


# ------------------------------------------------------------------ build
def build():
    md_text = io.open(SRC, encoding="utf-8").read()
    title, subtitle, meta, body_md = split_head(md_text)

    html = markdown.markdown(
        body_md,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
        output_format="html5",
    )

    # tieu de muc -> header co so muc + thanh trong so
    html = H2.sub(lambda m: section_header(m.group(1)), html)

    # bang: cho phep cuon ngang rieng, va noi rong hon cot chu
    html = html.replace("<table>", '<div class="wide tw"><table>')
    html = html.replace("</table>", "</table></div>")

    # hinh: nhung thanh data URI + chu thich lay tu alt
    def fig(m):
        alt, src = m.group(1), m.group(2)
        return ('<figure class="wide"><img src="%s" alt="%s">'
                '<figcaption>%s</figcaption></figure>') % (data_uri(src), alt, alt)

    html = re.sub(r'<p><img alt="([^"]*)" src="([^"]+)"\s*/?></p>', fig, html)

    # blockquote: canh bao (co dau canh bao) khac voi ghi chu
    def bq(m):
        inner = m.group(1)
        cls = "warn" if "⚠️" in inner else "note"
        return '<blockquote class="%s">%s</blockquote>' % (cls, inner)

    html = re.sub(r"<blockquote>(.*?)</blockquote>", bq, html, flags=re.S)

    # '---' trong markdown thanh <hr>; moi muc da co header rieng nen bo di
    html = re.sub(r"<hr\s*/?>", "", html)

    # muc luc chen ngay truoc muc 1 (sinh tu SECTIONS -- id luon khop)
    first = next((s for s in SECTIONS if s[1] == "1"), None)
    if first is None:
        sys.exit("khong tim thay muc 1 -- muc luc se rong")
    anchor = '<header class="sec" id="%s">' % first[0]
    if html.count(anchor) != 1:
        sys.exit("neo muc luc khong duy nhat: " + anchor)
    html = html.replace(anchor, toc_html() + anchor, 1)

    page = TEMPLATE.replace("{{TITLE}}", PAGE_TITLE)
    page = page.replace("{{DOC_TITLE}}", title)
    page = page.replace("{{SUBTITLE}}", subtitle)
    page = page.replace("{{META}}", "".join("<span>%s</span>" % m for m in meta))
    page = page.replace("{{BODY}}", html)

    io.open(OUT, "w", encoding="utf-8", newline="\n").write(page)
    print("da ghi %s (%.0f KB)" % (os.path.basename(OUT), os.path.getsize(OUT) / 1024))


TEMPLATE = r"""<title>{{TITLE}}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,600;0,700;1,400&family=Be+Vietnam+Pro:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --bg:#F6F8F8; --surface:#FFFFFF; --surface-2:#EDF2F2; --surface-3:#E3EAEA;
  --ink:#131F1F; --ink-2:#485958; --ink-3:#77898A;
  --rule:#D6DFDE; --rule-2:#C3D0CF;
  --accent:#0D6E6E; --accent-2:#0A5354; --accent-soft:#E0EFEE;
  --warn:#96591A; --warn-soft:#FAEFE1; --warn-rule:#E3C69B;
  --neg:#9E4048; --pos:#2A6B58;
  --link-rule:#8FBEBD; --accent-line:#7FB4B3;
  --shadow:0 1px 2px rgba(19,31,31,.05), 0 8px 24px -16px rgba(19,31,31,.18);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#0D1415; --surface:#131C1D; --surface-2:#192425; --surface-3:#202D2E;
    --ink:#E3EAE9; --ink-2:#A6B5B4; --ink-3:#7A8A8A;
    --rule:#263334; --rule-2:#324141;
    --accent:#45B2AB; --accent-2:#6BC6BF; --accent-soft:#14302F;
    --warn:#D69C55; --warn-soft:#2A2117; --warn-rule:#4A3A22;
    --neg:#D9848B; --pos:#5CB596;
    --link-rule:#2F6664; --accent-line:#2F6664;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 30px -18px rgba(0,0,0,.7);
  }
}
:root[data-theme="dark"]{
  --bg:#0D1415; --surface:#131C1D; --surface-2:#192425; --surface-3:#202D2E;
  --ink:#E3EAE9; --ink-2:#A6B5B4; --ink-3:#7A8A8A;
  --rule:#263334; --rule-2:#324141;
  --accent:#45B2AB; --accent-2:#6BC6BF; --accent-soft:#14302F;
  --warn:#D69C55; --warn-soft:#2A2117; --warn-rule:#4A3A22;
  --neg:#D9848B; --pos:#5CB596;
  --link-rule:#2F6664; --accent-line:#2F6664;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 30px -18px rgba(0,0,0,.7);
}

*{box-sizing:border-box}
body{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:"Be Vietnam Pro","Segoe UI",system-ui,-apple-system,sans-serif;
  font-weight:400; font-size:16.5px; line-height:1.72;
  -webkit-font-smoothing:antialiased;
}
img{max-width:100%; height:auto; display:block}
a{color:var(--accent); text-decoration:none; border-bottom:1px solid var(--link-rule)}
a:hover{border-bottom-color:var(--accent)}
:focus-visible{outline:2px solid var(--accent); outline-offset:3px; border-radius:2px}

/* ---------- masthead ---------- */
.mast{
  border-bottom:1px solid var(--rule);
  background:
    radial-gradient(120% 140% at 88% -30%, var(--accent-soft) 0%, transparent 58%),
    var(--surface);
}
.mast-in{max-width:1120px; margin:0 auto; padding:76px 28px 46px}
.eyebrow{
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:11.5px; letter-spacing:.16em;
  text-transform:uppercase; color:var(--accent); margin:0 0 22px;
}
h1{
  font-family:Spectral,Georgia,"Times New Roman",serif; font-weight:700;
  font-size:clamp(2.3rem,5.4vw,3.9rem); line-height:1.05; letter-spacing:-.018em;
  margin:0; text-wrap:balance; max-width:24ch;
}
.sub{
  font-family:Spectral,Georgia,serif; font-style:italic; font-weight:400;
  font-size:clamp(1.05rem,2vw,1.32rem); line-height:1.5; color:var(--ink-2);
  margin:20px 0 0; max-width:56ch; text-wrap:balance;
}
.mast-meta{
  display:flex; flex-wrap:wrap; gap:8px 22px; margin-top:34px;
  padding-top:22px; border-top:1px solid var(--rule);
  font-size:13.5px; color:var(--ink-3);
}
.mast-meta span{white-space:nowrap}

/* ---------- document grid ---------- */
.doc{
  max-width:1120px; margin:0 auto; padding:0 28px 96px;
  display:grid; grid-template-columns:1fr min(70ch,100%) 1fr;
}
.doc > *{grid-column:2; min-width:0}
.doc > .wide{grid-column:1 / -1; width:100%}

.doc > p{margin:0 0 1.15em}
.doc > ol, .doc > ul{margin:0 0 1.3em; padding-left:1.35em}
.doc > ol li, .doc > ul li{margin:0 0 .55em}
.doc > ol{counter-reset:none}
strong{font-weight:600}
em{font-style:italic}

/* ---------- section headers (so muc + trong so la thong tin thuc) ---------- */
.sec{grid-column:2; margin:72px 0 26px; scroll-margin-top:24px}
.sec:first-of-type{margin-top:56px}
.s-meta{display:flex; align-items:baseline; gap:14px; margin-bottom:10px}
.s-num{
  font-family:"IBM Plex Mono",monospace; font-size:12px; font-weight:500;
  letter-spacing:.1em; color:var(--accent);
  border:1px solid var(--accent-line);
  border-radius:2px; padding:2px 8px;
}
.s-w{
  font-family:"IBM Plex Mono",monospace; font-size:11.5px; letter-spacing:.1em;
  text-transform:uppercase; color:var(--ink-3);
}
.sec h2{
  font-family:Spectral,Georgia,serif; font-weight:600;
  font-size:clamp(1.6rem,3.2vw,2.16rem); line-height:1.14; letter-spacing:-.014em;
  margin:0; text-wrap:balance;
}
.s-kick{font-size:14.5px; color:var(--ink-3); margin:8px 0 0; font-style:italic}
.s-bar{
  margin-top:16px; height:3px; background:var(--rule); border-radius:2px; overflow:hidden;
}
.s-bar > span{display:block; height:100%; background:var(--accent); border-radius:2px}

h3{
  font-family:"Be Vietnam Pro",sans-serif; font-weight:600;
  font-size:1.12rem; line-height:1.4; letter-spacing:-.005em;
  margin:2.5em 0 .7em; color:var(--ink); text-wrap:balance;
}
h3 + p, h3 + .wide{margin-top:0}
h4{font-family:"Be Vietnam Pro",sans-serif; font-weight:600; font-size:1rem; margin:2em 0 .5em}

/* ---------- muc luc ---------- */
.toc{
  margin:8px 0 4px; display:grid; gap:1px;
  grid-template-columns:repeat(auto-fit,minmax(232px,1fr));
  background:var(--rule); border:1px solid var(--rule); border-radius:4px; overflow:hidden;
}
.toc-i{
  display:flex; align-items:baseline; gap:10px; padding:13px 15px;
  background:var(--surface); border-bottom:0; color:var(--ink);
  font-size:14.5px; transition:background .12s ease;
}
.toc-i:hover{background:var(--surface-2); border-bottom:0}
.toc-n{
  font-family:"IBM Plex Mono",monospace; font-size:11.5px; color:var(--accent);
  min-width:1.4em; font-variant-numeric:tabular-nums;
}
.toc-t{flex:1}
.toc-w{
  font-family:"IBM Plex Mono",monospace; font-size:11.5px; color:var(--ink-3);
  font-variant-numeric:tabular-nums;
}

/* ---------- tables ---------- */
.tw{margin:1.5em 0 1.9em; overflow-x:auto; border:1px solid var(--rule); border-radius:4px; background:var(--surface)}
table{border-collapse:collapse; width:100%; font-size:14.5px; line-height:1.5}
thead th{
  background:var(--surface-2); text-align:left; font-weight:600; font-size:12.5px;
  letter-spacing:.03em; color:var(--ink-2); padding:11px 14px;
  border-bottom:1px solid var(--rule-2); white-space:nowrap;
}
tbody td{
  padding:10px 14px; border-bottom:1px solid var(--rule); vertical-align:top;
  font-variant-numeric:tabular-nums;
}
tbody tr:last-child td{border-bottom:0}
tbody tr:nth-child(even) td{background:color-mix(in oklab, var(--surface-2) 45%, transparent)}
td strong{color:var(--accent-2); font-weight:600}
td em{color:var(--ink-3)}
table code{font-size:.92em}

/* ---------- figures ---------- */
figure{
  margin:2em 0 2.2em; padding:18px 18px 14px; background:var(--surface);
  border:1px solid var(--rule); border-radius:4px; box-shadow:var(--shadow);
}
figure img{margin:0 auto; border-radius:2px}
figcaption{
  margin-top:14px; padding-top:12px; border-top:1px solid var(--rule);
  font-size:13px; color:var(--ink-3); text-align:center;
}

/* ---------- code ---------- */
pre{
  margin:1.4em 0 1.7em; padding:15px 17px; background:var(--surface-2);
  border:1px solid var(--rule); border-left:2px solid var(--accent); border-radius:3px;
  overflow-x:auto; font-size:12.9px; line-height:1.6;
}
pre code{font-family:"IBM Plex Mono",ui-monospace,monospace; color:var(--ink-2); white-space:pre}
:not(pre) > code{
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.875em;
  background:var(--surface-2); border:1px solid var(--rule); border-radius:3px;
  padding:.08em .34em; color:var(--ink);
}

/* ---------- callouts ---------- */
blockquote{
  margin:1.6em 0; padding:16px 20px; border-radius:4px;
  font-size:15.4px; line-height:1.66;
}
blockquote > :first-child{margin-top:0}
blockquote > :last-child{margin-bottom:0}
blockquote.note{
  background:var(--accent-soft); border:1px solid var(--accent-line);
  border-left:3px solid var(--accent);
}
blockquote.warn{
  background:var(--warn-soft); border:1px solid var(--warn-rule);
  border-left:3px solid var(--warn);
}
blockquote.warn strong{color:var(--warn)}
blockquote p{margin:0 0 .8em}
blockquote p:last-child{margin-bottom:0}

@media (max-width:640px){
  .mast-in{padding:52px 20px 34px}
  .doc{padding:0 20px 64px}
  body{font-size:16px}
  .sec{margin-top:56px}
}
@media (prefers-reduced-motion:reduce){
  *{transition:none !important; animation:none !important}
}
</style>

<header class="mast">
  <div class="mast-in">
    <p class="eyebrow">AIN501 · Final project · Deep Learning</p>
    <h1>{{DOC_TITLE}}</h1>
    <p class="sub">{{SUBTITLE}}</p>
    <div class="mast-meta">{{META}}</div>
  </div>
</header>

<article class="doc">
{{BODY}}
</article>
"""


if __name__ == "__main__":
    build()
