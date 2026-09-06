# GastroVision paper (academic-paper rewrite)

English, single-column academic-paper version of `../BAO_CAO.md`, formatted in the
style of *Journal of Computer Science and Cybernetics* (the course sample paper).

## Files
- `paper.tex` — the paper source (self-contained, standard packages only).
- `paper.pdf` — compiled output (12 pages).
- `figures/fig_eda.png` — per-class distribution (English), from `../tables/06_eda.txt`.
- `figures/fig_perclass_f1.png` — per-class F1 (English), from `../tables/18_per_class_va_confusion.txt`.
- `make_figures.py` — regenerates the two figures with English labels.

## Compile
```bash
# Option A: tectonic (self-contained, downloads packages on first run)
tectonic paper.tex

# Option B: TeX Live / MacTeX
pdflatex paper.tex && pdflatex paper.tex

# Option C: upload paper.tex + figures/ to Overleaf
```

## Regenerate figures (optional)
```bash
python3 -m venv v && v/bin/pip install matplotlib
v/bin/python make_figures.py
```

All numbers trace back to `../tables*/` and `../RESULTS.md`; none are hand-typed.
