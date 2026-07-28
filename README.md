# samcaf.github.io

Personal website for Samuel Alipour-fard — a modern, dependency-free static site
(hand-written HTML/CSS/JS, no build step).

## Structure

```
index.html          Showcase homepage (hero · featured work · about · publications · skills · contact)
projects.html       Project gallery with filters, plus a Notes section (physics PDFs)
publications.html   Publications, thesis, talks, honors, service
styles/main.css     Design system (dark + light themes, physics accent)
scripts/main.js     Theme toggle, nav, scroll reveals, project filters
assets/media/       Video (RENC intro)
assets/projects/    Project screenshots / figures
images/             Brand mark + favicon
pdfs/               CV and note PDFs
```

## Local preview

No tooling required — just serve the folder:

```bash
python3 -m http.server 8000
# then open http://localhost:8000
```

## Deploying to GitHub Pages

This site is designed to live at **https://samcaf.github.io**. To publish:

1. Copy these files into your `samcaf/samcaf.github.io` repository
   (replacing the old site), or push this folder there as the repo root.
2. Commit and push to the `main` (or `master`) branch.
3. In the repo settings → **Pages**, make sure the source is that branch, root folder.

The `.nojekyll` file tells GitHub Pages to serve the files as-is.

## Things you may want to update

- **`pdfs/cv.pdf`** — this is now a freshly compiled, **no-Profile** build of your CV.
  It's generated from your LaTeX source at `~/Documents/cv/` via a new `web` target:
  `resume-web.tex` (sets `\def\apptarget{web}`) plus a one-line guard added to
  `cv/summary.tex` that omits the Profile section for that target. To refresh it:
  `cd ~/Documents/cv && xelatex resume-web.tex` (twice), then copy `resume-web.pdf`
  to `pdfs/cv.pdf`. Existing CV targets (oratomic/qec/finance/…) are unaffected.
- **Private repos** — four repos you link work were private, so their links were
  removed to avoid 404s: `Energy-Weighted-Observable-Correlators` (kept as a featured
  project + figures, but delinked), `Edu-QoL` (card removed), and `Paper-Template` /
  `Notes-Template` (links removed). If you make any public on GitHub, tell me and I'll
  re-add the links.
- **Hero interests / tagline** — in `index.html`, the `.hero-lead` paragraph and the
  `.tag-strip` chips ("Physics · Scientific Programming · AI & Machine Learning · Travel").
- **Featured projects** — the three `<article class="feature">` blocks in `index.html`.
- **Stats row** — the four `.stat` blocks under the hero.
- Colors/fonts live at the top of `styles/main.css` (the `:root` and
  `html[data-theme="light"]` token blocks).
