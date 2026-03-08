# causalLatentPolytree

Master's Thesis project — *LiNGAM Models on Minimal Latent Polytrees: A Cumulant-Based Discrepancy Approach*
Department of Mathematics, Technical University of Munich.

---

## Repository Structure

```
causalLatentPolytree/
├── thesis/          # LaTeX thesis (synced with Overleaf)
│   ├── main.tex
│   ├── references.bib
│   ├── Front_matter/
│   ├── Title_page/
│   └── Graphics/
├── experiments/     # Experiment run logs and configs
├── *.py             # Python source code
└── README.md
```

---

## Syncing with Overleaf

The `thesis/` folder is linked to the Overleaf project via a separate git remote called `overleaf`. All edits to the thesis (whether made locally or on Overleaf) go through this workflow.

### Setup (already done — for reference)

```bash
git remote add overleaf https://git:<TOKEN>@git.overleaf.com/67b3120e3230ebe864eb7956
```

A local working copy of the Overleaf repo lives at `/tmp/overleaf_push`. It was initialised with:

```bash
git clone https://git:<TOKEN>@git.overleaf.com/67b3120e3230ebe864eb7956 /tmp/overleaf_push
```

### Push local changes → Overleaf

```bash
# 1. Copy changed files from thesis/ into the Overleaf clone
cd /tmp/overleaf_push
cp /path/to/thesis/main.tex .                          # or whichever files changed
cp /path/to/thesis/Front_matter/*.tex Front_matter/    # etc.

# 2. Commit and push
git add -A
git commit -m "describe your changes"
git push
```

> **Note:** Force push is not allowed by Overleaf. Always pull before pushing if you have edited on Overleaf directly.

### Pull Overleaf changes → local

```bash
cd /tmp/overleaf_push
git pull

# Copy updated files back into the repo
cp -r . /path/to/thesis/

# Commit to the main repo
cd /path/to/causalLatentPolytree
git add thesis/
git commit -m "Sync thesis changes from Overleaf"
git push origin main
```

---

## GitHub Remote

The full repository (Python code + thesis) is pushed to GitHub:

```bash
git push origin main
```

GitHub tracks everything. Overleaf only sees the contents of `thesis/`.

---

## Token Renewal

Overleaf git tokens may expire. To regenerate: **overleaf.com → Account Settings → Git Integration → Generate token**, then update the remote URL:

```bash
git remote set-url overleaf https://git:<NEW_TOKEN>@git.overleaf.com/67b3120e3230ebe864eb7956
# and in /tmp/overleaf_push:
git remote set-url origin https://git:<NEW_TOKEN>@git.overleaf.com/67b3120e3230ebe864eb7956
```
