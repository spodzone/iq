# iq

**Disclaimer:** All the code in this repository and most of the documentation has been vibe-coded using Cursor and GLM-5. If “I felt the flow and shipped it” sits fine with your ethics, great — welcome aboard and enjoy it. If you need design docs and things, maybe grab a coffee and read the code first. It serves to solve my problems - if you find the scripts useful, all to the good. No warranty or guarantee is offered in any eventuality.

---

This repo holds three small sub-projects for image and timelapse work:

### 1. **Super-resolve**
Single-image super-resolution via a small U‑net that learns to clean up Lanczos upscaling artefacts instead of inventing detail. Train on your own photos, run at 2× with one image in, one image out.
→ See [SUPER-RESOLVE.md](SUPER-RESOLVE.md).

### 2. **pic3blend**
Batch pipeline for “coll-*” folders: upscale (or 2× super-resolve), align, optional ghost fixing, then blend—mean, HDR, focus stack, or min/median/max—into one 16‑bit TIFF. Handy when your source dir is on a network mount and you want one command per stack.
→ See [PIC3BLEND.md](PIC3BLEND.md).

### 3. **Timelapse interpolator**
Takes a folder of JPEGs (with EXIF or mtime as time) and spits out a fixed number of frames at even time steps, blending between neighbours so irregular or gappy captures become a smooth, steady timelapse. No optical flow, just simple time-based blending (GPU when available).
→ See [TIMELAPSE.md](TIMELAPSE.md).

---

License: [GPL v3](LICENSE.md).
