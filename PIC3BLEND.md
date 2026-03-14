# PIC3BLEND

`pic3blend.py` processes **collections of images** in `./coll-*` directories: it upscales them or optionally super-resolves at 2×, aligns them, corrects ghosting, then blends them into a single 16‑bit TIFF. The **blend mode** (mean, HDR, focus stack, min/median/max) is inferred from the directory name or specified with `--mode`.

---

## What it is

A batch pipeline that:

1. **Finds** all subdirectories matching  `./coll-*`.
2. **In each directory:** loads images, upscales them (Lanczos or, if the scale factor is 2 and `--model` specifies a path to a pretrained model, super-resolves), aligns every image to the first, optionally runs ghosting correction, then blends with the mode implied by the directory name.
3. **Writes** one 16‑bit TIFF per directory: `{blend-type}_{align-algorithm}_{basefilename}.tiff`.

Alignment uses OpenCV (AKAZE then ECC fallback); warping, ghosting, and built-in blending use the GPU (CUDA or Metal MPS) when available.

---

## Why you want it

- **Bracketed / multi-shot stacks:** You have several shots of the same scene (e.g. exposure brackets, focus steps, or multiple handheld shots) and want one output: mean, HDR, focus stack, or min/median/max.
- **Consistent workflow:** Put each stack in a `coll-*` folder whose name encodes the blend type; run one command to upscale, align, correct ghosts, and blend.
- **Quality:** Optional 2× super-resolution (when scale is 2 and a model is given), Lanczos upscaling, alignment, and optional ghost correction before blending.

---

## How to run it

From the directory that contains your `./coll-*` folders:

```bash
pip install -r requirements-pic3blend.txt
python pic3blend.py [options]
```

Or with a scale and optional super-resolve model:

```bash
python pic3blend.py --scale 2 --model path/to/super-resolve.pt
```

The script discovers all `coll-*` directories under the current working directory and processes each one. Progress and device info are logged to stderr with timestamps.

---

## Subdirectory name and structure

### Directory name

- **Must** match `coll-*` (e.g. `coll-mean`, `coll-hdr`, `coll-focus-stack`, `coll-median`).
- **Blend mode** is inferred from the directory name by the **first** matching keyword:
  - **mean** → mean blend (e.g. `coll-mean`, `coll-mean-brackets`)
  - **hdr** → HDR via enfuse (e.g. `coll-hdr`)
  - **focus** → focus stack via enfuse (e.g. `coll-focus`)
  - **min** → pixelwise minimum (e.g. `coll-min`)
  - **median** → pixelwise median (e.g. `coll-median`)
  - **max** → pixelwise maximum (e.g. `coll-max`)
- If none of these appear, **mean** is used.
- You can override the blend mode with `--mode` (see options below).

### Contents

- **Images only:** Put the images for that collection in the directory. Supported extensions: `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.bmp`, `.webp`.
- **Base filename:** The **first** image in **sorted filename order** is the “base”; its filename stem is used in the output name (e.g. `DSC_001.jpg` → output includes `DSC_001` in the TIFF name).

### Example layout

```
./
  coll-mean-01/
    DSC_001.jpg
    DSC_002.jpg
    DSC_003.jpg
  coll-hdr-02/
    exp1.png
    exp2.png
    exp3.png
  coll-focus-03/
    f1.tif
    f2.tif
    f3.tif
```

After running, you get e.g. `mean_akaze_DSC_001.tiff`, `hdr_akaze_exp1.tiff`, `focus_akaze_f1.tiff` inside each respective directory.

---

## What happens (pipeline)

For each `coll-*` directory, in order:

1. **Upscaling**
   Every image is upscaled by the `--scale` factor (Lanczos). If `--scale` is exactly 2 and `--model` is set, each image is instead passed through `super-resolve.py run` for 2× super-resolution.

2. **Alignment**
   All upscaled images are aligned to the first (base) image. The alignment method is the first available from `--align` (default: AKAZE, then ECC). Warping runs on GPU when possible.

3. **Ghosting (optional, default)**
   If `--ghosting 1` is set (as it is by default), a post-alignment step runs: pixels with high variance across the stack are replaced with the stack median to reduce double images / moving-object artefacts. Blend mode is unchanged.

4. **Blending**
   The aligned (and optionally ghost-corrected) stack is blended according to the mode:
   - **mean / min / median / max** → built-in pixelwise blend (GPU when available).
   - **hdr** → enfuse with fixed HDR-style weights (16‑bit).
   - **focus** → enfuse with fixed focus-stack options (16‑bit).

5. **Output**
   One 16‑bit TIFF per directory:
   `{blend-type}_{align-algorithm}_{basefilename}.tiff`
   (e.g. `mean_akaze_DSC_001.tiff`).

---

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--scale` | `2` | Scale factor: one number (e.g. `2`) or `width,height` (e.g. `1.5,2`). All images are upscaled by this before alignment and blending. |
| `--model` | — | Path to a super-resolve checkpoint. **Used only when scale is exactly 2.** Each image is then super-resolved via `super-resolve.py run` instead of Lanczos 2×. |
| `--align` | `akaze,ecc` | Comma-separated list of alignment methods to try in order. Supported: `akaze`, `ecc`. The first that succeeds is used; its name appears in the output filename. Use an empty value to skip alignment. |
| `--ghosting` | `0` | Set to `1` to run ghosting correction after alignment (high-variance pixels replaced with median). Does not change the blend mode. |
| `--mode` | — | Override blend mode: `mean`, `hdr`, `focus`, `min`, `median`, or `max`. If not set, mode is inferred from the directory name. |

---

## Requirements

- **Python deps:** See `requirements-pic3blend.txt` (opencv-python, numpy, torch). Install with `pip install -r requirements-pic3blend.txt`.
- **Optional:** For **hdr** and **focus** modes, **enfuse** must be installed and on your `PATH` (typically from the Hugin package).

---

## Logging

The script prints timestamped lines to **stderr** (e.g. when it starts, when upscaling/aligning/ghosting/blending begin, and when each output file is written). This keeps logs separate from any stdout output.
