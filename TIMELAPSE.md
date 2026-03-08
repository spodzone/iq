# Timelapse interpolator

A small Python tool that turns a folder of JPEG timelapse frames into a new sequence with a fixed number of frames, **interpolated in time**. Gaps, irregular intervals, or missing frames in the source are smoothed out so the output plays at a steady rate.

## What it does

- **Reads** all JPEGs from a source directory.
- **Assigns a timestamp** to each image: EXIF `DateTimeOriginal` if present, otherwise the file’s modification time.
- **Builds a linear time span** from the first to the last source timestamp.
- **Generates N output frames** at evenly spaced times over that span. For each output time it finds the two source frames that straddle it and **blends them** with a linear mix (no AI or optical flow—simple alpha blend).
- **Writes** the result as `img-00000.jpg` … `img-(N-1).jpg` at 99% JPEG quality.

So you get a smooth, fixed-length timelapse even when the original capture had variable or skipped intervals.

## Why you might want it

- Your camera dropped frames or had uneven intervals; you want a smooth, consistent framerate for video.
- You have “one frame every N seconds” source material and want a specific number of frames (e.g. for a 60fps clip of a given length).
- You want to slow down or stretch a timelapse by generating more in-between frames.

## Install

Requires Python 3.9+.

```bash
cd timelapse-v2
pip install -r requirements.txt
```

Dependencies: `numpy`, `Pillow`, and `torch`. If PyTorch is installed and CUDA is available, blending runs on the GPU; otherwise it uses the CPU.

## Usage

```text
python timelapse-v2.py <src_dir> <num_frames> <out_dir> [--threading N]
```

| Argument      | Meaning                          |
|---------------|----------------------------------|
| `src_dir`     | Directory containing source JPEGs |
| `num_frames`  | Number of output frames to generate |
| `out_dir`     | Directory where output JPEGs are written |

**Options**

- `--threading N` or `-j N` — Use up to N workers in parallel (default: 1). Speeds up I/O and, on CPU, blending; with GPU, blending is serialized per process.

**Examples**

```bash
# 1500 output frames from jpeg-in, written to jpeg-out
python timelapse-v2.py jpeg-in 1500 jpeg-out

# Same, with 5 parallel workers
python timelapse-v2.py jpeg-in 1500 jpeg-out --threading 5
```

Output files are always named `img-00000.jpg`, `img-00001.jpg`, … in `out_dir`. The script creates `out_dir` if it doesn’t exist.
