#!/usr/bin/env python3
"""
stack-32bit.py -o output.tiff *.tif

Average-mean stacker using an incremental rolling algorithm.
Images are accumulated in 32-bit float and written as a 32-bit float TIFF.
"""

import argparse
import os
import sys

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Average-mean stacker with 32-bit float output.")
    p.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output TIFF path (written as 32-bit float).",
    )
    p.add_argument(
        "inputs",
        nargs="+",
        help="Input images (TIFFs or other formats readable by OpenCV). Use shell globs like *.tif.",
    )
    p.add_argument(
        "--raw-range",
        action="store_true",
        help="Do NOT normalize inputs; average in the raw input numeric range (e.g. 0..65535). "
             "Most viewers will display float TIFFs correctly only when values are in 0..1.",
    )
    return p.parse_args()


def _to_float01(img: np.ndarray) -> np.ndarray:
    """Convert uint8/uint16/float to float32 in [0,1] (best-effort)."""
    if img.dtype == np.uint8:
        return (img.astype(np.float32) / 255.0).clip(0.0, 1.0)
    if img.dtype == np.uint16:
        return (img.astype(np.float32) / 65535.0).clip(0.0, 1.0)
    if np.issubdtype(img.dtype, np.floating):
        x = img.astype(np.float32)
        # If it already looks normalized, keep it. Otherwise, scale by max of the frame.
        mx = float(np.nanmax(x)) if x.size else 0.0
        if mx <= 1.5:
            return np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
        if mx > 0:
            return np.nan_to_num(x / mx, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
        return np.zeros_like(x, dtype=np.float32)
    # Fallback: interpret as integer-like and scale by dtype max if available
    info = np.iinfo(img.dtype) if np.issubdtype(img.dtype, np.integer) else None
    if info and info.max > 0:
        return (img.astype(np.float32) / float(info.max)).clip(0.0, 1.0)
    return img.astype(np.float32)


def main():
    args = parse_args()
    paths = [p for p in args.inputs if os.path.isfile(p)]
    if not paths:
        print("No input files found.", file=sys.stderr)
        sys.exit(1)

    running = None
    n = 0
    shape = None

    for path in paths:
        print(f"Reading {path}", file=sys.stderr, flush=True)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: could not read {path}, skipping.", file=sys.stderr)
            continue

        if shape is None:
            shape = img.shape
        elif img.shape != shape:
            print(f"Warning: shape mismatch for {path} (got {img.shape}, expected {shape}), skipping.", file=sys.stderr)
            continue

        img_f = img.astype(np.float32) if args.raw_range else _to_float01(img)

        if running is None:
            running = img_f.copy()
            n = 1
        else:
            n += 1
            running += (img_f - running) / float(n)

    if running is None:
        print("No valid input images to stack.", file=sys.stderr)
        sys.exit(1)

    # Write as 32-bit float TIFF; cv2 preserves dtype for TIFF output.
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if not cv2.imwrite(args.output, running.astype(np.float32)):
        print(f"Failed to write output to {args.output}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

