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
    return p.parse_args()


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
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: could not read {path}, skipping.", file=sys.stderr)
            continue

        if shape is None:
            shape = img.shape
        elif img.shape != shape:
            print(f"Warning: shape mismatch for {path} (got {img.shape}, expected {shape}), skipping.", file=sys.stderr)
            continue

        img_f = img.astype(np.float32)

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

