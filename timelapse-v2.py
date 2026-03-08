#!/usr/bin/env python3
"""
Timelapse interpolator: generate N frames linearly interpolated in time from
a source directory of JPEGs, using EXIF DateTimeOriginal or file mtime as
timestamps. Blending is done on GPU with PyTorch when available.
"""

import argparse
import bisect
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional


def tlog(msg: str) -> None:
    """Print datetimestamp followed by the log message."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} {msg}")

import numpy as np
from PIL import Image

# Optional GPU
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# EXIF tag id for DateTimeOriginal
DATETIME_ORIGINAL_TID = 36867


def get_timestamp_from_exif(path: Path) -> Optional[float]:
    """Return Unix timestamp from EXIF DateTimeOriginal, or None if missing."""
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return None
            raw = exif.get(DATETIME_ORIGINAL_TID)
            if raw is None:
                return None
            # Format: 'YYYY:MM:DD HH:MM:SS'
            dt = datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S")
            return dt.timestamp()
    except Exception:
        return None


def get_timestamp(path: Path) -> float:
    """Return Unix timestamp: EXIF DateTimeOriginal if present, else mtime."""
    t = get_timestamp_from_exif(path)
    if t is not None:
        return t
    return path.stat().st_mtime


def collect_frames(src_dir: Path) -> list[tuple[float, Path]]:
    """List all JPEGs in src_dir with their timestamps, sorted by time."""
    pairs = []
    for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
        for p in src_dir.glob(ext):
            pairs.append((get_timestamp(p), p))
    pairs.sort(key=lambda x: x[0])
    return pairs


def load_image_rgb(path: Path) -> np.ndarray:
    """Load image as RGB uint8 HWC numpy array."""
    with Image.open(path) as img:
        img = img.convert("RGB")
    return np.array(img, dtype=np.uint8)


def interpolate_cpu(img_a: np.ndarray, img_b: np.ndarray, alpha: float) -> np.ndarray:
    """Blend img_a and img_b with weight alpha (0=all A, 1=all B)."""
    a = img_a.astype(np.float64) * (1.0 - alpha)
    b = img_b.astype(np.float64) * alpha
    out = (a + b).clip(0, 255).astype(np.uint8)
    return out


def interpolate_gpu(
    img_a: np.ndarray, img_b: np.ndarray, alpha: float, device: torch.device
) -> np.ndarray:
    """Blend on GPU; returns HWC uint8 numpy."""
    a = torch.from_numpy(img_a).to(device=device, dtype=torch.float32) / 255.0
    b = torch.from_numpy(img_b).to(device=device, dtype=torch.float32) / 255.0
    out = (a * (1.0 - alpha) + b * alpha).clamp(0.0, 1.0)
    out = (out * 255.0).byte().cpu().numpy()
    return out


def find_straddling_pair(
    times: list[float], t_out: float
) -> tuple[int, int, float]:
    """
    Find indices (i, j) and alpha such that t_out is between times[i] and times[j]
    and alpha = (t_out - times[i]) / (times[j] - times[i]) when i != j.
    Clamps t_out to [times[0], times[-1]].
    """
    n = len(times)
    if n == 1:
        return 0, 0, 0.0
    t_min, t_max = times[0], times[-1]
    t = max(t_min, min(t_max, t_out))
    if t <= t_min:
        return 0, 0, 0.0
    if t >= t_max:
        return n - 1, n - 1, 1.0
    # i such that times[i] <= t < times[i+1]; use bisect for O(log n)
    i = bisect.bisect_right(times, t) - 1
    i = max(0, min(i, n - 2))
    denom = times[i + 1] - times[i]
    alpha = (t - times[i]) / denom if denom > 0 else 0.0
    return i, i + 1, alpha


def _progress_line(current: int, total: int, width: int = 24) -> str:
    """One-line progress: frame NNN/total [=====>    ] pp%"""
    pct = (current / total * 100) if total else 100.0
    filled = int(width * current / total) if total else width
    bar = "=" * filled + ">" * (1 if current < total else 0) + " " * (width - filled - (1 if current < total else 0))
    return f"frame {current}/{total} [{bar[:width]}] {pct:.0f}%"


def _render_one(
    idx: int,
    t_out: float,
    times: list[float],
    frames: list[tuple[float, Path]],
    out_dir: Path,
    blend_fn: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    gpu_lock: Optional[threading.Lock],
) -> int:
    """Generate one output frame. Returns idx (for ordering)."""
    i, j, alpha = find_straddling_pair(times, t_out)
    path_a, path_b = frames[i][1], frames[j][1]
    img_a = load_image_rgb(path_a)
    img_b = load_image_rgb(path_b)
    if img_a.shape != img_b.shape:
        pil_b = Image.fromarray(img_b)
        pil_b = pil_b.resize(
            (img_a.shape[1], img_a.shape[0]), Image.Resampling.LANCZOS
        )
        img_b = np.array(pil_b, dtype=np.uint8)
    if gpu_lock:
        with gpu_lock:
            out_img = blend_fn(img_a, img_b, alpha)
    else:
        out_img = blend_fn(img_a, img_b, alpha)
    out_path = out_dir / f"img-{idx:05d}.jpg"
    Image.fromarray(out_img).save(out_path, "JPEG", quality=99)
    return idx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interpolate a timelapse sequence from source JPEGs by timestamp."
    )
    parser.add_argument("src_dir", type=Path, help="Source directory of JPEGs")
    parser.add_argument(
        "num_frames", type=int, help="Number of output frames to generate"
    )
    parser.add_argument("out_dir", type=Path, help="Output directory")
    parser.add_argument(
        "--threading",
        "-j",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel workers (default: 1). GPU blends are serialized.",
    )
    args = parser.parse_args()

    src_dir = args.src_dir.resolve()
    out_dir = args.out_dir.resolve()
    n_out = args.num_frames

    if not src_dir.is_dir():
        print(f"Error: source directory does not exist: {src_dir}", file=sys.stderr)
        sys.exit(1)
    if n_out < 1:
        print("Error: num_frames must be >= 1", file=sys.stderr)
        sys.exit(1)
    n_workers = max(1, args.threading)

    start_time = time.perf_counter()
    tlog("Starting up.")
    tlog("Reading source file timestamps.")
    frames = collect_frames(src_dir)
    if not frames:
        print(f"Error: no JPEGs found in {src_dir}", file=sys.stderr)
        sys.exit(1)
    tlog(f"Found {len(frames)} source frames.")

    times = [t for t, _ in frames]
    t_min, t_max = times[0], times[-1]

    # Output timestamps: linear from t_min to t_max
    if n_out == 1:
        out_times = [t_min]
    else:
        out_times = [
            t_min + (t_max - t_min) * i / (n_out - 1) for i in range(n_out)
        ]

    use_gpu = HAS_TORCH and torch.cuda.is_available()
    device = torch.device("cuda") if use_gpu else torch.device("cpu")
    if use_gpu:
        blend = lambda a, b, alpha: interpolate_gpu(a, b, alpha, device)
    else:
        blend = interpolate_cpu

    out_dir.mkdir(parents=True, exist_ok=True)

    # Serialize GPU blends (one at a time per process); CPU can run fully parallel
    gpu_lock = threading.Lock() if use_gpu else None

    completed = 0
    completed_lock = threading.Lock()

    if n_workers == 1:
        for idx, t_out in enumerate(out_times):
            _render_one(idx, t_out, times, frames, out_dir, blend, gpu_lock)
            completed += 1
            print("\r" + _progress_line(completed, n_out), end="", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _render_one,
                    idx,
                    t_out,
                    times,
                    frames,
                    out_dir,
                    blend,
                    gpu_lock,
                ): idx
                for idx, t_out in enumerate(out_times)
            }
            for future in as_completed(futures):
                future.result()  # raise any exception
                with completed_lock:
                    completed += 1
                print("\r" + _progress_line(completed, n_out), end="", flush=True)

    print()
    elapsed = time.perf_counter() - start_time
    fps = n_out / elapsed if elapsed > 0 else 0
    tlog(f"Finished in {elapsed:.1f} seconds ({fps:.1f} frames/s).")
    tlog(f"Wrote {n_out} frames to {out_dir}")


if __name__ == "__main__":
    main()
