#!/usr/bin/env python3
"""
Process ./coll-* directories: upscale, optionally super-resolve, align, and blend
images according to the subdirectory name (mean, hdr, focus, min, median, max).
Output: 16-bit TIFF named {blend-type}_{align-algorithm}_{basefilename}.tiff
Uses GPU (CUDA / Metal MPS) for ghosting, alignment warping, and blending when available.
"""

import argparse
import glob
import os
import subprocess
import sys
import tempfile
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Allow large images (e.g. upscaled 16-bit TIFFs); we're loading our own files, not untrusted input.
Image.MAX_IMAGE_PIXELS = None

from concurrent.futures import ThreadPoolExecutor
import shutil

# Common image extensions (OpenCV imread)
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


def tlog(msg):
    """Emit a timestamped info log entry to stderr."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def get_device():
    """Prefer CUDA, then Metal (MPS), then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _numpy_to_tensor(img_bgr, device, batch=False):
    """HWC uint8/uint16 BGR -> CHW float32 [0,1] on device; optional batch dim."""
    x = torch.from_numpy(img_bgr).to(device=device, dtype=torch.float32)
    if img_bgr.dtype == np.uint16:
        x = x.div_(65535.0)
    else:
        x = x.div_(255.0)
    x = x.permute(2, 0, 1)  # HWC -> CHW
    if batch:
        x = x.unsqueeze(0)
    return x


def _tensor_to_numpy(x, dtype=np.uint8):
    """CHW or NCHW float [0,1] -> HWC BGR with requested dtype (uint8/uint16)."""
    if x.dim() == 4:
        x = x.squeeze(0)
    x = x.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    if dtype == np.uint16:
        return (x * 65535.0 + 0.5).astype(np.uint16)
    return (x * 255.0 + 0.5).astype(np.uint8)


def parse_scale(s):
    """Parse --scale: '2' or '1.5,2.0' -> (sx, sy)."""
    s = s.strip()
    if "," in s:
        a, b = s.split(",", 1)
        return float(a.strip()), float(b.strip())
    v = float(s)
    return v, v


def image_files_in_dir(d):
    """Return sorted list of image file paths in directory d."""
    names = [
        f for f in os.listdir(d)
        if os.path.isfile(os.path.join(d, f)) and os.path.splitext(f)[1].lower() in IMAGE_EXT
    ]
    return sorted([os.path.join(d, n) for n in names])


def load_image(path):
    """Robust image loader with 16-bit preference.

    For TIFF (`.tif`/`.tiff`), use Pillow first to avoid OpenCV TIFF bugs and to
    preserve 16-bit data. For other formats, try OpenCV first, then Pillow.

    Returns uint8 or uint16 BGR numpy array or None on failure.
    """
    ext = os.path.splitext(path)[1].lower()
    # For TIFFs, go straight to Pillow to dodge OpenCV's TIFFReadDirectory issues.
    if ext in (".tif", ".tiff"):
        try:
            with Image.open(path) as im:
                if im.mode in ("I;16", "I;16B", "I;16L"):
                    im = im.convert("I;16")
                    arr = np.array(im)  # (H,W) uint16
                    if arr.ndim == 2:
                        arr = np.stack([arr] * 3, axis=-1)
                    return arr
                if im.mode not in ("RGB", "RGBA"):
                    im = im.convert("RGB")
                arr = np.array(im)
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                if arr.shape[2] == 4:
                    arr = arr[:, :, :3]
                arr = arr[:, :, ::-1].copy()
                return arr.astype(arr.dtype)
        except Exception as e:
            tlog(f"load_image (TIFF) failed for {path}: {e}")
            return None

    # Non-TIFF: OpenCV first, then Pillow
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    # Fallback: Pillow can often read formats OpenCV cannot
    try:
        with Image.open(path) as im:
            # Preserve 16-bit where possible
            if im.mode in ("I;16", "I;16B", "I;16L"):
                im = im.convert("I;16")
                arr = np.array(im)  # (H,W) uint16
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                return arr
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGB")
            arr = np.array(im)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            # Convert RGB(A) -> BGR
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            arr = arr[:, :, ::-1].copy()
            return arr.astype(arr.dtype)
    except Exception as e:
        tlog(f"load_image failed for {path}: {e}")
        return None


def _tmpdir_for_base(base_name):
    """Return /tmp/iq-{basefilename}, with base_name sanitized for use in a path."""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in base_name).strip("._") or "out"
    safe = safe[:64]
    return f"/tmp/iq-{safe}"


def _paths_in_tmpdir(tmpdir, prefix):
    """Return sorted list of full paths to files in tmpdir whose filenames start with prefix."""
    names = sorted(f for f in os.listdir(tmpdir) if f.startswith(prefix) and os.path.isfile(os.path.join(tmpdir, f)))
    return [os.path.join(tmpdir, n) for n in names]


def _read_images_from_tmpdir(tmpdir, prefix):
    """Read images from tmpdir whose filenames start with prefix, sorted by name. Return list of BGR arrays."""
    paths = _paths_in_tmpdir(tmpdir, prefix)
    out = []
    for p in paths:
        im = load_image(p)
        if im is not None:
            out.append(im)
    return out


def _to_u16(img):
    """Ensure an image is uint16, preserving as much precision as possible."""
    if img.dtype == np.uint16:
        return img
    if img.dtype == np.uint8:
        return (img.astype(np.uint16) * 257)
    if np.issubdtype(img.dtype, np.floating):
        return (np.clip(img, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    return img.astype(np.uint16)


def _copy_exif_from_base(base_path, target_path):
    """Copy EXIF from base image into target using exiftool, if available."""
    try:
        # exiftool -overwrite_original -TagsFromFile base -all:all target
        subprocess.run(
            [
                "exiftool",
                "-overwrite_original",
                "-TagsFromFile",
                base_path,
                "-all:all",
                target_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        tlog("exiftool not found, skipping EXIF copy")
    except Exception as e:
        tlog(f"exiftool failed for {os.path.basename(target_path)}: {e}")


def upscale_lanczos(img, sx, sy):
    """Upscale image with Lanczos. img is (H,W,C), sx,sy scale factors."""
    h, w = img.shape[:2]
    nw, nh = int(round(w * sx)), int(round(h * sy))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)


def run_super_resolve(input_path, output_path, model_path, script_dir):
    """Run super-resolve.py run to produce 2x super-resolved image."""
    script = os.path.join(script_dir, "super-resolve.py")
    cmd = [
        sys.executable, script, "run",
        "--input", input_path,
        "--model_input", model_path,
        "--output", output_path,
    ]
    subprocess.run(cmd, check=True)


def _warp_perspective_gpu(img_tensor, H, out_h, out_w, device):
    """img_tensor: (C,H,W). H: 3x3 numpy (src->dst). Warp so dst is (out_h, out_w)."""
    H_inv = np.linalg.inv(H).astype(np.float32)
    h_src, w_src = img_tensor.shape[1], img_tensor.shape[2]
    # Build grid: for each (y,x) in output we need source (x_s, y_s) in pixel coords then normalize to [-1,1]
    yy = np.arange(out_h, dtype=np.float32)
    xx = np.arange(out_w, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")  # (H,W)
    ones = np.ones((out_h, out_w), dtype=np.float32)
    # dst homog: (3, H*W)
    dst = np.stack([grid_x, grid_y, ones], axis=0).reshape(3, -1)
    src = H_inv @ dst  # (3, H*W)
    x_s = (src[0] / (src[2] + 1e-8)).reshape(out_h, out_w)
    y_s = (src[1] / (src[2] + 1e-8)).reshape(out_h, out_w)
    # to normalized coords for grid_sample: [-1, 1]
    x_n = 2.0 * x_s / (w_src - 1) - 1.0
    y_n = 2.0 * y_s / (h_src - 1) - 1.0
    grid = np.stack([x_n, y_n], axis=-1)  # (H, W, 2)
    grid_t = torch.from_numpy(grid).to(device=device).unsqueeze(0)  # (1, H, W, 2)
    img_batch = img_tensor.unsqueeze(0)  # (1, C, H, W)
    out = F.grid_sample(img_batch, grid_t, mode="bilinear", padding_mode="reflection", align_corners=True)
    return out.squeeze(0)


def _warp_affine_gpu(img_tensor, M_2x3, out_h, out_w, device):
    """img_tensor: (C,H,W). M_2x3: affine (src->dst)."""
    # M maps [x_src, y_src, 1] -> [x_dst, y_dst]. We need for each dst (x,y) the src (x',y').
    M = np.vstack([M_2x3.astype(np.float32), [0, 0, 1]])
    M_inv = np.linalg.inv(M)
    M_inv_2x3 = M_inv[:2, :]  # (2, 3)
    yy = np.arange(out_h, dtype=np.float32)
    xx = np.arange(out_w, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")
    ones = np.ones((out_h, out_w), dtype=np.float32)
    dst = np.stack([grid_x, grid_y, ones], axis=0).reshape(3, -1)
    src = M_inv @ dst
    x_s = src[0].reshape(out_h, out_w)
    y_s = src[1].reshape(out_h, out_w)
    h_src, w_src = img_tensor.shape[1], img_tensor.shape[2]
    x_n = 2.0 * x_s / (w_src - 1) - 1.0
    y_n = 2.0 * y_s / (h_src - 1) - 1.0
    grid = np.stack([x_n, y_n], axis=-1)
    grid_t = torch.from_numpy(grid).to(device=device).unsqueeze(0)
    img_batch = img_tensor.unsqueeze(0)
    out = F.grid_sample(img_batch, grid_t, mode="bilinear", padding_mode="reflection", align_corners=True)
    return out.squeeze(0)


def align_akaze(ref_img, to_align_img):
    """Align to_align_img to ref_img using AKAZE + homography. Returns (aligned numpy image, 'homography', H) or (None, None, None)."""
    try:
        detector = cv2.AKAZE_create()
    except AttributeError:
        return None, None, None
    r_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    a_gray = cv2.cvtColor(to_align_img, cv2.COLOR_BGR2GRAY)
    kp1, desc1 = detector.detectAndCompute(r_gray, None)
    kp2, desc2 = detector.detectAndCompute(a_gray, None)
    if desc1 is None or desc2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None, None, None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    try:
        matches = bf.knnMatch(descs1=desc1, descs2=desc2, k=2)
    except Exception:
        return None, None, None
    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < 0.8 * n.distance:
            good.append(m)
    if len(good) < 4:
        return None, None, None
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None, None, None
    h, w = ref_img.shape[:2]
    aligned = cv2.warpPerspective(to_align_img, H, (w, h))
    return aligned, "homography", H


def align_ecc(ref_img, to_align_img, number_of_iterations=5000, termination_eps=1e-6):
    """Align to_align_img to ref_img using ECC. Returns (aligned numpy image, 'affine', M_2x3) or (None, None, None)."""
    r_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    a_gray = cv2.cvtColor(to_align_img, cv2.COLOR_BGR2GRAY)
    h, w = ref_img.shape[:2]
    M = np.eye(2, 3, dtype=np.float32)
    try:
        cc, M = cv2.findTransformECC(a_gray, r_gray, M, cv2.MOTION_EUCLIDEAN,
                                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                               number_of_iterations, termination_eps))
    except Exception:
        return None, None, None
    aligned = cv2.warpAffine(to_align_img, M, (w, h),
                             flags=cv2.INTER_LANCZOS4,
                             borderMode=cv2.BORDER_REFLECT_101)
    return aligned, "affine", M


def align_images_simple(images, method_order, ref_index, device):
    """Align each image to ref. When device is not cpu, run warping on GPU."""
    ref = images[ref_index]
    result = []
    use_gpu = device.type in ("cuda", "mps")
    for i in range(len(images)):
        if i == ref_index:
            result.append(ref)
            continue
        aligned_np, transform_type, transform = None, None, None
        for algo in method_order:
            if algo == "akaze":
                aligned_np, transform_type, transform = align_akaze(ref, images[i])
            elif algo == "ecc":
                aligned_np, transform_type, transform = align_ecc(ref, images[i])
            if aligned_np is not None:
                break
        if aligned_np is None:
            result.append(images[i])
            continue
        if use_gpu and transform is not None:
            try:
                h, w = ref.shape[:2]
                img_t = _numpy_to_tensor(images[i], device)
                if transform_type == "homography":
                    warped = _warp_perspective_gpu(img_t, transform, h, w, device)
                else:
                    warped = _warp_affine_gpu(img_t, transform, h, w, device)
                # keep dtype of original image when coming back
                aligned_np = _tensor_to_numpy(warped, dtype=images[i].dtype)
            except Exception as e:
                tlog(f"GPU warp failed, using CPU alignment for image {i}: {e}")
        result.append(aligned_np)
    return result


def detect_blend_type(dirname):
    """Infer blend type from directory name (e.g. coll-mean, coll-hdr)."""
    name = os.path.basename(dirname).lower()
    if "mean" in name:
        return "mean"
    if "hdr" in name:
        return "hdr"
    if "focus" in name:
        return "focus"
    if "min" in name:
        return "min"
    if "median" in name:
        return "median"
    if "max" in name:
        return "max"
    return "mean"  # default


def correct_ghosting(images, variance_percentile=95.0, device=None):
    """
    Post-alignment ghosting correction: at pixels where variance across the stack
    is high (moving objects / misalignment), replace values with the median.
    Uses GPU when device is cuda/mps. Returns list of images (same order).
    """
    if device is None:
        device = get_device()
    use_gpu = device.type in ("cuda", "mps")
    if use_gpu:
        try:
            # (N, C, H, W) on device, float [0,1]
            tensors = [_numpy_to_tensor(im, device) for im in images]
            stack = torch.stack(tensors, dim=0)
            var = stack.var(dim=0)  # (C, H, W)
            var_pixel, _ = var.max(dim=0)  # (H, W)
            q = variance_percentile / 100.0
            try:
                thresh = torch.quantile(var_pixel.flatten(), q).item()
            except Exception:
                # Fallback to NumPy percentile on CPU if torch.quantile is unsupported
                thresh = float(np.percentile(var_pixel.cpu().numpy(), variance_percentile))
            ghost_mask_2d = var_pixel > thresh  # (H, W)
            ghost_mask = ghost_mask_2d.unsqueeze(0).expand(stack.shape[1], -1, -1)  # (C, H, W)
            med = stack.median(dim=0).values
            out = []
            for i in range(len(images)):
                img = stack[i].clone()
                img[ghost_mask] = med[ghost_mask]
                img = img.clamp(0, 1)
                dtype = images[i].dtype
                if dtype == np.uint16:
                    out.append(_tensor_to_numpy(img, dtype=np.uint16))
                else:
                    out.append(_tensor_to_numpy(img, dtype=np.uint8))
            return out
        except Exception as e:
            tlog(f"GPU ghosting failed, falling back to CPU: {e}")
    # CPU path – preserve original bit depth
    dtype = images[0].dtype
    stack = np.stack(images, axis=0).astype(np.float64)
    var = np.var(stack, axis=0)
    var_pixel = np.max(var, axis=-1)
    thresh = np.percentile(var_pixel, variance_percentile)
    ghost_mask_2d = var_pixel > thresh
    ghost_mask = np.broadcast_to(ghost_mask_2d[:, :, np.newaxis], stack.shape[1:])
    med = np.median(stack, axis=0)
    out = []
    maxv = 65535.0 if dtype == np.uint16 else 255.0
    for i in range(len(images)):
        img = stack[i].copy()
        img[ghost_mask] = med[ghost_mask]
        img = np.clip(img, 0.0, maxv)
        out.append(img.astype(dtype))
    return out


def _rolling_mean_blend(paths, device=None):
    """Compute mean of images by streaming one at a time (running average). Only two images in memory at once.
    paths: list of paths to image files (e.g. tmpdir al_/gh_ TIFFs). Returns float array [0,1] HWC."""
    if not paths:
        raise ValueError("_rolling_mean_blend requires at least one path")
    use_gpu = device is not None and device.type in ("cuda", "mps")
    n = 0
    running_mean = None
    for p in paths:
        im = load_image(p)
        if im is None:
            continue
        if im.dtype == np.uint16:
            f = im.astype(np.float64) / 65535.0
        else:
            f = im.astype(np.float64) / 255.0
        f = np.clip(f, 0.0, 1.0)
        if use_gpu:
            t = _numpy_to_tensor(im, device)  # CHW [0,1]
            if n == 0:
                running_mean = t.clone()
                n = 1
            else:
                n += 1
                running_mean.add_(t.sub(running_mean).div_(n))
            del t
        else:
            if n == 0:
                running_mean = f.copy()
                n = 1
            else:
                n += 1
                running_mean += (f - running_mean) / n
    if running_mean is None:
        raise ValueError("_rolling_mean_blend: no valid image loaded")
    if use_gpu:
        out = running_mean.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    else:
        out = np.clip(running_mean, 0.0, 1.0)
    return out


def _stack_blend(images, kind="mean", device=None):
    """kind: mean, min, median, max. Uses GPU when device is cuda/mps. Returns float array in [0,1]."""
    if device is None:
        device = get_device()
    use_gpu = device.type in ("cuda", "mps")
    if use_gpu:
        try:
            tensors = [_numpy_to_tensor(im, device) for im in images]
            stack = torch.stack(tensors, dim=0)
            if kind == "mean":
                out = stack.mean(dim=0)
            elif kind == "min":
                out = stack.min(dim=0).values
            elif kind == "median":
                out = stack.median(dim=0).values
            elif kind == "max":
                out = stack.max(dim=0).values
            else:
                out = stack.mean(dim=0)
            out = out.clamp(0, 1)
            return out.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC in [0,1]
        except Exception as e:
            tlog(f"GPU blend '{kind}' failed, falling back to CPU: {e}")
    # CPU path: normalize by bit depth, operate in [0,1]
    stack_np = np.stack(images, axis=0)
    if stack_np.dtype == np.uint16:
        stack = stack_np.astype(np.float64) / 65535.0
    else:
        stack = stack_np.astype(np.float64) / 255.0
    if kind == "mean":
        out = np.mean(stack, axis=0)
    elif kind == "min":
        out = np.min(stack, axis=0)
    elif kind == "median":
        out = np.median(stack, axis=0)
    elif kind == "max":
        out = np.max(stack, axis=0)
    else:
        out = np.mean(stack, axis=0)
    return np.clip(out, 0.0, 1.0)


def float_to_16bit_tiff(arr):
    """Convert [0,1] float array to 16-bit for TIFF (0-65535)."""
    return (np.clip(arr, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)


def run_enfuse_hdr(input_paths, output_path):
    """Run enfuse for HDR with -d 16 and given weights."""
    cmd = [
        "enfuse", "-d", "16",
        "--exposure-weight=1", "--entropy-weight=1",
        "--saturation-weight=0.3", "--contrast-weight=0.4",
        "-o", output_path,
    ] + input_paths
    subprocess.run(cmd, check=True)


def run_enfuse_focus(input_paths, output_path):
    """Run enfuse for focus stacking (16-bit output)."""
    cmd = [
        "enfuse", "-d", "16", "-l", "29", "--hard-mask",
        "--saturation-weight=0", "--entropy-weight=1", "--contrast-weight=1",
        "--exposure-weight=0.4", "--gray-projector=l-star", "--contrast-edge-scale=0.3",
        "-o", output_path,
    ] + input_paths
    subprocess.run(cmd, check=True)


def process_coll_dir(coll_dir, scale_x, scale_y, model_path, align_order, ghosting, mode_override, script_dir, device, threads, keep_tmp):
    """Process one coll-* directory and write output TIFF. Intermediates go to /tmp/iq-{basefilename}."""
    files = image_files_in_dir(coll_dir)
    if not files:
        print(f"[{os.path.basename(coll_dir)}] No images, skip.")
        return
    base_path = files[0]
    base_name = os.path.splitext(os.path.basename(base_path))[0]
    blend_type = mode_override if mode_override else detect_blend_type(coll_dir)
    align_used = align_order[0] if align_order else "none"

    tmpdir = _tmpdir_for_base(base_name)
    os.makedirs(tmpdir, exist_ok=True)

    # Load and upscale; write upscaled images to tmpdir to avoid I/O on (possibly network) photo dir
    tlog(f"{os.path.basename(coll_dir)}: start upscaling")
    scale_is_two = (abs(scale_x - 2.0) < 1e-6 and abs(scale_y - 2.0) < 1e-6)
    use_super_resolve = scale_is_two and model_path and os.path.isfile(model_path)

    def _upscale_one(idx, path):
        img = load_image(path)
        if img is None:
            tlog(f"{os.path.basename(coll_dir)}: warning: could not read {os.path.basename(path)}")
            return
        tlog(f"{os.path.basename(coll_dir)}: upscaling {os.path.basename(path)}")
        if use_super_resolve:
            # super-resolve.py currently operates in 8-bit; immediately expand
            # its output back to 16-bit for all downstream processing.
            tmp_in = os.path.join(tmpdir, f"_sr_in_{idx:04d}.tif")
            tmp_out = os.path.join(tmpdir, f"_sr_out_{idx:04d}.tif")
            cv2.imwrite(tmp_in, img)
            run_super_resolve(tmp_in, tmp_out, model_path, script_dir)
            up = load_image(tmp_out)
            if up is not None:
                if up.dtype == np.uint8:
                    up = (up.astype(np.uint16) * 257)  # 0-255 -> full 0-65535 ladder
                up_path = os.path.join(tmpdir, f"up_{idx:04d}.tif")
                cv2.imwrite(up_path, up)
        else:
            up = upscale_lanczos(img, scale_x, scale_y)
            if up.dtype == np.uint8:
                up = (up.astype(np.uint16) * 257)
            up_path = os.path.join(tmpdir, f"up_{idx:04d}.tif")
            cv2.imwrite(up_path, up)

    with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
        for idx, f in enumerate(files):
            ex.submit(_upscale_one, idx, f)

    images = _read_images_from_tmpdir(tmpdir, "up_")
    if not images:
        print(f"[{os.path.basename(coll_dir)}] No images in tmpdir, skip.")
        shutil.rmtree(tmpdir, ignore_errors=True)
        return

    # Align (all to first); warping uses GPU when available; write aligned to tmpdir
    tlog(f"{os.path.basename(coll_dir)}: start aligning")
    if len(align_order) > 0:
        aligned = align_images_simple(images, align_order, ref_index=0, device=device)
    else:
        aligned = images
    for i, im in enumerate(aligned):
        cv2.imwrite(os.path.join(tmpdir, f"al_{i:04d}.tif"), _to_u16(im))

    # Ghosting correction (separate step after alignment; read/write tmpdir)
    if ghosting and len(aligned) > 1:
        tlog(f"{os.path.basename(coll_dir)}: start ghosting")
        aligned = correct_ghosting(aligned, device=device)
        for i, im in enumerate(aligned):
            cv2.imwrite(os.path.join(tmpdir, f"gh_{i:04d}.tif"), _to_u16(im))
        blend_prefix = "gh_"
    else:
        blend_prefix = "al_"

    # Blend (uses GPU when available); final output first goes to tmpdir, then is copied to coll_dir
    tlog(f"{os.path.basename(coll_dir)}: start blending")
    out_name = f"{blend_type}_{align_used}_{base_name}.tiff"
    tmp_out_path = os.path.join(tmpdir, out_name)
    final_out_path = os.path.join(coll_dir, out_name)

    if blend_type == "mean":
        # Rolling average: stream one image at a time; only two in memory (current + running mean)
        paths = _paths_in_tmpdir(tmpdir, blend_prefix)
        out_float = _rolling_mean_blend(paths, device=device)
        out_16 = float_to_16bit_tiff(out_float)
        cv2.imwrite(tmp_out_path, out_16)
    elif blend_type in ("min", "median", "max"):
        aligned = _read_images_from_tmpdir(tmpdir, blend_prefix)
        out_float = _stack_blend(aligned, kind=blend_type, device=device)
        out_16 = float_to_16bit_tiff(out_float)
        cv2.imwrite(tmp_out_path, out_16)
    elif blend_type == "hdr":
        aligned = _read_images_from_tmpdir(tmpdir, blend_prefix)
        paths = []
        for i, im in enumerate(aligned):
            p = os.path.join(tmpdir, f"enfuse_hdr_{i:04d}.tif")
            cv2.imwrite(p, _to_u16(im))
            paths.append(p)
        run_enfuse_hdr(paths, tmp_out_path)
    elif blend_type == "focus":
        aligned = _read_images_from_tmpdir(tmpdir, blend_prefix)
        paths = []
        for i, im in enumerate(aligned):
            p = os.path.join(tmpdir, f"enfuse_focus_{i:04d}.tif")
            cv2.imwrite(p, _to_u16(im))
            paths.append(p)
        run_enfuse_focus(paths, tmp_out_path)

    # Copy EXIF from base into tmp result, then copy final result into the real output directory
    try:
        _copy_exif_from_base(base_path, tmp_out_path)
        shutil.copy2(tmp_out_path, final_out_path)
        tlog(f"{os.path.basename(coll_dir)}: wrote {out_name}")
        print(f"[{os.path.basename(coll_dir)}] -> {out_name}")
    finally:
        # Clean up temporary directory for this collection unless --keep was requested
        if keep_tmp:
            tlog(f"{os.path.basename(coll_dir)}: keeping tmpdir {tmpdir}")
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    ap = argparse.ArgumentParser(description="Blend images in coll-* dirs: upscale, align, blend.")
    ap.add_argument("--scale", type=str, default="2",
                    help="Scale factor: single number or 'width,height' (e.g. 2 or 1.5,2)")
    ap.add_argument("--model", type=str, default=None,
                    help="Path to super-resolve model; used only when scale is 2")
    ap.add_argument("--align", type=str, default="akaze,ecc",
                    help="Alignment methods to try, comma-separated (e.g. akaze,ecc)")
    ap.add_argument("--ghosting", type=int, default=1, choices=[0, 1],
                    help="Enable(1)/disable(0) ghosting detection/correction after alignment (default 1)")
    ap.add_argument("--mode", type=str, default=None, dest="mode_override",
                    choices=["mean", "hdr", "focus", "min", "median", "max"],
                    help="Override blend mode (otherwise inferred from directory name)")
    ap.add_argument("--threads", type=int, default=3,
                    help="Number of threads for the upscaling phase (default 3)")
    ap.add_argument("--keep", action="store_true",
                    help="Keep per-coll tmp dirs (upscaled/aligned/deghosted) instead of deleting them")
    args = ap.parse_args()

    scale_x, scale_y = parse_scale(args.scale)
    align_order = [a.strip().lower() for a in args.align.split(",") if a.strip()]
    device = get_device()
    print(f"Using device: {device}", file=sys.stderr)
    tlog("started")
    coll_dirs = sorted(glob.glob(os.path.join(os.getcwd(), "coll-*")))
    if not coll_dirs:
        coll_dirs = sorted(glob.glob("coll-*"))
    for d in coll_dirs:
        if os.path.isdir(d):
            process_coll_dir(d, scale_x, scale_y, args.model, align_order, args.ghosting == 1,
                             args.mode_override, script_dir, device, args.threads, args.keep)


if __name__ == "__main__":
    main()
