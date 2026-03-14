#!/usr/bin/env python

import argparse
import os
import cv2
import numpy as np
import torch
from PIL import Image

# Allow large TIFFs when using Pillow
Image.MAX_IMAGE_PIXELS = None
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from collections import deque
import time

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# Neural Network Definition
# ==========================================

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class ArtifactRemovalNet(nn.Module):
    def __init__(self, num_channels=3, num_features=64, num_blocks=8):
        super(ArtifactRemovalNet, self).__init__()
        nf = num_features

        self.head = nn.Sequential(nn.Conv2d(num_channels, nf, 3, 1, 1), nn.ReLU(True))

        # Encoder: 4 levels (1/2, 1/4, 1/8, 1/16) for multi-scale artifact context
        self.enc1 = nn.Sequential(nn.Conv2d(nf, nf*2, 3, 2, 1), nn.ReLU(True), ResidualBlock(nf*2))
        self.enc2 = nn.Sequential(nn.Conv2d(nf*2, nf*4, 3, 2, 1), nn.ReLU(True), ResidualBlock(nf*4))
        self.enc3 = nn.Sequential(nn.Conv2d(nf*4, nf*8, 3, 2, 1), nn.ReLU(True), ResidualBlock(nf*8))
        self.enc4 = nn.Sequential(nn.Conv2d(nf*8, nf*16, 3, 2, 1), nn.ReLU(True), ResidualBlock(nf*16))

        # Bottleneck at 1/16 scale
        self.bottleneck = nn.Sequential(*[ResidualBlock(nf*16) for _ in range(max(1, num_blocks//2))])
        # Damp the coarsest scale so it contributes but doesn't dominate (learnable, init 0.5)
        self.coarse_scale = nn.Parameter(torch.tensor(0.5))

        # Decoder
        self.dec4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(nf*16 + nf*16, nf*8, 3, 1, 1), nn.ReLU(True), ResidualBlock(nf*8))
        self.dec3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(nf*8 + nf*8, nf*4, 3, 1, 1), nn.ReLU(True), ResidualBlock(nf*4))
        self.dec2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(nf*4 + nf*4, nf*2, 3, 1, 1), nn.ReLU(True), ResidualBlock(nf*2))
        self.dec1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(nf*2 + nf*2, nf, 3, 1, 1), nn.ReLU(True), ResidualBlock(nf))

        self.tail = nn.Sequential(nn.Conv2d(nf + nf, nf, 3, 1, 1), nn.ReLU(True), nn.Conv2d(nf, num_channels, 3, 1, 1))

    def forward(self, x):
        x_head = self.head(x)
        e1 = self.enc1(x_head)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)
        b_scaled = self.coarse_scale * b
        d4 = self.dec4(torch.cat([b_scaled, e4], 1))
        d3 = self.dec3(torch.cat([d4, e3], 1))
        d2 = self.dec2(torch.cat([d3, e2], 1))
        d1 = self.dec1(torch.cat([d2, e1], 1))
        return self.tail(torch.cat([d1, x_head], 1))

# ==========================================
# Processing Logic
# ==========================================

def load_image(path):
    """Load with Pillow; return BGR numpy uint16 or uint8 (no truncation)."""
    with Image.open(path) as im:
        if im.mode in ("I;16", "I;16B", "I;16L"):
            im = im.convert("I;16")
            arr = np.array(im, dtype=np.uint16)
        elif im.mode in ("RGB", "RGBA"):
            arr = np.array(im)
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
        else:
            im = im.convert("RGB")
            arr = np.array(im)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
    return arr[:, :, ::-1].copy()  # RGB -> BGR


def to_float(img):
    """uint16 or uint8 -> float [0,1]. Call only when ready to feed the net."""
    if img.dtype == np.uint16:
        return (img.astype(np.float32) / 65535.0).clip(0.0, 1.0)
    return (img.astype(np.float32) / 255.0).clip(0.0, 1.0)


def add_chroma_noise(img, n=50, amount=0.01):
    if n <= 0: return img
    h, w, c = img.shape
    flat = img.reshape(-1, c)
    idx = np.random.choice(h*w, max(1, h*w//n), replace=False)
    ch = np.random.randint(0, c, len(idx))
    noise = (np.random.rand(len(idx))-0.5)*2*amount
    flat[idx, ch] = np.clip(flat[idx, ch] + noise, 0, 1)
    return flat.reshape(h, w, c)

def extract_windows(img, ws):
    h, w = img.shape[:2]
    pad_h = (ws - h % ws) % ws
    pad_w = (ws - w % ws) % ws
    
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            
    h_new, w_new = img.shape[:2]
    windows = []
    
    for y in range(0, h_new, ws):
        for x in range(0, w_new, ws):
            windows.append(img[y:y+ws, x:x+ws])
            
    return windows

# ==========================================
# Main Logic
# ==========================================


def train_mode(args):
    filepath = args.file
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)

    model = ArtifactRemovalNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss()
    scaler = GradScaler('cuda')

    if os.path.exists(args.model_output):
        print(f"Resuming from {args.model_output}")
        ckpt = torch.load(args.model_output)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])

    model.train()

    gt_raw = load_image(filepath)
    gt = to_float(gt_raw)
    del gt_raw
    h_full, w_full = gt.shape[:2]
    h_tgt, w_tgt = h_full // 2, w_full // 2
    tgt = cv2.resize(gt, (w_tgt, h_tgt), interpolation=cv2.INTER_LANCZOS4)
    qrt = cv2.resize(gt, (w_tgt // 2, h_tgt // 2), interpolation=cv2.INTER_LANCZOS4)
    qrt = add_chroma_noise(qrt, args.noise_freq, args.noise_amount)
    inp = cv2.resize(qrt, (w_tgt, h_tgt), interpolation=cv2.INTER_LANCZOS4)
    inp_patches = extract_windows(inp, args.window_size)
    tgt_patches = extract_windows(tgt, args.window_size)
    n_patches = len(inp_patches)
    batches_per_epoch = (n_patches + args.batch_size - 1) // args.batch_size
    max_iter = args.max_iterations if args.max_iterations > 0 else float('inf')

    print(f"Training on single image: {os.path.basename(filepath)}")
    print(f"Config: {n_patches} windows, {batches_per_epoch} batches/epoch | Min {args.min_iterations} | Max {max_iter} | Stop {args.stop}% | Window {args.window_size}")
    print("Stop/max checked only after a full epoch (every pixel seen once).")

    iteration_counter = 0
    loss_history = deque(maxlen=10)
    start_time = time.time()
    epoch = 0

    while True:
        indices = np.arange(n_patches)
        np.random.shuffle(indices)
        inp_perm = [inp_patches[i] for i in indices]
        tgt_perm = [tgt_patches[i] for i in indices]

        pbar = tqdm(range(0, n_patches, args.batch_size), desc=f"Epoch {epoch}", leave=False,
                    bar_format='{desc} {bar} {postfix}')

        for i in pbar:
            batch_inp = inp_perm[i:i + args.batch_size]
            batch_tgt = tgt_perm[i:i + args.batch_size]
            batch_inp_t = torch.from_numpy(np.stack([p.astype(np.float32) for p in batch_inp])).permute(0, 3, 1, 2).to(device)
            batch_tgt_t = torch.from_numpy(np.stack([p.astype(np.float32) for p in batch_tgt])).permute(0, 3, 1, 2).to(device)

            optimizer.zero_grad()
            with autocast('cuda'):
                res = model(batch_inp_t)
                loss = criterion(res, batch_tgt_t - batch_inp_t)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            current_loss = loss.item()
            loss_history.append(current_loss)
            iteration_counter += 1

            stop_pct_val = 0.0
            if len(loss_history) >= 10:
                hist = list(loss_history)
                avg_prev = sum(hist[:5]) / 5.0
                avg_recent = sum(hist[5:]) / 5.0
                if avg_prev > 1e-9:
                    stop_pct_val = abs(avg_recent - avg_prev) / avg_prev * 100.0
            elapsed = time.time() - start_time
            speed = iteration_counter / elapsed if elapsed > 0 else 0
            pbar.set_postfix_str(f"Loss {current_loss:.5f} | Stop {stop_pct_val:.2f}% | {speed:.1f} it/s")

            if iteration_counter % args.save_interval == 0:
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scaler': scaler.state_dict()}, args.model_output)
                tqdm.write(f" > Model saved ({iteration_counter} iters)")

        # Only after a full epoch: allow max_iter or early-stop
        if iteration_counter >= max_iter:
            print("\nMax iterations reached.")
            break
        if iteration_counter >= args.min_iterations and args.stop > 0 and len(loss_history) >= 10:
            hist = list(loss_history)
            avg_prev = sum(hist[:5]) / 5.0
            avg_recent = sum(hist[5:]) / 5.0
            if avg_prev > 1e-9:
                change_pct = abs(avg_recent - avg_prev) / avg_prev * 100.0
                if change_pct <= args.stop:
                    tqdm.write(f"\nStopping early after full pass: {change_pct:.4f}% <= {args.stop}% tolerance.")
                    break
        epoch += 1

    final_loss = loss_history[-1] if loss_history else 0
    print(f"Training complete. Iterations: {iteration_counter}, Final Loss: {final_loss:.5f}")

    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler': scaler.state_dict()}, args.model_output)

def run_mode(args):
    model = ArtifactRemovalNet().to(device)
    ckpt = torch.load(args.model_input)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    img_raw = load_image(args.input)
    is_16bit = img_raw.dtype == np.uint16
    img = to_float(img_raw)
    del img_raw

    h, w = img.shape[:2]
    up = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

    ws = args.window_size
    ph, pw = (ws - up.shape[0] % ws) % ws, (ws - up.shape[1] % ws) % ws
    if ph > 0 or pw > 0:
        up = cv2.copyMakeBorder(up, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
    hn, wn = up.shape[:2]

    out = np.zeros((hn, wn, 3), dtype=np.float32)

    print("Running inference...")
    with torch.no_grad():
        with autocast('cuda'):
            for y in tqdm(range(0, hn, ws), desc="Enhancing"):
                for x in range(0, wn, ws):
                    patch = up[y:y + ws, x:x + ws]
                    bt = torch.from_numpy(patch.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
                    res = model(bt).squeeze(0).permute(1, 2, 0).cpu().numpy()
                    corrected = np.clip(patch.astype(np.float32) + res, 0.0, 1.0)
                    out[y:y + ws, x:x + ws] = corrected

    out_crop = out[: h * 2, : w * 2]
    if is_16bit:
        cv2.imwrite(args.output, (np.clip(out_crop, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16))
    else:
        cv2.imwrite(args.output, (out_crop * 255.0 + 0.5).astype(np.uint8))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode")
    
    noise_p = argparse.ArgumentParser(add_help=False)
    noise_p.add_argument("--noise_freq", type=int, default=50)
    noise_p.add_argument("--noise_amount", type=float, default=0.01)
    
    t = sub.add_parser("train", parents=[noise_p])
    t.add_argument("--model_output", required=True)
    t.add_argument("--window_size", type=int, default=32)
    
    t.add_argument("--min-iterations", type=int, default=0)
    t.add_argument("--max-iterations", type=int, default=0)
    t.add_argument("--stop", type=float, default=0.0)
    
    t.add_argument("--batch_size", type=int, default=64)
    t.add_argument("--save_interval", type=int, default=100)
    t.add_argument("file", help="Single image to train on (one run = one file; use resume for chaining)")
    
    r = sub.add_parser("run")
    r.add_argument("--input", required=True)
    r.add_argument("--model_input", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--window_size", type=int, default=32)
    
    args = parser.parse_args()
    
    if args.mode == "train": train_mode(args)
    elif args.mode == "run": run_mode(args)

