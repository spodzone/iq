# SUPER-RESOLVE

`super-resolve.py` trains and runs a **U-net–style deep learning model** for **single-image super-resolution**. The network does not learn to invent high-frequency detail from nothing; instead it learns to **correct the artifacts introduced by Lanczos upscaling**, so that a Lanczos-upscaled image can be refined toward what a native high-resolution image would look like.
In practice, based on training on a few thousand real-world nature photographs, this beats ImageMagick's native upscale hands-down - where lanczos preserves sharpness to the point of spikiness, this algorithm anti-aliases edges for a smoother, more realistic and more pleasant look.

---

## What it does

- **Train:** Learn from full-resolution images by creating synthetic “low-res → Lanczos upscale” pairs and using the true mid-resolution image as ground truth. The model is trained to predict the **residual** (difference) between the Lanczos result and that ground truth.
- **Run:** Load an image, upscale it 2× with Lanczos, then apply the trained model in overlapping or tiled windows to add the learned correction. The result is a 2× super-resolved image with reduced Lanczos artifacts.

---

## How the training pipeline works

1. **Build a clean “ground truth” at half size**
   Each training image is downscaled to **50%** (using Lanczos). This half-size image is the **target** the model is trained to match after correction. This eliminates pixel-level artefacts in the source image.

2. **Create the “artifactful” input**
   The same image is downscaled to **25%**, then (optionally) **chroma noise** is applied. That 25% image is then **upscaled back to 50%** using Lanczos. This 50% Lanczos-upscaled image has typical upscaling artifacts (ringing, blur, etc.) but no extra real detail—it’s a degraded version of the 50% ground truth.

3. **Learn the residual**
   The network’s **input** is the Lanczos-upscaled 50% image. Its **target** is
   `ground_truth_50% − input`, i.e. the **residual** that, when added to the input, recovers the clean 50% image. So the model learns: “given a Lanczos-upscaled image, what correction do I add to get closer to reality?”

4. **Training in windows**
   Both the input and the ground truth are split into **non-overlapping windows** (e.g. 32×32). Training proceeds on batches of these windows, so the model sees many small patches per image and can generalize to arbitrary image sizes at run time.

At **run** time, you upscale the image 2× with Lanczos, then run the same model over windows of that upscaled image. The model’s output (the learned residual) is added to the Lanczos result and clipped to [0,1], giving the final super-resolved image. So the script is effectively learning “Lanczos upscaling artifacts vs. reality” and removing those artifacts.

---

## Command-line usage

The script has two subcommands: **`train`** and **`run`**. All options are described below:


### Train options

Use: `python super-resolve.py train [options] --model_output <path> file1 [file2 ...]`

- **Required**
  - **`--model_output`**
    Path where the checkpoint will be written (and read when resuming). The script saves the model state, optimizer state, and optional GradScaler state. If this file already exists, training **resumes** from it (same weights, optimizer, and scaler).

  - **`files`** (positional, one or more)
    List of image paths used for training. Each image is loaded, the 50% / 25% pipeline is applied, and windows are extracted. The script cycles through these files until stopping conditions are met.

- **Iteration control**
  - **`--min-iterations`** (int, default: `0`)
    Minimum number of training iterations (forward + backward passes) before the script is allowed to stop. If the file list is exhausted before this, the script restarts from the first file and keeps training until at least `min-iterations` is reached. Use this to ensure a minimum amount of learning before considering early stopping or “dataset exhausted.”

  - **`--max-iterations`** (int, default: `0`)
    Hard cap on the number of training iterations. `0` means no cap (train until early stop or dataset exhausted). Any positive value stops training as soon as the iteration count reaches it.

  - **`--stop`** (float, default: `0.0`)
    Early-stopping tolerance in percent. After `min_iterations`, every 10 iterations the script compares the average loss of the previous 5 iterations with the average of the 5 before that. If the relative change is **≤ `--stop`%**, training stops (“loss has stabilized within tolerance”). Set to `0` to disable early stopping.

- **Spatial / batching**
  - **`--window_size`** (int, default: `32`)
    Side length (in pixels) of the square windows used for training (and for run). Input and ground truth are tiled into non-overlapping `window_size × window_size` patches (with reflection padding if dimensions aren’t divisible). Must match at run time if you change it (same value should be used for `run --window_size`).

  - **`--batch_size`** (int, default: `64`)
    Number of windows per training step. Larger values use more GPU memory but can give more stable gradients. Reduce if you run out of memory.

- **Checkpointing**
  - **`--save_interval`** (int, default: `100`)
    Save a checkpoint to `--model_output` every this many iterations. Also saves once at the end of training.

- **Data augmentation (noise)**
  - **`--noise_freq`** (int, default: `50`)
    Controls how many pixels are eligible for chroma noise: one pixel per this many image pixels (approximate). Higher value → fewer pixels get noise; lower value → more. Set to `0` or negative to disable chroma noise.

  - **`--noise_amount`** (float, default: `0.01`)
    Magnitude of chroma noise. Each chosen pixel channel is perturbed by a value in `[−noise_amount, +noise_amount]` (then clipped to [0,1]). Larger values add stronger color noise to the 25% image before upscaling, which can improve robustness.

---

### Run options

Use: `python super-resolve.py run --input <path> --model_input <path> --output <path> [options]`

- **Required**
  - **`--input`**
    Path to the input image to super-resolve. Any format supported by OpenCV (e.g. PNG, JPEG) is fine.

  - **`--model_input`**
    Path to the saved checkpoint file (the same format written by `train --model_output`). Only the `model_state_dict` is loaded; optimizer and scaler are ignored.

  - **`--output`**
    Path where the super-resolved image will be written. The result is 2× the input width and height (same aspect ratio).

- **Spatial**
  - **`--window_size`** (int, default: `32`)
    Must match the window size used when training the model. The upscaled (2×) image is padded so its dimensions are divisible by `window_size`, then processed in non-overlapping windows. The model’s residual is added per window and the final crop is exactly 2× the original size.

---

## Option tree (quick reference)

```
super-resolve.py
├── train
│   ├── --model_output (required)
│   ├── files (required, positional, one or more)
│   ├── --window_size (default: 32)
│   ├── --min-iterations (default: 0)
│   ├── --max-iterations (default: 0)
│   ├── --stop (default: 0.0)
│   ├── --batch_size (default: 64)
│   ├── --save_interval (default: 100)
│   ├── --noise_freq (default: 50)
│   └── --noise_amount (default: 0.01)
│
└── run
    ├── --input (required)
    ├── --model_input (required)
    ├── --output (required)
    └── --window_size (default: 32)
```

---

## Example

**Train** for at least 2000 iterations, stop when loss change is under 1%, save every 200 steps:

```bash
python super-resolve.py train \
  --model_output model.pt \
  --min-iterations 2000 \
  --max-iterations 10000 \
  --stop 1.0 \
  --save_interval 200 \
  --window_size 32 \
  image1.png image2.png image3.png
```

**Run** inference on a single image:

```bash
python super-resolve.py run \
  --input lowres.png \
  --model_input model.pt \
  --output super.png \
  --window_size 32
```

`super.png` will be twice the width and height of `lowres.png`, with Lanczos artifacts reduced by the learned residual.
