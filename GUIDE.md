# 📖 image Turbo Pro — User Guide

> **Everything you need to know to use image Turbo Pro, step by step.**
> Written for beginners — no coding experience required.

---

## 📑 Table of Contents

| Section | What You'll Learn |
|---------|-------------------|
| [🤔 What Is This?](#-what-is-this) | Basic overview in plain English |
| [💻 What You Need](#-what-you-need) | Requirements before starting |
| [🚀 Getting Started](#-getting-started) | Opening the notebook in Google Colab |
| [🔧 Step 1 — Initialize](#-step-1--initialize) | Setting up the environment |
| [🚀 Step 2 — Load Engine & Generate](#-step-2--load-engine--generate) | Loading models and creating images |
| [💾 Step 3 — Export](#-step-3--export) | Downloading your results |
| [🧹 Step 4 — Cache & Cleanup](#-step-4--cache--cleanup) | Managing Drive cache and freeing disk space |
| [🎛️ All Settings Explained](#️-all-settings-explained) | Every parameter in detail |
| [🖼️ Resolution Guide](#️-resolution-guide) | Choosing the right image size |
| [🎲 Sampler & Scheduler Guide](#-sampler--scheduler-guide) | What these do and which to pick |
| [✍️ Writing Good Prompts](#️-writing-good-prompts) | How to get better results |
| [📂 Where Are My Files?](#-where-are-my-files) | Finding downloaded models and outputs |
| [❓ FAQ](#-faq) | Answers to common questions |
| [🐛 Common Problems & Fixes](#-common-problems--fixes) | Troubleshooting guide |

---

## 🤔 What Is This?

**image Turbo Pro** is a tool that generates images using AI. You describe what you want in text, and the AI creates it.

Here's the simple version of how it works:

```
You type a description → AI reads it → AI generates an image → You download it
```

**What makes it special:**
- **FP8 optimized** — Uses 8-bit floating point quantization, cutting VRAM usage nearly in half while keeping full quality
- **Free GPU** — It runs on Google's computers (via Google Colab), so you don't need an expensive graphics card
- **Smart cache** — Models are downloaded once and cached in Google Drive. Subsequent sessions load from Drive in ~30 seconds instead of re-downloading ~7 GB
- **No installation** — Everything runs in your web browser. Just open the notebook and click play

### What Is a "Notebook"?

A **notebook** (also called a Jupyter notebook or Colab notebook) is an interactive document that contains code. Think of it like a step-by-step recipe:

1. You run **Cell 1** → it sets everything up, mounts Drive, and downloads models
2. You run **Cell 2** → it loads the AI and generates your image
3. You run **Cell 3** → it downloads the result
4. You run **Cell 4** (optional) → manage cache and free disk space

Each cell is a box of code. You click the **▶ Play button** on each cell to run it, one at a time, in order.

### What Is Google Colab?

**Google Colab** (short for "Colaboratory") is a free service from Google that lets you run code in your browser. It gives you:
- A **free GPU** (graphics card) — needed to generate images fast
- **Python** pre-installed — no setup needed
- **Temporary storage** — files last for the session

Think of it as a free computer in the cloud that you access through your browser.

### What Is FP8?

**FP8** (8-bit floating point) is a way of storing AI models in less space. Think of it like compressing a photo:

- **Full precision** = huge file, best quality, needs lots of VRAM
- **FP8** = half the file size, nearly identical quality, fits on a free T4 GPU

image Turbo Pro uses FP8, which means you can run it on Google Colab's **free** T4 GPU without running out of memory.

---

## 💻 What You Need

| Requirement | Details |
|-------------|---------|
| **A Google account** | Free — sign up at [accounts.google.com](https://accounts.google.com) |
| **A web browser** | Chrome, Firefox, Edge, Safari — any modern browser |
| **Internet connection** | Stable connection for downloading models (~8 GB total) |
| **Patience (first run)** | First time takes ~8 minutes for downloads. After that, it's fast. |

> ⚠️ **You do NOT need:**
> - A graphics card on your own computer
> - Python installed
> - Any coding knowledge
> - Any paid software

---

## 🚀 Getting Started

### Opening the Notebook

1. **Click this link** (or the "Open in Colab" button in the README):

   👉 [Open ImageUpscaler in Google Colab](https://colab.research.google.com/github/festverse/Image-Upscaler/blob/main/notebook/ImageUpscaler.ipynb)

2. **Sign in** with your Google account if prompted.

3. You'll see the notebook open in Google Colab. It looks like a document with boxes (cells) of code.

### Setting Up the GPU

Before running anything, you need to tell Colab to use a GPU:

1. Click **Runtime** in the top menu bar
2. Click **Change runtime type**
3. Under "Hardware accelerator", select **T4 GPU**
4. Click **Save**

> 💡 **Why do I need a GPU?**
> A GPU (Graphics Processing Unit) is a special chip that can do many calculations at once. AI image generation requires massive parallel computation — a regular CPU would take 30+ minutes per image, while a GPU does it in seconds.

---

## 🔧 Step 1 — Initialize

### What This Step Does

This cell prepares everything behind the scenes:

1. **Downloads the ImageUpscaler code** — the project files (this repository)
2. **Downloads ComfyUI** — the AI engine that actually generates images
3. **Installs Python libraries** — the software dependencies needed to run
4. **Installs aria2c** — a fast download tool (downloads files 16x faster than normal)
5. **Downloads 3 AI models:**

| Model | Size | What It Does |
|-------|------|-------------|
| **image Turbo FP8** | ~4 GB | The main image generator — this is the core AI |
| **Qwen 3 4B** | ~2.5 GB | Text encoder — reads and understands your prompts |
| **VAE (ae.safetensors)** | ~300 MB | Converts AI output into a viewable image |

### How to Run It

1. **Click on the first code cell** — it's labeled "🛠️ 1. Initialize"
2. **Click the ▶ Play button** on the left side of the cell (or press `Ctrl+Enter`)
3. **Wait** — you'll see text output showing progress. This takes about **5–8 minutes** the first time.
4. **Wait for the green checkmark** ✅ — it appears when the cell finishes successfully

### What You'll See

```
🛠️ Step 1/3 — Initialize
  Set up core environment & download models

🚀 Initializing Core Architecture...
   ✓ Core Engine Cloned
📦 Installing Dependencies (This takes a moment)...
📥 Fetching Models...
   📥 Fetching: image-turbo-fp8-e4m3fn.safetensors...
   ✓ Saved as: image-turbo-fp8-e4m3fn.safetensors
   📥 Fetching: qwen_3_4b.safetensors...
   ✓ Saved as: qwen_3_4b.safetensors
   📥 Fetching: ae.safetensors...
   ✓ Saved as: ae.safetensors
✅ Environment Ready! Please load the engine below.
```

> ⚠️ **If you see a warning about restarting the runtime:**
> Click "Cancel" — you don't need to restart. The notebook will work fine.

> 💡 **Second run is instant!** Models are cached in Google Drive. If you run Cell 1 again in a new session, it copies models from Drive (~30s) instead of re-downloading (~5 min). First-time setup takes longer because it downloads ~7 GB of models.

---

## 🚀 Step 2 — Load Engine & Generate

### What This Step Does

This is the fun part! This cell:

1. **Loads all 3 models into GPU VRAM** — puts the AI brain into the graphics card (~30 seconds)
2. **Encodes your prompt** — the text encoder reads your description
3. **Generates the image** — the AI creates your image from noise (takes ~10-20 seconds)
4. **Decodes the result** — converts the AI output into a viewable image
5. **Displays the image** — shows it right in the notebook

### Filling In the Settings

Before running, you'll see several fields to fill in. Here's what each one means:

#### Prompt Fields

| Field | What to Type | Example |
|-------|-------------|---------|
| **positive_prompt** | Describe the image you want | `a cat sitting on a windowsill, rainy day, cozy` |
| **negative_prompt** | What to avoid in the image | `blurry, low quality, text, watermark, distorted` |

#### Image Settings

| Field | Default | What It Does |
|-------|---------|-------------|
| **aspect_ratio** | 16:9 Landscape | Shape of the image (dropdown with 5 presets) |
| **steps** | 20 | Quality/detail level (more = slightly better but slower) |
| **guidance_scale** | 1.0 | How closely the AI follows your prompt |

#### Advanced Settings

| Field | Default | What It Does |
|-------|---------|-------------|
| **seed** | -1 | Random number seed (-1 = random each time, any number = reproducible) |
| **auto_download** | False | Automatically download the image after generation |

### Running the Generation

1. **Fill in at least the positive_prompt** (the aspect_ratio and other fields have good defaults)
2. **Click the ▶ Play button** on the "🚀 2. Load Engine & Generate" cell
3. **Wait ~30 seconds** — you'll see progress messages
4. **Your image appears!** — displayed right below the cell

### What You'll See

```
🚀 Step 2/3 — Load Engine & Generate
  Load FP8 weights, configure prompt, and generate

🧠 Loading models into VRAM...
   ⏳ Loading Checkpoints into VRAM...
✓ Engine Online. Ready to Generate.
➜ Generating: 1280x720 | Steps: 20 | Seed: 48291
✓ Saved to: A_cinematic_shot_of_a_fut_4a2b.png
[Your image displayed here]
```

### Running Again

Want to generate another image? Just change the **positive_prompt** (or any other settings) and click ▶ again. Each run creates a new image.

> 💡 **Tip:** Change the **seed** to a specific number to get reproducible results. Same seed + same settings = same image.

---

## 💾 Step 3 — Export

### What This Step Does

This cell packages all your generated images into a `.zip` file and downloads it to your computer.

### How to Run It

1. **Click the ▶ Play button** on the "💾 3. Export" cell
2. **A download dialog appears** — choose where to save on your computer
3. **Done!** Your images are in the zip file

> 💡 **If the download doesn't start automatically:**
> - Check your browser's download bar
> - Or find the file at `/content/Z_Image_Pro_Artworks.zip` in Colab's file browser (folder icon on the left sidebar)

### Where Are My Generated Images?

All generated images are saved in:
```
/content/results/
```

You can browse them anytime using Colab's file browser (click the 📁 folder icon in the left sidebar).

---

## 🧹 Step 4 — Cache & Cleanup

### What This Step Does

This optional cell helps you manage disk space and the Google Drive model cache. Free-tier Colab has limited disk space, so this is useful when you're running low.

### How to Run It

1. **Click the ▶ Play button** on the "🧹 4. Cache & Cleanup" cell
2. **Review the status** — it shows disk space, output count, and cache size
3. **Optionally check the boxes:**
   - `clear_drive_cache` — Deletes all cached models from Drive (reclaims ~7 GB)
   - `clear_old_outputs` — Deletes generated images from `/content/results`
   - `keep_latest_images` — Set to N to keep your N most recent images

### When to Use It

| Situation | What to Do |
|-----------|-----------|
| "Low disk space" warning | Run Cell 4, check `clear_old_outputs` |
| Want to free Drive space | Check `clear_drive_cache` (next run re-downloads models) |
| Starting fresh | Check both boxes for a clean slate |
| Everything works fine | Skip this cell entirely |

> 💡 **Tip:** The Drive cache (~7 GB) persists across sessions. Only clear it if you need the space back — otherwise you'll have to re-download models next time.

---

## 🎛️ All Settings Explained

### Prompt Settings

| Setting | Type | Default | Explanation |
|---------|------|---------|-------------|
| `positive_prompt` | Text | *(example provided)* | **What you want to see.** Describe the image in detail. Be specific — "a watercolor painting of a mountain at sunset" is better than "mountain". |
| `negative_prompt` | Text | `blurry, low quality, text, watermark, distorted` | **What to avoid.** Tells the AI what NOT to include. The default works well for most cases. |

### Image Size Settings

| Setting | Type | Default | Range | Explanation |
|---------|------|---------|-------|-------------|
| `aspect_ratio` | Dropdown | 16:9 Landscape | 5 presets | Shape of the output image. Choose from Square, Landscape, Portrait, Photo, or Cinema. |
| `steps` | Slider | 20 | 10–50 | Number of denoising iterations. More steps = slightly more detail, but slower. 20 is the sweet spot for image Turbo. |
| `guidance_scale` | Slider | 1.0 | 1.0–10.0 | **Classifier-Free Guidance** — how strictly the AI follows your prompt. Lower = more creative freedom, higher = stricter adherence. 1.0 works well with image. |

### Advanced Settings

| Setting | Type | Default | Explanation |
|---------|------|---------|-------------|
| `seed` | Number | -1 | Random number generator seed. `-1` = different image each time. Any other number = reproducible. Same seed + same settings = same image. |
| `auto_download` | Checkbox | False | If checked, the image automatically downloads to your computer after generation. |

---

## 🖼️ Resolution Guide

Choose a resolution based on what you're creating:

| Aspect Ratio | Resolution | Best For |
|:---:|:---:|---|
| **1:1** | 1024 × 1024 | Social media posts, avatars, profile pictures, Instagram posts |
| **16:9** | 1280 × 720 | YouTube thumbnails, desktop wallpapers, presentations, landscape photos |
| **9:16** | 720 × 1280 | Phone wallpapers, Instagram/TikTok stories, vertical content |
| **4:3** | 1152 × 864 | Classic photography, print layouts, traditional aspect ratio |
| **21:9** | 1344 × 576 | Cinematic shots, ultrawide wallpapers, dramatic compositions |

> 💡 **Best quality:** Stick to 1024×1024 or close to it. The model was trained at this resolution.

---

## 🎲 Sampler & Scheduler Guide

### What Is a Sampler?

A **sampler** is the algorithm that turns random noise into your image. Different samplers produce slightly different results.

### What Is a Scheduler?

A **scheduler** controls how the noise is removed during generation. Think of it as the "pace" of image formation.

### Recommended Combinations

| Use Case | Sampler | Scheduler | Why |
|----------|---------|-----------|-----|
| **Default / Fast** | `euler` | `simple` | Clean results, fast generation — the notebook default |
| **Best quality** | `res_multistep` | `beta` | Optimized for image — slightly better detail |
| **More variation** | `euler_ancestral` | `karras` | Each run looks more unique |
| **Maximum quality** | `dpmpp_3m_sde` | `beta` | Best detail, but 3x slower |

### Full Sampler List

| Sampler | Speed | Quality | Notes |
|---------|:---:|:---:|-------|
| `euler` | ⚡⚡⚡ | ⭐⭐⭐⭐ | **Default** — fast and clean |
| `res_multistep` | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | Best for image |
| `euler_ancestral` | ⚡⚡⚡ | ⭐⭐⭐⭐ | More randomness between runs |
| `dpmpp_2m` | ⚡⚡ | ⭐⭐⭐⭐⭐ | High quality, slower |
| `dpmpp_2m_sde` | ⚡⚡ | ⭐⭐⭐⭐⭐ | Excellent fine detail |
| `dpmpp_3m_sde` | ⚡ | ⭐⭐⭐⭐⭐ | Best quality, slowest |
| `lcm` | ⚡⚡⚡⚡ | ⭐⭐⭐ | Ultra-fast, lower quality |
| `ddim` | ⚡⚡⚡ | ⭐⭐⭐ | Deterministic, predictable |
| `uni_pc` | ⚡⚡ | ⭐⭐⭐⭐ | Good balance |

---

## ✍️ Writing Good Prompts

### The Golden Rule

> **Be specific.** The more detail you give, the better the result.

### Bad vs Good Prompts

| ❌ Bad | ✅ Good |
|--------|---------|
| `a cat` | `a fluffy orange tabby cat sleeping on a velvet cushion, warm afternoon sunlight through a window, soft bokeh background, photorealistic, 8k detail` |
| `landscape` | `majestic mountain range at golden hour, dramatic clouds, crystal clear lake reflection, pine forest, atmospheric perspective, national geographic style` |
| `anime girl` | `anime girl with long silver hair, blue eyes, school uniform, cherry blossom petals falling, sunset rooftop, studio ghibli style, beautiful detailed eyes` |

### Prompt Structure

A good prompt follows this pattern:

```
[Subject] + [Details] + [Setting/Background] + [Style] + [Quality]
```

**Example:**
```
cyberpunk woman with neon face paint        ← Subject + Details
standing in rain-soaked alley               ← Setting
purple and cyan neon lights, reflections    ← Atmosphere
blade runner style, cinematic portrait      ← Style
sharp focus, 8k                             ← Quality
```

### Useful Style Words

| Category | Words to Use |
|----------|-------------|
| **Art Style** | `oil painting`, `watercolor`, `digital art`, `anime`, `pixel art`, `pencil sketch`, `concept art` |
| **Photography** | `photorealistic`, `cinematic`, `portrait`, `landscape`, `macro`, `fujifilm simulation`, `canon 85mm f1.4` |
| **Lighting** | `golden hour`, `dramatic lighting`, `soft light`, `neon lights`, `god rays`, `candlelight`, `backlit` |
| **Quality** | `8k`, `ultra detailed`, `highly detailed`, `masterpiece`, `sharp focus`, `intricate details` |
| **Mood** | `moody`, `ethereal`, `cozy`, `epic`, `dreamy`, `dark`, `warm`, `mysterious` |

### Using Negative Prompts

The negative prompt tells the AI what to **avoid**. The default works well, but you can customize it:

**Default (works for most cases):**
```
blurry, low quality, text, watermark, distorted
```

**For portraits, add:**
```
extra fingers, bad anatomy, deformed face, ugly
```

**For landscapes, add:**
```
oversaturated, cartoon, text, watermark
```

**For anime style, add:**
```
3d render, realistic photo, bad anatomy, extra limbs
```

---

## 📂 Where Are My Files?

### In Google Colab

| What | Where | How to Find |
|------|-------|-------------|
| **image FP8 model** | `/content/ComfyUI/models/diffusion_models/` | 📁 File browser → ComfyUI → models → diffusion_models |
| **Text Encoder** | `/content/ComfyUI/models/clip/` | 📁 File browser → ComfyUI → models → clip |
| **VAE** | `/content/ComfyUI/models/vae/` | 📁 File browser → ComfyUI → models → vae |
| **Generated images** | `/content/results/` | 📁 File browser → content → results |
| **Downloaded zip** | `/content/Z_Image_Pro_Artworks.zip` | 📁 File browser → content |
| **Drive model cache** | `/content/drive/MyDrive/ImageUpscaler/models/` | 📁 File browser → Drive → MyDrive → ImageUpscaler → models |

### Opening the File Browser

1. Click the **📁 folder icon** on the left sidebar in Google Colab
2. Navigate through the folders to find your files
3. Right-click any file to download it to your computer

> ⚠️ **Important:** Colab storage is **temporary**. When your session ends, all files are deleted. Always download your generated images using Step 3 before closing!

---

## ❓ FAQ

<details>
<summary><b>Do I need a powerful computer?</b></summary>

No! The heavy computation runs on Google's free T4 GPU in the cloud. You just need a web browser and internet connection. Your computer can be a basic laptop, Chromebook, or even a tablet.
</details>

<details>
<summary><b>Is it really free?</b></summary>

Yes, Google Colab's free tier includes a T4 GPU. There are limits (session time, daily GPU hours), but for occasional image generation, the free tier is sufficient. If you need more, Colab Pro offers longer sessions and faster GPUs.
</details>

<details>
<summary><b>How long does it take to generate one image?</b></summary>

After the initial setup (first run only: ~8 minutes for downloads), each image takes about **20–30 seconds** to generate (including model loading). On subsequent sessions, models load from Drive cache (~30s) instead of downloading.
</details>

<details>
<summary><b>Can I use the generated images commercially?</b></summary>

The ImageUpscaler project is MIT licensed. However, check the license of the specific image model you use — it may have its own terms.
</details>

<details>
<summary><b>What happens if Colab disconnects?</b></summary>

Your session ends and temporary files are deleted. However, **models cached in Google Drive persist** — just re-run Cell 1 and it copies from Drive (~30s) instead of re-downloading (~5 min). Always download your generated images using Step 3 before closing!
</details>

<details>
<summary><b>Can I run this on my own computer instead of Colab?</b></summary>

Yes, but you need a GPU with 16GB+ VRAM (like an NVIDIA RTX 4060 or better), Python 3.10+, and ComfyUI installed. The `src/` modules can be used directly outside of Colab.
</details>

<details>
<summary><b>Why 20 steps? Can I use fewer?</b></summary>

20 steps provides excellent quality with the `euler` sampler. You can reduce to 10–15 for faster generation or increase to 30–50 for maximum detail. image Turbo converges quickly — going higher has diminishing returns.
</details>

<details>
<summary><b>How do I get the same image again?</b></summary>

Set the **seed** to a specific number (not -1). Same seed + same prompt + same settings = same image. The seed is shown in the output: `Seed: 48291`.
</details>

<details>
<summary><b>Is there a content filter?</b></summary>

No. image is an unfiltered model — it does not have built-in NSFW filters. Users are solely responsible for the content they generate. Do not use this tool to create illegal, harmful, or non-consensual content. By using this project, you agree to comply with all applicable laws and the [HuggingFace content policy](https://huggingface.co/content-guidelines). The authors assume no liability for misuse.
</details>

---

## 🐛 Common Problems & Fixes

### `CUDA out of memory`

**What it means:** The GPU ran out of memory.

**Fixes:**
- Reduce resolution (try 768×768 or use a smaller aspect ratio)
- Restart the runtime: **Runtime → Restart runtime**, then re-run all cells
- Make sure no other Colab notebooks are running

---

### `Module not found: comfyui`

**What it means:** ComfyUI wasn't installed properly.

**Fix:** Re-run **Cell 1 (Initialize)**. Make sure it completes without errors.

---

### `Module not found: src`

**What it means:** The ImageUpscaler repository wasn't cloned properly.

**Fix:** Re-run **Cell 1 (Initialize)**. It clones the repo automatically.

---

### No images appear after generation

**What it means:** The generation didn't complete.

**Fixes:**
- Make sure you see `✅ Engine Online. Ready to Generate.` in Cell 2's output
- Check that Cell 2 completed without errors
- Re-run Cell 2

---

### `Download failed` in Step 1

**What it means:** The model download timed out or the URL is invalid.

**Fixes:**
- Check your internet connection
- Re-run Cell 1 — it will resume partial downloads
- If it keeps failing, try a different network

---

### Image is black or garbled

**What it means:** The model didn't load correctly or there's a compatibility issue.

**Fixes:**
- Make sure you selected **T4 GPU** as the runtime type
- Re-run Cell 1 and Cell 2 from scratch
- Try a different seed value

---

### Colab says "Session disconnected"

**What it means:** Your session timed out or hit daily limits.

**Fixes:**
- Stay active — Colab disconnects after ~90 minutes of inactivity
- If you hit daily limits, wait 24 hours or upgrade to Colab Pro
- Always download your images promptly after generation

---

### Images look distorted or have artifacts

**What it means:** Settings may need adjustment.

**Fixes:**
- Try a different **seed** value
- Simplify your prompt
- Make sure you're using the default sampler (`euler`) and scheduler (`simple`)
- Try increasing **steps** to 30

---

## ⚠️ Content Safety Notice

image is an **unfiltered diffusion model** — it does not have built-in NSFW filters.

You are **solely responsible** for the content you generate.

**Do NOT** use this tool to create:
- Illegal content
- Harmful, abusive, or violent content
- Non-consensual intimate imagery
- Content involving minors in any form

By running this notebook, you agree to comply with all applicable laws and the [HuggingFace Content Guidelines](https://huggingface.co/content-guidelines). The authors assume no liability for misuse.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

**Made with ⚡ by [Utsav Vasava](https://github.com/festverse)**

[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/festverse/Image-Upscaler)
[![Telegram](https://img.shields.io/badge/-Telegram-2CA5E0?style=for-the-badge&logo=Telegram&logoColor=white)](https://telegram.me/festverse)

</div>
