# ======= • ======= • ======= • ======= • =======• =======
# image Pro — config.py
# Repository: https://github.com/festverse/Image-Upscaler
#
# @description
#   All configuration constants, default generation parameters,
#   model URLs, directory maps, and supported option lists.
#   Every tunable value the pipeline uses lives here.
#
# @exports
#   WORKSPACE, UNET_URL, TEXT_ENCODER_URL, VAE_URL,
#   MODEL_DIRS, DEFAULTS, RESOLUTIONS, SAMPLERS, SCHEDULERS
#
# @version 1.0.0
# @author  Utsav Vasava
# @license MIT
# ======= • ======= • ======= • ======= • =======• =======

import os

# ══════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════

# ---- FEATURE: Workspace root ----
# Notebook: ROOT = Path("/content"), COMFY_PATH = ROOT / "ComfyUI"
WORKSPACE = "/content/ComfyUI"

# ---- FEATURE: Google Drive cache root ----
# Models cached here persist across Colab session restarts.
# On first run: downloads to ComfyUI, then copies to Drive.
# On subsequent runs: copies from Drive to ComfyUI (skips download).
DRIVE_CACHE_DIR = "/content/drive/MyDrive/ImageUpscaler/models"

# ---- FEATURE: Cache versioning ----
# Bump this when model URLs or filenames change. Old cached models
# with a different version are treated as stale and re-downloaded.
CACHE_VERSION = "1"

# ---- FEATURE: Total model download size (approximate, for user messaging) ----
TOTAL_MODEL_SIZE_GB = 7.0

# ---- FEATURE: Disk space thresholds for free-tier safety ----
DISK_WARN_GB = 2.0   # Warn when less than this free
DISK_MIN_GB  = 1.0   # Refuse to download when less than this free

# ══════════════════════════════════════════════════════════════
# MODEL URLS
# ══════════════════════════════════════════════════════════════

# ---- FEATURE: Pre-configured model download URLs ----
# Notebook: model_map = [
#   ("https://huggingface.co/T5B/image-Turbo-FP8/resolve/main/image-turbo-fp8-e4m3fn.safetensors", ...),
#   ("https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors", ...),
#   ("https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors", ...),
# ]
UNET_URL = "https://huggingface.co/T5B/image-Turbo-FP8/resolve/main/image-turbo-fp8-e4m3fn.safetensors"
TEXT_ENCODER_URL = "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors"
VAE_URL = "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors"

# ══════════════════════════════════════════════════════════════
# DIRECTORY MAP
# ══════════════════════════════════════════════════════════════

# ---- FEATURE: Model subdirectories (relative to WORKSPACE) ----
# Notebook: MODELS_PATH / "diffusion_models", MODELS_PATH / "clip", MODELS_PATH / "vae"
MODEL_DIRS = {
    "unet": "models/diffusion_models",
    "clip": "models/clip",
    "vae":  "models/vae",
}

# ══════════════════════════════════════════════════════════════
# DEFAULT GENERATION PARAMETERS
# ══════════════════════════════════════════════════════════════

# ---- FEATURE: Sensible defaults for image generation ----
# All values extracted from notebook Cell 3 (Creator Studio) @param defaults
DEFAULTS = {
    "width":           1024,
    "height":          1024,
    "batch_size":      1,
    "steps":           20,       # notebook: steps = 20 #@param {type:"slider", min:10, max:50, step:1}
    "cfg":             1,        # notebook: guidance_scale = 1 #@param {type:"slider", min:1.0, max:10.0, step:0.5}
    "sampler_name":    "euler",  # notebook: KSampler(..., "euler", ...)
    "scheduler":       "simple", # notebook: KSampler(..., ..., "simple", ...)
    "seed":            -1,       # notebook: seed = -1 #@param {type:"number"}
    "negative_prompt": "blurry, low quality, text, watermark, distorted",
}

# ══════════════════════════════════════════════════════════════
# RESOLUTION PRESETS
# ══════════════════════════════════════════════════════════════

# ---- FEATURE: Aspect ratio → (width, height) mapping ----
# Notebook: aspect_ratio = "1280x720 (16:9 Landscape)" with 5 preset options
RESOLUTIONS = {
    "1:1":  (1024, 1024),   # "1024x1024 (1:1 Square)"
    "16:9": (1280, 720),    # "1280x720 (16:9 Landscape)"
    "9:16": (720, 1280),    # "720x1280 (9:16 Portrait)"
    "4:3":  (1152, 864),    # "1152x864 (4:3 Photo)"
    "21:9": (1344, 576),    # "1344x576 (21:9 Cinema)"
}

# ══════════════════════════════════════════════════════════════
# SUPPORTED SAMPLERS & SCHEDULERS
# ══════════════════════════════════════════════════════════════

# ---- FEATURE: Available KSampler options ----
SAMPLERS = [
    "euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
    "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral",
    "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu",
    "ddpm", "lcm", "ddim", "uni_pc", "uni_pc_bh2", "res_multistep",
]

# ---- FEATURE: Available scheduler options ----
SCHEDULERS = [
    "normal", "karras", "exponential", "sgm_uniform",
    "simple", "ddim_uniform", "beta",
]

# ══════════════════════════════════════════════════════════════
# EXPORTS
# ══════════════════════════════════════════════════════════════

__all__ = [
    "WORKSPACE", "DRIVE_CACHE_DIR", "CACHE_VERSION", "TOTAL_MODEL_SIZE_GB",
    "DISK_WARN_GB", "DISK_MIN_GB",
    "UNET_URL", "TEXT_ENCODER_URL", "VAE_URL",
    "MODEL_DIRS", "DEFAULTS", "RESOLUTIONS", "SAMPLERS", "SCHEDULERS",
]

# ══════════════════════════════════════════════════════════════ END: config.py
