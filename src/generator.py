# ======= • ======= • ======= • ======= • =======• =======
# image Pro — generator.py
# Repository: https://github.com/festverse/Image-Upscaler
#
# @description
#   ComfyUI node-based image generator. Loads UNet, CLIP, and
#   VAE models in-process, encodes prompts, samples latents
#   via KSampler, and decodes final images. Operates directly
#   in the notebook environment without a server — lightweight
#   and fast for single-session Colab usage.
#
#   Robust Colab error handling:
#   - CUDA OOM → clear fix instructions
#   - Missing models → points to Cell 1
#   - Invalid sampler/scheduler → suggests valid alternatives
#   - VAE decode failures → seed/retry guidance
#   - Session disconnect detection
#
# @exports
#   load_models, generate_image
#
# @version 1.1.0
# @author  Utsav Vasava
# @license MIT
# ======= • ======= • ======= • ======= • =======• =======

import os
import re
import uuid
import gc

import torch
import numpy as np
from PIL import Image

from . import log
from .config import WORKSPACE, DEFAULTS, SAMPLERS, SCHEDULERS

# ══════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════

def _validate_sampler(name):
    """
    Validate sampler name against known ComfyUI samplers.
    Returns the name if valid, or raises with suggestions.

    @param {str} name — Sampler name to validate
    @returns {str} validated sampler name
    @raises ValueError if sampler is unknown
    """
    if name in SAMPLERS:
        return name
    # Fuzzy match: find close matches
    close = [s for s in SAMPLERS if name.lower() in s.lower() or s.lower() in name.lower()]
    if close:
        raise ValueError(
            f"Unknown sampler '{name}'. Did you mean: {', '.join(close[:3])}?\n"
            f"   Available: {', '.join(SAMPLERS)}"
        )
    raise ValueError(
        f"Unknown sampler '{name}'.\n"
        f"   Available: {', '.join(SAMPLERS)}"
    )


def _validate_scheduler(name):
    """
    Validate scheduler name against known ComfyUI schedulers.
    Returns the name if valid, or raises with suggestions.

    @param {str} name — Scheduler name to validate
    @returns {str} validated scheduler name
    @raises ValueError if scheduler is unknown
    """
    if name in SCHEDULERS:
        return name
    close = [s for s in SCHEDULERS if name.lower() in s.lower() or s.lower() in name.lower()]
    if close:
        raise ValueError(
            f"Unknown scheduler '{name}'. Did you mean: {', '.join(close[:3])}?\n"
            f"   Available: {', '.join(SCHEDULERS)}"
        )
    raise ValueError(
        f"Unknown scheduler '{name}'.\n"
        f"   Available: {', '.join(SCHEDULERS)}"
    )


# ══════════════════════════════════════════════════════════════
# MODEL LOADER
# ══════════════════════════════════════════════════════════════

# ---- FEATURE: In-process ComfyUI node loading ----
# Notebook Cell 2: NODE_CLASS_MAPPINGS → 7 nodes → load_unet/load_clip/load_vae

def _check_cuda():
    """
    Pre-flight check: verify CUDA-enabled PyTorch is available.
    Raises a clear, actionable error if CUDA is missing.

    @raises RuntimeError — if PyTorch was not compiled with CUDA support
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "\n"
            "   ✗ CUDA is not available!\n"
            "\n"
            "   PyTorch installed without CUDA support. This usually means:\n"
            "     1. The Colab runtime is set to CPU — change it to T4:\n"
            "        Runtime → Change runtime type → T4 GPU\n"
            "     2. A CPU-only PyTorch was cached — restart runtime and re-run Cell 1\n"
            "\n"
            "   Fix: Runtime → Disconnect and delete runtime → Re-run all cells\n"
        )


def _check_models_exist():
    """
    Verify all required model files exist on disk.
    Returns True if all present, False with guidance if missing.

    @returns {bool}
    """
    from .config import MODEL_DIRS
    models = {
        "UNet (FP8)": os.path.join(WORKSPACE, MODEL_DIRS["unet"], "image-turbo-fp8-e4m3fn.safetensors"),
        "CLIP (Qwen 3 4B)": os.path.join(WORKSPACE, MODEL_DIRS["clip"], "qwen_3_4b.safetensors"),
        "VAE": os.path.join(WORKSPACE, MODEL_DIRS["vae"], "ae.safetensors"),
    }
    missing = [name for name, path in models.items() if not os.path.isfile(path)]
    if missing:
        log.error(f"   Missing models: {', '.join(missing)}")
        log.error("   Fix: Re-run Cell 1 (Initialize) to download models")
        return False
    return True


def load_models(unet_filename="image-turbo-fp8-e4m3fn.safetensors"):
    """
    Load UNet, CLIP, and VAE models into VRAM using ComfyUI nodes.
    Also initializes all required processing nodes.
    Validates models exist before loading. Reports VRAM usage after.

    @param {str} unet_filename — UNet model file name
    @returns {tuple} (nodes_dict, unet_model, clip_model, vae_model)
    @raises RuntimeError if CUDA unavailable or models missing
    """
    _check_cuda()

    if not _check_models_exist():
        raise RuntimeError(
            "Models not found. Run Cell 1 (Initialize) first.\n"
            "   If Cell 1 completed but models are still missing,\n"
            "   your session may have disconnected — re-run Cell 1."
        )

    try:
        from nodes import NODE_CLASS_MAPPINGS
    except ImportError as e:
        raise RuntimeError(
            "\n"
            "   ✗ ComfyUI nodes not found!\n"
            "\n"
            "   This usually means ComfyUI wasn't cloned properly.\n"
            "   Fix: Re-run Cell 1 (Initialize)\n"
            "\n"
            f"   Technical: {e}"
        )

    log.info("Booting ComfyUI Backend...")

    try:
        with torch.inference_mode():
            nodes = {
                "unet":    NODE_CLASS_MAPPINGS["UNETLoader"](),
                "clip":    NODE_CLASS_MAPPINGS["CLIPLoader"](),
                "vae":     NODE_CLASS_MAPPINGS["VAELoader"](),
                "enc":     NODE_CLASS_MAPPINGS["CLIPTextEncode"](),
                "sampler": NODE_CLASS_MAPPINGS["KSampler"](),
                "decode":  NODE_CLASS_MAPPINGS["VAEDecode"](),
                "empty":   NODE_CLASS_MAPPINGS["EmptyLatentImage"](),
            }

            print("   ⏳ Loading Checkpoints into VRAM...", end="\r")
            unet_model = nodes["unet"].load_unet(unet_filename, "fp8_e4m3fn_fast")[0]
            clip_model = nodes["clip"].load_clip("qwen_3_4b.safetensors", type="lumina2")[0]
            vae_model  = nodes["vae"].load_vae("ae.safetensors")[0]

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "\n"
            "   ✗ CUDA Out of Memory during model loading!\n"
            "\n"
            "   The T4 GPU (16 GB) can't fit all models. This is rare with FP8.\n"
            "   Fix:\n"
            "     1. Runtime → Disconnect and delete runtime\n"
            "     2. Re-run all cells (fresh VRAM)\n"
            "     3. If it persists, try a different Colab runtime\n"
        )

    except FileNotFoundError as e:
        raise RuntimeError(
            "\n"
            f"   ✗ Model file not found: {e}\n"
            "\n"
            "   Models may be corrupted or incomplete.\n"
            "   Fix: Re-run Cell 1 (Initialize) to re-download\n"
        )

    except Exception as e:
        raise RuntimeError(
            "\n"
            f"   ✗ Failed to load models: {e}\n"
            "\n"
            "   This could be a corrupted model or ComfyUI version mismatch.\n"
            "   Fix:\n"
            "     1. Re-run Cell 1 (Initialize)\n"
            "     2. If it keeps failing: Runtime → Disconnect and delete runtime → Re-run all\n"
        )

    log.success("Engine Online. Ready to Generate.")

    # ─── VRAM Report ───
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        total     = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        log.info(f"   VRAM: {allocated:.1f} GB allocated / {total:.1f} GB total")

    return nodes, unet_model, clip_model, vae_model


# ══════════════════════════════════════════════════════════════
# IMAGE GENERATOR
# ══════════════════════════════════════════════════════════════

# ---- FEATURE: End-to-end image generation with error handling ----

def generate_image(
    nodes,
    unet_model,
    clip_model,
    vae_model,
    prompt,
    negative_prompt=DEFAULTS["negative_prompt"],
    width=DEFAULTS["width"],
    height=DEFAULTS["height"],
    steps=DEFAULTS["steps"],
    cfg=DEFAULTS["cfg"],
    sampler_name=DEFAULTS["sampler_name"],
    scheduler=DEFAULTS["scheduler"],
    seed=DEFAULTS["seed"],
    save_dir=None,
):
    """
    Generate a single image from a text prompt.
    Validates sampler/scheduler names, handles CUDA OOM,
    catches VAE decode errors, and reports timing.

    @param {dict} nodes — ComfyUI node instances from load_models()
    @param {object} unet_model — Loaded UNet model
    @param {object} clip_model — Loaded CLIP model
    @param {object} vae_model — Loaded VAE model
    @param {str} prompt — Positive prompt text
    @param {str} negative_prompt — Negative prompt text
    @param {int} width — Image width in pixels
    @param {int} height — Image height in pixels
    @param {int} steps — Sampling steps (10–50, default 20)
    @param {float} cfg — CFG scale (1.0–10.0, default 1)
    @param {str} sampler_name — KSampler algorithm name (default "euler")
    @param {str} scheduler — Noise schedule type (default "simple")
    @param {int} seed — RNG seed (-1 = random via torch.randint)
    @param {str} save_dir — Output directory (defaults to /content/results)
    @returns {tuple} (PIL.Image, save_path)
    @raises RuntimeError on generation failures
    @raises ValueError on invalid sampler/scheduler
    """
    # ─── Validate inputs ───
    sampler_name = _validate_sampler(sampler_name)
    scheduler = _validate_scheduler(scheduler)

    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty. Describe what you want to generate.")

    if width < 512 or height < 512 or width > 2048 or height > 2048:
        raise ValueError(f"Resolution {width}×{height} is out of range. Use 512–2048 (step 64).")

    if steps < 1 or steps > 100:
        raise ValueError(f"Steps {steps} is out of range. Use 10–50 (recommended: 20).")

    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(WORKSPACE), "results")
    os.makedirs(save_dir, exist_ok=True)

    # Notebook: gen_seed = torch.randint(0, 2**63 - 1, (1,)).item() if seed == -1 else seed
    gen_seed = torch.randint(0, 2**63 - 1, (1,)).item() if seed == -1 else seed

    # Notebook: flush_mem() → gc.collect() + torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

    log.info(f"Generating: {width}x{height} | Steps: {steps} | Sampler: {sampler_name} | Seed: {gen_seed}")

    try:
        with torch.inference_mode():
            # ─── Encode ───
            pos_enc = nodes["enc"].encode(clip_model, prompt)[0]
            neg_enc = nodes["enc"].encode(clip_model, negative_prompt)[0]

            # ─── Latent ───
            latent = nodes["empty"].generate(width, height, batch_size=1)[0]

            # ─── Sample ───
            sample = nodes["sampler"].sample(
                unet_model, gen_seed, steps, cfg,
                sampler_name, scheduler, pos_enc, neg_enc, latent, denoise=1.0
            )[0]

            # ─── Decode ───
            decoded = nodes["decode"].decode(vae_model, sample)[0].detach()

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        raise RuntimeError(
            "\n"
            "   ✗ CUDA Out of Memory during generation!\n"
            "\n"
            "   The image is too large for the T4's 16 GB VRAM.\n"
            "   Fix:\n"
            f"     • Lower resolution (current: {width}×{height}) → try 1024×1024 or 720×1280\n"
            f"     • Reduce steps (current: {steps}) → try 15 or 10\n"
            "     • Runtime → Disconnect and delete runtime → Re-run all (fresh VRAM)\n"
        )

    except RuntimeError as e:
        err_str = str(e).lower()
        if "sampler" in err_str or "scheduler" in err_str:
            raise RuntimeError(
                f"\n"
                f"   ✗ Sampler/Scheduler error: {e}\n"
                f"\n"
                f"   The sampler '{sampler_name}' or scheduler '{scheduler}' may not be\n"
                f"   supported by this ComfyUI version.\n"
                f"   Fix: Use defaults (euler / simple) or try another combination.\n"
            )
        raise RuntimeError(
            f"\n"
            f"   ✗ Generation failed: {e}\n"
            f"\n"
            f"   Try:\n"
            f"     1. Different seed (set seed to -1 for random)\n"
            f"     2. Lower resolution\n"
            f"     3. Re-run Cell 2 to reload models\n"
        )

    except Exception as e:
        raise RuntimeError(
            f"\n"
            f"   ✗ Unexpected error during generation: {e}\n"
            f"\n"
            f"   Try:\n"
            f"     1. Re-run Cell 2 (Load Engine)\n"
            f"     2. If it persists: Runtime → Disconnect and delete runtime → Re-run all\n"
        )

    # ─── Save ───
    try:
        img_out = Image.fromarray(np.array(decoded * 255, dtype=np.uint8)[0])
        clean_prompt = re.sub(r'[^a-zA-Z0-9_-]', '_', prompt)[:20]
        filename = f"{clean_prompt}_{uuid.uuid4().hex[:4]}.png"
        save_path = os.path.join(save_dir, filename)
        img_out.save(save_path)
    except Exception as e:
        raise RuntimeError(
            f"\n"
            f"   ✗ Failed to save image: {e}\n"
            f"\n"
            f"   The image was generated but couldn't be saved.\n"
            f"   Check disk space: run Cell 4 (Cleanup) to free space.\n"
        )

    log.success(f"Saved to: {filename}")
    return img_out, save_path


# ══════════════════════════════════════════════════════════════
# EXPORTS
# ══════════════════════════════════════════════════════════════

__all__ = ["load_models", "generate_image"]

# ══════════════════════════════════════════════════════════════ END: generator.py
