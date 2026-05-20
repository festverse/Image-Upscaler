# ======= • ======= • ======= • ======= • =======• =======
# image Pro — downloader.py
# Repository: https://github.com/festverse/Image-Upscaler
#
# @description
#   Asset downloader with aria2c acceleration and Google Drive
#   model caching. Handles direct HTTP(S) downloads, Google Drive
#   links (via gdown), and Civitai content-disposition URLs.
#   Parallel chunk downloading with 16 connections per file.
#
#   Drive Cache Strategy:
#   - First run: download → save to Drive cache
#   - Next runs: copy from Drive cache → skip download
#   - Cache versioning: stale cache auto-invalidated
#   - Disk space checks: warns before running out of space
#
# @exports
#   ensure_aria2, mount_drive, download_file, process_downloads,
#   get_cache_status, clear_cache, check_disk_space
#
# @version 1.1.0
# @author  Utsav Vasava
# @license MIT
# ======= • ======= • ======= • ======= • =======• =======

import os
import shutil
import subprocess
import urllib.parse

try:
    import gdown
except ImportError:
    gdown = None

from . import log
from .config import (
    DRIVE_CACHE_DIR, CACHE_VERSION, TOTAL_MODEL_SIZE_GB,
    DISK_WARN_GB, DISK_MIN_GB,
)

# ══════════════════════════════════════════════════════════════
# DISK SPACE
# ══════════════════════════════════════════════════════════════

def _free_gb(path="/content"):
    """Return free disk space in GB for the given path."""
    try:
        stat = shutil.disk_usage(path)
        return stat.free / (1024 ** 3)
    except Exception:
        return float("inf")


def check_disk_space(required_gb=TOTAL_MODEL_SIZE_GB):
    """
    Check if enough disk space is available.
    Warns at DISK_WARN_GB, refuses at DISK_MIN_GB.

    @param {float} required_gb — Space needed in GB
    @returns {bool} True if OK, False if critically low
    """
    free = _free_gb()
    if free < DISK_MIN_GB:
        log.error(f"   Only {free:.1f} GB free — need at least {DISK_MIN_GB} GB")
        log.error("   Free up space: delete old outputs or clear Drive cache")
        return False
    if free < DISK_WARN_GB:
        log.warn(f"   Only {free:.1f} GB free — things might get tight")
    if free < required_gb:
        log.warn(f"   {free:.1f} GB free but models need ~{required_gb:.0f} GB")
        log.warn("   Drive cache may help — models copy from Drive instead of downloading")
    return True


# ══════════════════════════════════════════════════════════════
# CACHE VERSIONING
# ══════════════════════════════════════════════════════════════

def _version_file():
    """Path to the cache version stamp file."""
    return os.path.join(DRIVE_CACHE_DIR, ".cache_version")


def _read_cache_version():
    """Read the stored cache version, or None if missing."""
    vf = _version_file()
    if os.path.isfile(vf):
        try:
            return open(vf).read().strip()
        except Exception:
            return None
    return None


def _write_cache_version():
    """Write the current cache version stamp."""
    os.makedirs(DRIVE_CACHE_DIR, exist_ok=True)
    with open(_version_file(), "w") as f:
        f.write(CACHE_VERSION)


def _is_cache_stale():
    """Check if the Drive cache version mismatches (stale)."""
    stored = _read_cache_version()
    return stored is not None and stored != CACHE_VERSION


# ══════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ══════════════════════════════════════════════════════════════

# ---- FEATURE: Google Drive mount + cache helpers ----

def mount_drive():
    """
    Mount Google Drive at /content/drive if not already mounted.
    Returns True if Drive is available, False otherwise.

    @returns {bool}
    """
    if os.path.isdir("/content/drive/MyDrive"):
        return True
    try:
        from google.colab import drive
        log.info("   📂 Mounting Google Drive...")
        drive.mount("/content/drive", force_remount=False)
        return True
    except Exception as e:
        log.warn(f"   Could not mount Drive: {e}")
        return False


def _cache_filename(url):
    """Derive a stable cache filename from a download URL."""
    parsed = urllib.parse.urlparse(url)
    return os.path.basename(parsed.path)


def _file_size_gb(filepath):
    """Return file size in GB, or 0 if not found."""
    try:
        return os.path.getsize(filepath) / (1024 ** 3)
    except OSError:
        return 0.0


def _try_load_from_cache(url, target_dir):
    """
    Check if a model exists in the Google Drive cache.
    If found, copy it to the target directory (fast local copy).

    @param {str} url — Model download URL
    @param {str} target_dir — ComfyUI model directory
    @returns {bool} True if cache hit and copied, False on miss
    """
    filename = _cache_filename(url)
    cache_path = os.path.join(DRIVE_CACHE_DIR, filename)
    dest_path = os.path.join(target_dir, filename)

    if os.path.isfile(cache_path):
        size = _file_size_gb(cache_path)
        log.info(f"   💾 Drive cache hit: {filename[:40]} ({size:.1f} GB)")
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy2(cache_path, dest_path)
        log.success(f"   Copied from Drive cache")
        return True
    return False


def _save_to_cache(url, source_dir):
    """
    Copy a downloaded model file to the Google Drive cache for future runs.
    Creates the cache directory if needed.

    @param {str} url — Model download URL (used to derive filename)
    @param {str} source_dir — Directory where the file was downloaded
    @returns {None}
    """
    filename = _cache_filename(url)
    source_path = os.path.join(source_dir, filename)
    cache_path = os.path.join(DRIVE_CACHE_DIR, filename)

    if not os.path.isfile(source_path):
        return

    os.makedirs(DRIVE_CACHE_DIR, exist_ok=True)
    try:
        size = _file_size_gb(source_path)
        log.info(f"   📤 Saving to Drive cache: {filename[:40]} ({size:.1f} GB)...")
        shutil.copy2(source_path, cache_path)
        _write_cache_version()
        log.success("   Cached to Drive for next session")
    except Exception as e:
        log.warn(f"   Could not cache to Drive: {e}")


# ---- FEATURE: aria2c installer ----
# Notebook: run_quiet("apt -y install -qq aria2", "Installing Accelerator (Aria2)")

def ensure_aria2():
    """
    Install aria2c if not already present on the system.
    Called once during initialization to guarantee availability.

    Matches notebook: run_quiet("apt -y install -qq aria2", "Installing Accelerator (Aria2)")

    @returns {None}
    """
    try:
        subprocess.run(
            ["aria2c", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        subprocess.run(
            ["apt-get", "-y", "install", "-qq", "aria2"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        log.info("aria2c installed successfully")


# ══════════════════════════════════════════════════════════════
# CACHE STATUS & MANAGEMENT
# ══════════════════════════════════════════════════════════════

def get_cache_status(urls):
    """
    Check which models are cached in Drive and report status.
    Returns a dict with per-model info and totals.

    @param {list} urls — List of model download URLs to check
    @returns {dict} Status info with keys: cached, missing, total_size_gb, stale
    """
    status = {"cached": [], "missing": [], "total_size_gb": 0.0, "stale": False}

    if not os.path.isdir(DRIVE_CACHE_DIR):
        return status

    status["stale"] = _is_cache_stale()

    for url in urls:
        filename = _cache_filename(url)
        cache_path = os.path.join(DRIVE_CACHE_DIR, filename)
        if os.path.isfile(cache_path):
            size = _file_size_gb(cache_path)
            status["cached"].append({"name": filename, "size_gb": size})
            status["total_size_gb"] += size
        else:
            status["missing"].append(filename)

    return status


def clear_cache():
    """
    Remove the entire Drive cache directory.
    Prints what was deleted and how much space was freed.

    @returns {None}
    """
    if not os.path.isdir(DRIVE_CACHE_DIR):
        log.info("   No cache to clear — directory doesn't exist")
        return

    # Calculate size before deleting
    total = 0
    for f in os.listdir(DRIVE_CACHE_DIR):
        fp = os.path.join(DRIVE_CACHE_DIR, f)
        if os.path.isfile(fp):
            total += os.path.getsize(fp)

    shutil.rmtree(DRIVE_CACHE_DIR)
    freed = total / (1024 ** 3)
    log.success(f"   Cache cleared — freed {freed:.1f} GB")


# ══════════════════════════════════════════════════════════════
# DOWNLOAD ENGINE
# ══════════════════════════════════════════════════════════════

# ---- FEATURE: Single file downloader ----
# Notebook: run_quiet(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M '{url}' -d '{path.parent}' -o '{path.name}'", ...)

def download_file(url, target_dir, use_drive_cache=True):
    """
    Download a single file to the target directory.
    Automatically detects Google Drive, Civitai, and direct URLs.
    Checks Google Drive cache first — if cached, copies locally (fast).
    After download, saves a copy to Drive cache for future sessions.

    Matches notebook's aria2c flags: --console-log-level=error -c -x 16 -s 16 -k 1M

    @param {str} url — Full download URL
    @param {str} target_dir — Local directory to save into
    @param {bool} use_drive_cache — Whether to check/save Drive cache (default True)
    @returns {None}
    """
    os.makedirs(target_dir, exist_ok=True)

    # ─── Check disk space before downloading ───
    if not check_disk_space():
        raise RuntimeError("Insufficient disk space for model download")

    # ─── Try Drive cache first ───
    if use_drive_cache and "drive.google.com" not in url:
        if _is_cache_stale():
            log.warn("   ⚠ Cache version mismatch — re-downloading fresh")
        elif _try_load_from_cache(url, target_dir):
            return

    before = set(os.listdir(target_dir))

    try:
        # ─── Google Drive (gdown) ───
        if "drive.google.com" in url:
            if gdown is None:
                raise ImportError("gdown is required for Google Drive downloads")
            log.info("   📥 Downloading from Drive...")
            gdown.download(url, output=target_dir + "/", quiet=False, fuzzy=True)

        # ─── Direct / Civitai / HuggingFace ───
        else:
            parsed = urllib.parse.urlparse(url)
            filename = os.path.basename(parsed.path)
            log.info(f"   📥 Fetching: {filename[:40]}...")

            # Notebook exact flags: aria2c --console-log-level=error -c -x 16 -s 16 -k 1M
            cmd = [
                "aria2c",
                "--console-log-level=error",
                "--summary-interval=10",
                "-c", "-x", "16", "-s", "16", "-k", "1M",
            ]

            if "civitai.com" in url:
                cmd.extend(["--content-disposition", url, "-d", target_dir])
            elif filename:
                cmd.extend(["-o", filename, url, "-d", target_dir])
            else:
                cmd.extend(["--content-disposition", url, "-d", target_dir])

            subprocess.run(cmd, check=True)

        # ─── Report new files ───
        after = set(os.listdir(target_dir))
        new = after - before
        if new:
            log.success(f"   Saved as: {list(new)[0]}")
        else:
            log.success("   Download complete")

        # ─── Save to Drive cache for next session ───
        if use_drive_cache and "drive.google.com" not in url:
            _save_to_cache(url, target_dir)

    except Exception as e:
        log.error(f"   Failed: {url}\n      Error: {e}\n")


# ---- FEATURE: Batch download processor ----

def process_downloads(urls, target_dir):
    """
    Parse a comma or newline-separated URL string and download each.
    Skips empty lines and whitespace-only entries gracefully.

    @param {str} urls — Raw URL string (comma/newline separated)
    @param {str} target_dir — Local directory to save all files
    @returns {None}
    """
    if not urls.strip():
        return

    url_list = [u.strip() for u in urls.replace(",", "\n").split("\n") if u.strip()]
    os.makedirs(target_dir, exist_ok=True)
    log.info(f"\n📂 Directory: {os.path.basename(target_dir)}")

    for url in url_list:
        download_file(url, target_dir)


# ══════════════════════════════════════════════════════════════
# EXPORTS
# ══════════════════════════════════════════════════════════════

__all__ = [
    "ensure_aria2", "mount_drive", "download_file", "process_downloads",
    "get_cache_status", "clear_cache", "check_disk_space",
]

# ══════════════════════════════════════════════════════════════ END: downloader.py
