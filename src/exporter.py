# ======= • ======= • ======= • ======= • =======• =======
# image Pro — exporter.py
# Repository: https://github.com/festverse/Image-Upscaler
#
# @description
#   Output export utilities. Zips all generated PNG images
#   from the results directory and triggers a browser download
#   when running in Google Colab. Falls back to printing the
#   local zip path in non-Colab environments.
#
#   Features for free-tier users:
#   - Auto-cleanup of old outputs to save disk space
#   - Optional Drive export (outputs persist across sessions)
#   - Output count and size reporting
#
# @exports
#   zip_outputs, download_zip, cleanup_outputs, get_output_stats
#
# @version 1.1.0
# @author  Utsav Vasava
# @license MIT
# ======= • ======= • ======= • ======= • =======• =======

import os
import shutil
import subprocess

from . import log

# ══════════════════════════════════════════════════════════════
# OUTPUT STATS
# ══════════════════════════════════════════════════════════════

def get_output_stats(output_dir="/content/results"):
    """
    Count and measure all PNG outputs in the results directory.

    @param {str} output_dir — Results directory
    @returns {dict} {count, total_mb, files}
    """
    if not os.path.isdir(output_dir):
        return {"count": 0, "total_mb": 0.0, "files": []}

    files = [f for f in os.listdir(output_dir) if f.endswith(".png")]
    total = sum(os.path.getsize(os.path.join(output_dir, f)) for f in files)
    return {
        "count": len(files),
        "total_mb": total / (1024 ** 2),
        "files": sorted(files),
    }


# ══════════════════════════════════════════════════════════════
# AUTO-CLEANUP
# ══════════════════════════════════════════════════════════════

def cleanup_outputs(output_dir="/content/results", keep_latest=0):
    """
    Remove old output files to free disk space.
    If keep_latest > 0, preserves the N most recent files.

    @param {str} output_dir — Results directory
    @param {int} keep_latest — Number of most recent files to keep (0 = delete all)
    @returns {int} Number of files removed
    """
    if not os.path.isdir(output_dir):
        return 0

    files = sorted(
        [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".png")],
        key=os.path.getmtime,
        reverse=True,
    )

    if not files:
        return 0

    to_remove = files[keep_latest:] if keep_latest > 0 else files
    freed = 0
    for f in to_remove:
        freed += os.path.getsize(f)
        os.remove(f)

    freed_mb = freed / (1024 ** 2)
    if to_remove:
        log.info(f"   🧹 Cleaned {len(to_remove)} old outputs ({freed_mb:.1f} MB freed)")
    return len(to_remove)


# ══════════════════════════════════════════════════════════════
# ARCHIVE
# ══════════════════════════════════════════════════════════════

# ---- FEATURE: Zip generated images ----

def zip_outputs(
    output_dir="/content/results",
    zip_path="/content/Z_Image_Pro_Artworks.zip",
):
    """
    Zip all PNG files in the output directory.
    Creates the zip at the given path using the system zip command.
    Returns None and prints a warning if no images are found.

    @param {str} output_dir — Results output directory (matches notebook's SAVE_DIR)
    @param {str} zip_path — Destination zip file path
    @returns {str|None} Path to the created zip, or None if empty
    """
    stats = get_output_stats(output_dir)
    if stats["count"] == 0:
        log.warn("No images found in the output directory yet!")
        return None

    log.info(f"🗜️ Zipping {stats['count']} artworks ({stats['total_mb']:.1f} MB)...")
    png_files = [
        os.path.join(output_dir, f)
        for f in stats["files"]
    ]
    subprocess.run(["zip", "-j", "-q", zip_path, *png_files], check=True)
    log.success(f"Zipped to: {zip_path}")
    return zip_path


# ══════════════════════════════════════════════════════════════
# BROWSER DOWNLOAD
# ══════════════════════════════════════════════════════════════

# ---- FEATURE: Colab file download trigger ----
# Notebook Cell 3: if auto_download: from google.colab import files; files.download(str(save_path))

def download_zip(zip_path="/content/Z_Image_Pro_Artworks.zip"):
    """
    Trigger a browser download of the zip file.
    Uses google.colab.files API when available, otherwise
    prints the local path for manual retrieval.

    Matches notebook: from google.colab import files; files.download(str(save_path))

    @param {str} zip_path — Path to the zip file
    @returns {None}
    """
    try:
        from google.colab import files
        log.info("📥 Initiating download...")
        files.download(zip_path)
    except ImportError:
        log.warn(f"Not running in Colab. Zip saved at: {zip_path}")


# ══════════════════════════════════════════════════════════════
# EXPORTS
# ══════════════════════════════════════════════════════════════

__all__ = ["zip_outputs", "download_zip", "cleanup_outputs", "get_output_stats"]

# ══════════════════════════════════════════════════════════════ END: exporter.py
