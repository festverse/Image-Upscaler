# ======= • ======= • ======= • ======= • =======• =======
# image Pro — __init__.py
# Repository: https://github.com/festverse/Image-Upscaler
#
# @description
#   Package marker for the image Pro modular pipeline.
#   Exposes version string and shared logger for all modules.
#
# @exports __version__, log
#
# @version 1.0.0
# @author  Utsav Vasava
# @license MIT
# ======= • ======= • ======= • ======= • =======• =======

__version__ = "1.4.1"


# ══════════════════════════════════════════════════════════════
# SHARED UI LOGGER
# ══════════════════════════════════════════════════════════════

# ---- FEATURE: Colored terminal logger (from notebook) ----

class _Log:
    """
    Colored terminal logger matching the notebook's UI style.
    Uses ANSI escape codes for colored output with symbol prefixes.

    Levels:
        info    → Blue  (➜)
        success → Green (✓)
        warn    → Yellow (⚠)
        error   → Red   (✗)
    """

    _COLORS = {
        "info":    "\033[94m",
        "success": "\033[92m",
        "warn":    "\033[93m",
        "error":   "\033[91m",
        "reset":   "\033[0m",
    }

    _SYMBOLS = {
        "info":    "➜",
        "success": "✓",
        "warn":    "⚠",
        "error":   "✗",
    }

    def __call__(self, msg, level="info"):
        """Log a message with colored output. Default level is 'info'."""
        color = self._COLORS.get(level, self._COLORS["info"])
        symbol = self._SYMBOLS.get(level, self._SYMBOLS["info"])
        reset = self._COLORS["reset"]
        print(f"{color}{symbol} {msg}{reset}")

    @staticmethod
    def info(*args):
        """Log an info message (blue, ➜)."""
        print("\033[94m➜", *args, "\033[0m")

    @staticmethod
    def success(*args):
        """Log a success message (green, ✓)."""
        print("\033[92m✓", *args, "\033[0m")

    @staticmethod
    def warn(*args):
        """Log a warning message (yellow, ⚠)."""
        print("\033[93m⚠", *args, "\033[0m")

    @staticmethod
    def error(*args):
        """Log an error message (red, ✗)."""
        print("\033[91m✗", *args, "\033[0m")


log = _Log()


# ══════════════════════════════════════════════════════════════
# COMMAND RUNNER
# ══════════════════════════════════════════════════════════════

# ---- FEATURE: Silent command executor with progress UI ----

def run_quiet(cmd, desc):
    """
    Run a shell command silently with a progress indicator.
    Shows ⏳ spinner while running, then ✓ on success or ✗ on failure.

    Matches the notebook's run_quiet() helper exactly.

    @param {str} cmd — Shell command to execute
    @param {str} desc — Human-readable description of the operation
    @returns {bool} True on success, False on failure
    """
    import subprocess

    print(f"   ⏳ {desc}...", end="\r")
    try:
        subprocess.check_call(
            cmd, shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"\033[92m   ✓ {desc}\033[0m    ")
        return True
    except Exception as e:
        print(f"\033[91m   ✗ {desc} Failed\033[0m")
        print(e)
        return False


__all__ = ["log", "run_quiet", "__version__"]

# ══════════════════════════════════════════════════════════════ END: __init__.py
