# Contributing to image Turbo Pro

Thank you for your interest in contributing! Here's how you can help.

---

## Ways to Contribute

### 🐛 Report Bugs
Found something broken? [Open an Issue](https://github.com/festverse/Image-Upscaler/issues) with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Your environment (Colab GPU type, browser)

### 💡 Suggest Features
Have an idea? [Start a Discussion](https://github.com/festverse/Image-Upscaler/issues) with:
- What problem does it solve?
- How should it work?
- Any alternatives you've considered?

### 🔀 Submit Code
Ready to contribute code? Here's the workflow:

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/ImageUpscaler.git`
3. **Create a branch**: `git checkout -b feature/your-feature-name`
4. **Make your changes**
5. **Test** in Colab (run all cells, verify output)
6. **Commit**: `git commit -m "feat: description of change"`
7. **Push**: `git push origin feature/your-feature-name`
8. **Open a Pull Request**

---

## Development Setup

```bash
# Clone the repo
git clone https://github.com/festverse/Image-Upscaler.git
cd ImageUpscaler

# Install dependencies
pip install -r requirements.txt

# Test imports
python -c "from src.config import DEFAULTS; print(DEFAULTS)"
python -c "from src.downloader import ensure_aria2"
python -c "from src.generator import load_models, generate_image"
python -c "from src.exporter import zip_outputs, download_zip"
```

---

## Code Style

- Follow the existing code style in `src/` modules
- Use the shared `log` from `src/__init__.py` for all output
- Add `@param` and `@returns` docstrings to all functions
- Add `# ---- FEATURE: name ----` section markers
- Keep `__all__` exports updated in each module
- Update `CHANGELOG.md` with every change (newest at top)

---

## Commit Messages

Use conventional commits:
- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation only
- `refactor:` — code restructure (no behavior change)
- `style:` — formatting, whitespace
- `test:` — adding tests

---

## Code of Conduct

Be respectful, constructive, and inclusive. We're all here to build cool stuff.
