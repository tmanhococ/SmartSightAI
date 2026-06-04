"""
Sanity tests to verify Hugging Face Space configuration files are
complete and contain all required dependencies.
"""

import os
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REQUIRED_PIP_PACKAGES = [
    "gradio",
    "transformers",
    "torch",
    "pillow",
    "deep-translator",
    "gTTS",
    "pyttsx3",
    "psutil",
    "sentencepiece",
    "sacremoses",
    "huggingface_hub",
    # Moondream2 VLM dependencies
    "pyvips",
    "einops",
    "timm",
    "accelerate",
]

REQUIRED_APT_PACKAGES = [
    "libvips-dev",
    "espeak-ng",
    "libespeak1",
    "ffmpeg",
]


def _read_file_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip() and not line.strip().startswith("#")]


def test_requirements_txt_exists():
    path = os.path.join(ROOT, "requirements.txt")
    assert os.path.isfile(path), "requirements.txt must exist at the project root"


def test_packages_txt_exists():
    path = os.path.join(ROOT, "packages.txt")
    assert os.path.isfile(path), (
        "packages.txt must exist at the project root for Hugging Face Spaces "
        "to install system-level (apt) dependencies"
    )


def test_requirements_txt_contains_all_pip_deps():
    path = os.path.join(ROOT, "requirements.txt")
    content = open(path, "r", encoding="utf-8").read().lower()
    missing = []
    for pkg in REQUIRED_PIP_PACKAGES:
        # Match package name regardless of version specifiers
        pkg_base = pkg.split(">=")[0].split("<=")[0].split("==")[0].split("<")[0].lower()
        if pkg_base not in content:
            missing.append(pkg)
    assert not missing, f"requirements.txt is missing these packages: {missing}"


def test_packages_txt_contains_all_apt_deps():
    path = os.path.join(ROOT, "packages.txt")
    lines = _read_file_lines(path)
    content_set = set(l.lower() for l in lines)
    missing = [pkg for pkg in REQUIRED_APT_PACKAGES if pkg.lower() not in content_set]
    assert not missing, f"packages.txt is missing these system packages: {missing}"


def test_readme_has_hf_metadata():
    """README.md must contain the HF Spaces YAML front-matter for the Space to be detected correctly."""
    path = os.path.join(ROOT, "README.md")
    assert os.path.isfile(path), "README.md must exist"
    content = open(path, "r", encoding="utf-8").read()
    assert "sdk: gradio" in content, "README.md must contain 'sdk: gradio' in YAML front-matter"
    assert "app_file:" in content, "README.md must contain 'app_file:' in YAML front-matter"


def test_app_py_exists_at_root():
    """HF Spaces requires the app file to be at the project root."""
    path = os.path.join(ROOT, "app.py")
    assert os.path.isfile(path), "app.py must exist at the project root for HF Spaces to discover it"


def test_app_py_has_no_standalone_output_mp3():
    """Verify the root app.py uses UUID-based audio filenames, not hardcoded 'output.mp3'."""
    path = os.path.join(ROOT, "src", "app.py")
    content = open(path, "r", encoding="utf-8").read()
    assert '"output.mp3"' not in content and "'output.mp3'" not in content, (
        "src/app.py must not use a hardcoded 'output.mp3' filename. "
        "Use UUID-based filenames to prevent race conditions in multi-user environments."
    )
    assert "uuid" in content, "src/app.py must import and use uuid for unique audio filenames"
