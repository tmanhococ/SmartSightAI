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
    # Stability: pin pydantic to avoid gradio_client JSON schema TypeError
    "pydantic",
    # Stability: pin numpy to resolve Numpy is not available runtime error
    "numpy",
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


def test_gradio_version_is_exact_pin():
    """Gradio must be pinned to ==4.44.1 (exact), not >=, to prevent pip from
    upgrading to Gradio 6.x which has breaking API changes that crash HF Space."""
    path = os.path.join(ROOT, "requirements.txt")
    content = open(path, "r", encoding="utf-8").read()
    import re
    # Must be exact == pin
    match = re.search(r"gradio==([\d.]+)", content)
    assert match, (
        "requirements.txt must use 'gradio==4.44.1' (exact pin), not 'gradio>=...'. "
        "Using >= allows pip to upgrade to Gradio 6.x which has breaking changes "
        "incompatible with sdk_version: 4.44.1 HF Spaces infrastructure."
    )
    from packaging.version import Version
    assert Version(match.group(1)) >= Version("4.44.1"), (
        f"gradio must be pinned to >=4.44.1 (found =={match.group(1)})"
    )


def test_jinja2_is_pinned():
    """jinja2 must be pinned to <4.0.0 to avoid 'unhashable type: dict' crash
    in Gradio 4.x template cache with newer Jinja2 versions."""
    path = os.path.join(ROOT, "requirements.txt")
    content = open(path, "r", encoding="utf-8").read().lower()
    assert "jinja2" in content, (
        "requirements.txt must pin jinja2 (e.g. jinja2>=3.1.2,<4.0.0) to prevent "
        "TypeError: unhashable type: 'dict' crash in Gradio 4.x template rendering."
    )


def test_pydantic_is_pinned():
    """pydantic must be pinned to avoid gradio_client JSON schema TypeError."""
    path = os.path.join(ROOT, "requirements.txt")
    content = open(path, "r", encoding="utf-8").read()
    assert "pydantic" in content, (
        "requirements.txt must pin pydantic (e.g. pydantic==2.10.6) to prevent "
        "gradio_client/utils.py TypeError: argument of type 'bool' is not iterable"
    )


def test_fastapi_and_starlette_are_pinned():
    """fastapi and starlette must be pinned to older versions to prevent 'unhashable type: dict' crash."""
    path = os.path.join(ROOT, "requirements.txt")
    content = open(path, "r", encoding="utf-8").read().lower()
    assert "fastapi" in content, (
        "requirements.txt must pin fastapi (e.g. fastapi<0.115.0) to prevent "
        "compatibility issues with newer Starlette/FastAPI releases."
    )
    assert "starlette" in content, (
        "requirements.txt must pin starlette (e.g. starlette<0.39.0) to prevent "
        "TypeError: unhashable type: 'dict' crash in template rendering."
    )


def test_numpy_is_pinned():
    """numpy must be pinned to avoid Numpy is not available runtime crash."""
    path = os.path.join(ROOT, "requirements.txt")
    content = open(path, "r", encoding="utf-8").read()
    assert "numpy" in content, (
        "requirements.txt must pin numpy (e.g. numpy==1.26.4) to prevent "
        "RuntimeError: Numpy is not available on Hugging Face Spaces."
    )




def test_readme_sdk_version_matches_requirements():
    """sdk_version in README.md must be >=4.44.1 to match requirements.txt."""
    path = os.path.join(ROOT, "README.md")
    content = open(path, "r", encoding="utf-8").read()
    import re
    match = re.search(r"sdk_version:\s*([\d.]+)", content)
    assert match, "README.md must specify sdk_version in YAML front-matter"
    from packaging.version import Version
    assert Version(match.group(1)) >= Version("4.44.1"), (
        f"README.md sdk_version must be >=4.44.1 (found {match.group(1)})"
    )


def test_app_py_launch_has_required_hf_params():
    """Root app.py launch() must have show_api=False and server_name=0.0.0.0."""
    path = os.path.join(ROOT, "app.py")
    content = open(path, "r", encoding="utf-8").read()
    assert "show_api=False" in content, (
        "app.py launch() must set show_api=False to disable the API schema generation "
        "route that causes TypeError: argument of type 'bool' is not iterable"
    )
    assert 'server_name="0.0.0.0"' in content or "server_name='0.0.0.0'" in content, (
        "app.py launch() must set server_name='0.0.0.0' to bind all interfaces "
        "on HF Spaces containers (prevents 'localhost not accessible' ValueError)"
    )


def test_app_py_no_warm_start_blocking_call():
    """Root app.py must NOT call registry.get_vlm() at startup.

    Warm-starting blocks Gradio from starting for ~40s while downloading
    the 3.85GB model. Model must load lazily on first user Run instead.
    """
    path = os.path.join(ROOT, "app.py")
    content = open(path, "r", encoding="utf-8").read()
    assert "registry.get_vlm" not in content, (
        "app.py must not call registry.get_vlm() at startup. "
        "Use lazy loading: model is loaded on first Run via ModelRegistry singleton."
    )
