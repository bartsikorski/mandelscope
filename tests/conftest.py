import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

if os.environ.get("MANDELSCOPE_SOFTWARE") == "1":
    # CPU mode (the Windows CI job): Mesa llvmpipe must be loaded before any OpenGL context
    from mandelscope import software
    print(software.enable())


@pytest.fixture(scope="session")
def engine():
    try:
        from mandelscope.cli import make_headless_context
        from mandelscope.renderer import GPUEngine
        ctx = make_headless_context()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"no OpenGL 4.3 context: {e}")
    return GPUEngine(ctx)


@pytest.fixture(scope="session")
def locations():
    return ROOT / "mandelscope" / "locations"
