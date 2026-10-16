"""Prepare repository entry points before importing Isaac Gym or PyTorch."""

import os
from pathlib import Path
import sys
import tempfile


def configure_runtime(isaac_gym=False):
    root = Path(__file__).resolve().parents[2]
    for source in (root, root / "rsl_rl"):
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
    if isaac_gym:
        # Directly invoking .venv/bin/python does not activate its bin directory.
        # Gym's torch extension must also be able to find that environment's ninja.
        executable_dir = str(Path(sys.executable).parent)
        search_path = os.environ.get("PATH", "").split(os.pathsep)
        if executable_dir not in search_path:
            os.environ["PATH"] = os.pathsep.join([executable_dir] + search_path)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault(
            "TORCH_EXTENSIONS_DIR",
            str(Path(tempfile.gettempdir()) / "lzhmine_torch_extensions"),
        )
