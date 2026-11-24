"""Export the G12 v1 student policy."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import export_main
from experiments.registry import PLATFORM_V1_EXPERIMENT


if __name__ == "__main__":
    export_main(PLATFORM_V1_EXPERIMENT)
