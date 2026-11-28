"""Export the G13 v2 geometry-conditioned student."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import export_main
from experiments.registry import SKILL_V2_EXPERIMENT


if __name__ == "__main__":
    export_main(SKILL_V2_EXPERIMENT)
