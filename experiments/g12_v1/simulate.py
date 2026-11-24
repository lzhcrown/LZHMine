"""Run the G12 v1 student in MuJoCo."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import simulate_main
from experiments.registry import PLATFORM_V1_EXPERIMENT


if __name__ == "__main__":
    simulate_main(PLATFORM_V1_EXPERIMENT)
