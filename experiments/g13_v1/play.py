"""Replay the G13 v1 geometry-conditioned student in Isaac Gym."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import play_main
from experiments.registry import SKILL_V1_EXPERIMENT


if __name__ == "__main__":
    play_main(SKILL_V1_EXPERIMENT)
