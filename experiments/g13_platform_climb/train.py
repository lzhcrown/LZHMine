"""Train the G13 platform climbing task."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import train_main
from experiments.registry import PLATFORM_CLIMB_EXPERIMENT


if __name__ == "__main__":
    train_main(PLATFORM_CLIMB_EXPERIMENT)
