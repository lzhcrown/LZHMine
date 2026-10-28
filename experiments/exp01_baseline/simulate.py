"""Run experiment 1's exported policy in the shared MuJoCo terrain course."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import simulate_main


if __name__ == "__main__":
    simulate_main("exp01_baseline")
