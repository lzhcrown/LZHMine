"""Evaluate this CTS ablation in Isaac Gym and export its student policy."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import play_main


if __name__ == "__main__":
    play_main("g6_actor_critic_moe")

