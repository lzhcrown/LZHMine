"""Train experiment 2: CTS student MoE plus an eight-expert MoE Actor."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import train_main


if __name__ == "__main__":
    train_main("exp02_actor_moe")
