"""Export an eight-expert Actor checkpoint into experiment 2's policy folder."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import export_main


if __name__ == "__main__":
    export_main("exp02_actor_moe")
