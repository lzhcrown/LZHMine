"""Export the high-platform CTS policy into its own deployment policy directory."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import export_main


if __name__ == "__main__":
    export_main("g12_high_platform")

