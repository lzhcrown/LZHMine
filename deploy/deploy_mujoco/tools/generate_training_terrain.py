"""Export the exact Nezha training curriculum heightfield for MuJoCo.

This script imports ``NezhaMINECfg`` and the same ``Terrain`` class used by
Isaac Gym. It does not maintain a second hand-written terrain definition.
Run it with LZHMine's training virtual environment whenever the training
terrain configuration changes.
"""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legged_gym.envs.nezha.nezha_mine_config import (
    NezhaMINECfg,
    NezhaMINECfgPPO,
)
from legged_gym.utils.terrain import Terrain


DEPLOY_DIR = Path(__file__).resolve().parents[1]
ASSET_DIR = DEPLOY_DIR / "nezha" / "mjcf" / "assets"
IMAGE_PATH = ASSET_DIR / "nezha_training_curriculum.png"
BINARY_PATH = ASSET_DIR / "nezha_training_curriculum.bin"
METADATA_PATH = ASSET_DIR / "nezha_training_curriculum.json"

def _column_labels(cfg):
    cumulative = np.cumsum(cfg.terrain_proportions)
    labels = []
    for column in range(cfg.num_cols):
        choice = column / cfg.num_cols + 0.001
        if choice < cumulative[0]:
            label = "plane"
        elif choice < cumulative[1]:
            midpoint = cumulative[0] + (cumulative[1] - cumulative[0]) / 2
            label = "slope_down" if choice < midpoint else "slope_up"
        elif choice < cumulative[2]:
            midpoint = cumulative[1] + (cumulative[2] - cumulative[1]) / 2
            label = "rough_slope_down" if choice < midpoint else "rough_slope_up"
        elif choice < cumulative[3]:
            label = "stairs_down"
        else:
            label = "stairs_up"
        labels.append(label)
    return labels


def main():
    env_cfg = NezhaMINECfg()
    terrain_cfg = env_cfg.terrain
    if terrain_cfg.mesh_type != "trimesh":
        raise ValueError("Nezha training terrain must be configured as trimesh")
    if not terrain_cfg.curriculum:
        raise ValueError("Expected the Nezha curriculum terrain configuration")

    # task_registry.set_seed() runs before Terrain during training. Terrain
    # generation consumes NumPy's legacy global RNG, so reproduce that state.
    np.random.seed(NezhaMINECfgPPO.seed)
    terrain = Terrain(terrain_cfg, env_cfg.env.num_envs)
    heights_m = terrain.height_field_raw.astype(np.float64) * terrain_cfg.vertical_scale
    minimum = float(heights_m.min())
    maximum = float(heights_m.max())
    if maximum <= minimum:
        raise ValueError("Training terrain heightfield has no height range")

    # MuJoCo heightfield samples are normalized to [0, 1]. Use the training
    # extrema as the normalization extrema; the MJCF maps them back to metres.
    normalized = (heights_m - minimum) / (maximum - minimum)
    # Training stores height_field_raw[x, y], while MuJoCo's row-major matrix
    # is hfield_data[y, x]. A float32 binary asset avoids PNG's 8-bit elevation
    # quantization, preserving every 0.005 m training height sample.
    binary_data = np.ascontiguousarray(normalized.T, dtype=np.float32)
    # Keep a conventional preview image for inspection only. MuJoCo flips PNG
    # rows when loading, so its display-oriented representation needs flipud.
    pixels = np.rint(np.flipud(normalized.T) * 255.0).astype(np.uint8)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="L").save(IMAGE_PATH)
    binary_header = np.asarray(binary_data.shape, dtype=np.int32).tobytes()
    BINARY_PATH.write_bytes(binary_header + binary_data.tobytes())

    raw_digest = hashlib.sha256(terrain.height_field_raw.tobytes()).hexdigest()
    metadata = {
        "generator": "legged_gym.utils.terrain.Terrain",
        "profile": terrain_cfg.generator_profile,
        "seed": NezhaMINECfgPPO.seed,
        "mesh_type_training": terrain_cfg.mesh_type,
        "curriculum": terrain_cfg.curriculum,
        "num_rows": terrain_cfg.num_rows,
        "num_cols": terrain_cfg.num_cols,
        "terrain_length_m": terrain_cfg.terrain_length,
        "terrain_width_m": terrain_cfg.terrain_width,
        "horizontal_scale_m": terrain_cfg.horizontal_scale,
        "vertical_scale_m": terrain_cfg.vertical_scale,
        "border_size_m": terrain_cfg.border_size,
        "terrain_proportions": terrain_cfg.terrain_proportions,
        "max_init_terrain_level": terrain_cfg.max_init_terrain_level,
        "column_types": _column_labels(terrain_cfg),
        "env_origins": terrain.env_origins.tolist(),
        "height_shape": list(terrain.height_field_raw.shape),
        "image_shape": list(pixels.shape),
        "binary_shape": list(binary_data.shape),
        "height_min_m": minimum,
        "height_max_m": maximum,
        "encoding_min_m": minimum,
        "encoding_max_m": maximum,
        "height_field_sha256": raw_digest,
        "binary_data_sha256": hashlib.sha256(binary_data.tobytes()).hexdigest(),
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {BINARY_PATH} ({binary_data.shape[1]}x{binary_data.shape[0]}), "
        f"height range [{minimum:.3f}, {maximum:.3f}] m"
    )
    print(f"wrote preview {IMAGE_PATH}")
    print(f"wrote {METADATA_PATH}")
    print(f"training heightfield sha256: {raw_digest}")


if __name__ == "__main__":
    main()
