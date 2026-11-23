"""Export the exact Nezha training curriculum heightfield for MuJoCo.

This script imports ``NezhaMoECTSCfg`` and the same ``Terrain`` class used by
Isaac Gym. It does not maintain a second hand-written terrain definition.
Run it with LZHMine's training virtual environment whenever the training
terrain configuration changes.
"""

import copy
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import isaacgym  # noqa: F401 - must precede torch through the training config
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "rsl_rl"))

from legged_gym.envs.nezha.nezha_moe_cts_config import (
    NezhaMoECTSCfg,
    NezhaMoECTSCfgPPO,
)
from legged_gym.utils.terrain import Terrain


DEPLOY_DIR = Path(__file__).resolve().parents[1]
ASSET_DIR = DEPLOY_DIR / "nezha" / "mjcf" / "assets"
IMAGE_PATH = ASSET_DIR / "nezha_training_curriculum.png"
BINARY_PATH = ASSET_DIR / "nezha_training_curriculum.bin"
METADATA_PATH = ASSET_DIR / "nezha_training_curriculum.json"
SCENE_PATH = DEPLOY_DIR / "nezha" / "mjcf" / "scene_training.xml"

def _column_labels(cfg):
    if getattr(cfg, "generator_profile", None) == "legbot_wf":
        from legged_gym.terrains.wf import column_types
        return column_types(cfg)
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
    env_cfg = NezhaMoECTSCfg()
    terrain_cfg = env_cfg.terrain
    if terrain_cfg.mesh_type != "trimesh":
        raise ValueError("Nezha training terrain must be configured as trimesh")
    if not terrain_cfg.curriculum:
        raise ValueError("Expected the Nezha curriculum terrain configuration")

    # task_registry.set_seed() runs before Terrain during training. Terrain
    # generation consumes NumPy's legacy global RNG, so reproduce that state.
    np.random.seed(NezhaMoECTSCfgPPO.seed)
    # Export the finest physical grid, including the half-grid stair edges.
    # The default Gym trimesh stores a separate coarser height observation grid.
    export_cfg = copy.deepcopy(terrain_cfg)
    export_cfg.mesh_type = "heightfield"
    terrain = Terrain(export_cfg, env_cfg.env.num_envs)
    sample_spacing = terrain.cfg.horizontal_scale
    raw = terrain.height_field_raw
    low, high = int(raw.min()), int(raw.max())
    if high <= low:
        raise ValueError("Training terrain heightfield has no height range")
    minimum, maximum = low * terrain_cfg.vertical_scale, high * terrain_cfg.vertical_scale

    # MuJoCo heightfield samples are normalized to [0, 1]. Use the training
    # extrema as the normalization extrema; the MJCF maps them back to metres.
    # Training stores height_field_raw[x, y], while MuJoCo's row-major matrix
    # is hfield_data[y, x]. A float32 binary asset avoids PNG's 8-bit elevation
    # quantization, preserving every 0.005 m training height sample.
    binary_data = np.ascontiguousarray(raw.T, dtype=np.float32)
    binary_data -= low
    binary_data /= high - low
    # Keep a conventional preview image for inspection only. MuJoCo flips PNG
    # rows when loading, so its display-oriented representation needs flipud.
    pixels = np.rint(np.flipud(binary_data) * 255.0).astype(np.uint8)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="L").save(IMAGE_PATH)
    with BINARY_PATH.open("wb") as stream:
        np.asarray(binary_data.shape, dtype=np.int32).tofile(stream)
        binary_data.tofile(stream)

    raw_digest = hashlib.sha256(raw).hexdigest()
    metadata = {
        "generator": "legged_gym.utils.terrain.Terrain",
        "profile": terrain_cfg.generator_profile,
        "seed": NezhaMoECTSCfgPPO.seed,
        "mesh_type_training": terrain_cfg.mesh_type,
        "curriculum": terrain_cfg.curriculum,
        "num_rows": terrain_cfg.num_rows,
        "num_cols": terrain_cfg.num_cols,
        "terrain_length_m": terrain_cfg.terrain_length,
        "terrain_width_m": terrain_cfg.terrain_width,
        "horizontal_scale_m": sample_spacing,
        "vertical_scale_m": terrain_cfg.vertical_scale,
        "border_size_m": terrain_cfg.border_size,
        "terrain_proportions": terrain_cfg.terrain_proportions,
        "max_init_terrain_level": terrain_cfg.max_init_terrain_level,
        "column_types": _column_labels(terrain_cfg),
        "difficulty_range": getattr(terrain_cfg, "difficulty_range", [0.0, 1.0]),
        "difficulty_reference_rows": getattr(terrain_cfg, "difficulty_reference_rows", terrain_cfg.num_rows),
        "tile_difficulties": terrain.difficulties.tolist() if hasattr(terrain, "difficulties") else None,
        "sub_terrain_border_width_m": getattr(terrain_cfg, "sub_terrain_border_width", 0.0),
        "env_origins": terrain.env_origins.tolist(),
        "height_shape": list(terrain.height_field_raw.shape),
        "image_shape": list(pixels.shape),
        "binary_shape": list(binary_data.shape),
        "height_min_m": minimum,
        "height_max_m": maximum,
        "encoding_min_m": minimum,
        "encoding_max_m": maximum,
        "height_field_sha256": raw_digest,
        "binary_data_sha256": hashlib.sha256(binary_data).hexdigest(),
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    # Keep the optional full-map scene aligned with the regenerated dimensions
    # and elevation range instead of retaining the old 10x10 m Gym map values.
    scene = ET.parse(SCENE_PATH)
    scene.getroot().set("model", "nezha WF-CTS-MOE full-training-curriculum sim-to-sim")
    span_x = (terrain.tot_rows - 1) * sample_spacing
    span_y = (terrain.tot_cols - 1) * sample_spacing
    scene.find("./asset/hfield").set("size", "{} {} {} 0.1".format(span_x / 2, span_y / 2, maximum - minimum))
    scene.find("./worldbody/geom[@name='training_terrain']").set(
        "pos", "{} {} {}".format(span_x / 2 - terrain_cfg.border_size,
                                  span_y / 2 - terrain_cfg.border_size, minimum))
    scene_xml = ET.tostring(scene.getroot(), encoding="unicode")
    SCENE_PATH.write_text("\n".join(line.rstrip() for line in scene_xml.splitlines()) + "\n", encoding="utf-8")
    print(
        f"wrote {BINARY_PATH} ({binary_data.shape[1]}x{binary_data.shape[0]}), "
        f"height range [{minimum:.3f}, {maximum:.3f}] m"
    )
    print(f"wrote preview {IMAGE_PATH}")
    print(f"wrote {METADATA_PATH}")
    print(f"training heightfield sha256: {raw_digest}")


if __name__ == "__main__":
    main()
