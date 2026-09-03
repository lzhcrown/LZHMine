"""Build a fixed MuJoCo terrain gallery from Nezha's training curriculum.

The robot starts on a shared flat staging area. Seven training terrain samples
are arranged side-by-side in front of it, allowing the operator to select a
lane with lateral velocity and then drive forward without restarting MuJoCo.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from legged_gym.envs.nezha.nezha_mine_config import (
    NezhaMINECfg,
    NezhaMINECfgPPO,
)
from legged_gym.utils.terrain import Terrain


ASSET_DIR = Path(__file__).resolve().parent / "models" / "assets"
BINARY_PATH = ASSET_DIR / "nezha_terrain_course.bin"
METADATA_PATH = ASSET_DIR / "nezha_terrain_course.json"

# Symmetric layout: negative-y lanes descend, positive-y lanes ascend.
# Columns refer directly to the 20-column training curriculum.
LANE_SOURCES = (
    ("stairs_down", 11),
    ("rough_slope_down", 4),
    ("slope_down", 2),
    ("plane", 0),
    ("slope_up", 3),
    ("rough_slope_up", 8),
    ("stairs_up", 15),
)


def main():
    env_cfg = NezhaMINECfg()
    cfg = env_cfg.terrain
    if cfg.mesh_type != "trimesh" or not cfg.curriculum:
        raise ValueError("The Nezha training configuration must use curriculum trimesh")

    # Reproduce the same generated curriculum and select actual samples from
    # its maximum initial difficulty row (row 5 in the current configuration).
    np.random.seed(NezhaMINECfgPPO.seed)
    training = Terrain(cfg, env_cfg.env.num_envs)
    source_row = min(int(cfg.max_init_terrain_level), int(cfg.num_rows) - 1)
    scale = float(cfg.horizontal_scale)
    tile_x = int(cfg.terrain_length / scale)
    tile_y = int(cfg.terrain_width / scale)

    staging_length_m = 10.0
    runout_length_m = 5.0
    lane_width_m = 5.0
    side_margin_m = 2.5
    obstacle_length_m = float(cfg.terrain_length)
    course_length_m = staging_length_m + obstacle_length_m + runout_length_m
    course_width_m = len(LANE_SOURCES) * lane_width_m + 2.0 * side_margin_m
    course_x = int(round(course_length_m / scale)) + 1
    course_y = int(round(course_width_m / scale)) + 1
    course_raw = np.zeros((course_x, course_y), dtype=np.int16)

    obstacle_x0 = int(round(staging_length_m / scale))
    lane_width_px = int(round(lane_width_m / scale))
    side_margin_px = int(round(side_margin_m / scale))
    source_y0 = (tile_y - lane_width_px) // 2
    border_px = int(round(cfg.border_size / scale))
    source_x0 = border_px + source_row * tile_x

    lanes = []
    for lane_index, (name, source_col) in enumerate(LANE_SOURCES):
        source_tile_y0 = border_px + source_col * tile_y + source_y0
        source = training.height_field_raw[
            source_x0 : source_x0 + tile_x,
            source_tile_y0 : source_tile_y0 + lane_width_px,
        ]
        destination_y0 = side_margin_px + lane_index * lane_width_px
        course_raw[
            obstacle_x0 : obstacle_x0 + tile_x,
            destination_y0 : destination_y0 + lane_width_px,
        ] = source
        y_min = -course_width_m / 2.0 + destination_y0 * scale
        lanes.append(
            {
                "name": name,
                "source_curriculum_row": source_row,
                "source_curriculum_col": source_col,
                "y_min_m": y_min,
                "y_max_m": y_min + lane_width_m,
                "y_center_m": y_min + lane_width_m / 2.0,
                "obstacle_x_start_m": 0.0,
                "obstacle_x_end_m": obstacle_length_m,
            }
        )

    heights_m = course_raw.astype(np.float64) * float(cfg.vertical_scale)
    height_min = float(heights_m.min())
    height_max = float(heights_m.max())
    if height_max <= height_min:
        raise ValueError("Terrain course has no elevation range")
    normalized = (heights_m - height_min) / (height_max - height_min)
    binary_data = np.ascontiguousarray(normalized.T, dtype=np.float32)
    binary_header = np.asarray(binary_data.shape, dtype=np.int32).tobytes()

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    BINARY_PATH.write_bytes(binary_header + binary_data.tobytes())

    world_x_min = -staging_length_m
    world_y_min = -course_width_m / 2.0
    half_x = (course_x - 1) * scale / 2.0
    half_y = (course_y - 1) * scale / 2.0
    metadata = {
        "generator": "legged_gym.utils.terrain.Terrain",
        "profile": cfg.generator_profile,
        "training_seed": int(NezhaMINECfgPPO.seed),
        "source_difficulty": source_row / int(cfg.num_rows),
        "horizontal_scale_m": scale,
        "vertical_scale_m": float(cfg.vertical_scale),
        "binary_shape": list(binary_data.shape),
        "height_min_m": height_min,
        "height_max_m": height_max,
        "hfield_size": [half_x, half_y, height_max - height_min, 0.1],
        "hfield_position": [world_x_min + half_x, world_y_min + half_y, height_min],
        "world_bounds": {
            "x": [world_x_min, world_x_min + course_length_m],
            "y": [world_y_min, world_y_min + course_width_m],
        },
        "spawn_ground_position": [-5.0, 0.0, 0.0],
        "staging_area": {"x": [world_x_min, 0.0], "surface": "plane"},
        "runout_area": {"x": [obstacle_length_m, world_x_min + course_length_m], "surface": "plane"},
        "lanes": lanes,
        "height_field_sha256": hashlib.sha256(course_raw.tobytes()).hexdigest(),
        "binary_data_sha256": hashlib.sha256(binary_data.tobytes()).hexdigest(),
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {BINARY_PATH}: {course_length_m:.1f} x {course_width_m:.1f} m, "
        f"height [{height_min:.3f}, {height_max:.3f}] m"
    )
    print(f"wrote {METADATA_PATH}")
    for lane in lanes:
        print(f"  y={lane['y_center_m']:+5.1f} m  {lane['name']}")


if __name__ == "__main__":
    main()
