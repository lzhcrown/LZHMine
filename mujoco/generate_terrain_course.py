"""Generate the three-lane MuJoCo evaluation course.

The course is intentionally independent of Isaac Gym so it can be rebuilt on
macOS with only NumPy installed. From left to right in the viewer it contains
20 cm stairs, a 40 cm rectangular platform, and deterministic rubble terrain.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


ASSET_DIR = Path(__file__).resolve().parent / "models" / "assets"
BINARY_PATH = ASSET_DIR / "nezha_terrain_course.bin"
METADATA_PATH = ASSET_DIR / "nezha_terrain_course.json"

HORIZONTAL_SCALE_M = 0.05
STAGING_LENGTH_M = 10.0
OBSTACLE_LENGTH_M = 10.0
RUNOUT_LENGTH_M = 5.0
LANE_WIDTH_M = 5.0
SIDE_MARGIN_M = 2.5
STAIR_HEIGHT_M = 0.20
STAIR_TREAD_M = 0.70
STAIR_COUNT = 5
STAIR_PLATFORM_M = 1.0
BLOCK_HEIGHT_M = 0.40
BLOCK_X_RANGE_M = (1.5, 7.5)
RUBBLE_MAX_HEIGHT_M = 0.12
RUBBLE_SEED = 20260904

LANES = (
    ("stairs_20cm", -5.0),
    ("platform_40cm", 0.0),
    ("rubble", 5.0),
)


def _interval_mask(values, lower, upper):
    return (values >= lower) & (values < upper)


def _stairs_profile(x):
    """Five 20 cm steps up, a flat top, and five steps down."""
    profile = np.zeros_like(x)
    ascent_start = 1.0
    ascent_end = ascent_start + STAIR_COUNT * STAIR_TREAD_M
    top_end = ascent_end + STAIR_PLATFORM_M
    descent_end = top_end + STAIR_COUNT * STAIR_TREAD_M

    ascending = _interval_mask(x, ascent_start, ascent_end)
    profile[ascending] = (
        np.floor((x[ascending] - ascent_start) / STAIR_TREAD_M) + 1
    ) * STAIR_HEIGHT_M
    profile[_interval_mask(x, ascent_end, top_end)] = (
        STAIR_COUNT * STAIR_HEIGHT_M
    )
    descending = _interval_mask(x, top_end, descent_end)
    descent_step = np.floor(
        (x[descending] - top_end) / STAIR_TREAD_M
    ).astype(int)
    profile[descending] = (
        STAIR_COUNT - 1 - descent_step
    ) * STAIR_HEIGHT_M
    return profile


def _rubble_surface(x_grid, y_grid):
    """Create a dense field of smooth, rock-like positive bumps."""
    rng = np.random.default_rng(RUBBLE_SEED)
    rubble = np.zeros_like(x_grid)
    for _ in range(170):
        center_x = rng.uniform(0.6, 9.4)
        center_y = rng.uniform(-2.25, 2.25)
        radius_x = rng.uniform(0.10, 0.28)
        radius_y = rng.uniform(0.10, 0.32)
        height = rng.uniform(0.025, RUBBLE_MAX_HEIGHT_M)
        distance = (
            ((x_grid - center_x) / radius_x) ** 2
            + ((y_grid - center_y) / radius_y) ** 2
        )
        rubble = np.maximum(rubble, height * np.exp(-2.2 * distance))

    # Fade the bumps into the flat approach, runout, and lane boundaries.
    x_fade = np.minimum(
        np.clip((x_grid - 0.25) / 0.50, 0.0, 1.0),
        np.clip((9.75 - x_grid) / 0.50, 0.0, 1.0),
    )
    y_fade = np.clip((2.5 - np.abs(y_grid)) / 0.25, 0.0, 1.0)
    rubble *= x_fade * y_fade
    peak = float(rubble.max())
    if peak > 0.0:
        rubble *= RUBBLE_MAX_HEIGHT_M / peak
    return rubble


def main():
    course_length_m = STAGING_LENGTH_M + OBSTACLE_LENGTH_M + RUNOUT_LENGTH_M
    course_width_m = len(LANES) * LANE_WIDTH_M + 2.0 * SIDE_MARGIN_M
    world_x_min = -STAGING_LENGTH_M
    world_y_min = -course_width_m / 2.0

    x_count = int(round(course_length_m / HORIZONTAL_SCALE_M)) + 1
    y_count = int(round(course_width_m / HORIZONTAL_SCALE_M)) + 1
    world_x = world_x_min + np.arange(x_count) * HORIZONTAL_SCALE_M
    world_y = world_y_min + np.arange(y_count) * HORIZONTAL_SCALE_M
    heights_m = np.zeros((x_count, y_count), dtype=np.float64)

    obstacle_x = world_x
    lane_metadata = []
    for name, center_y in LANES:
        lane_mask = np.abs(world_y - center_y) <= LANE_WIDTH_M / 2.0
        if name == "stairs_20cm":
            heights_m[:, lane_mask] = _stairs_profile(obstacle_x)[:, None]
            parameters = {
                "step_height_m": STAIR_HEIGHT_M,
                "tread_depth_m": STAIR_TREAD_M,
                "step_count": STAIR_COUNT,
                "top_height_m": STAIR_COUNT * STAIR_HEIGHT_M,
            }
        elif name == "platform_40cm":
            platform_mask = _interval_mask(
                obstacle_x, BLOCK_X_RANGE_M[0], BLOCK_X_RANGE_M[1]
            )
            heights_m[np.ix_(platform_mask, lane_mask)] = BLOCK_HEIGHT_M
            parameters = {
                "height_m": BLOCK_HEIGHT_M,
                "x_range_m": list(BLOCK_X_RANGE_M),
            }
        elif name == "rubble":
            x_grid, relative_y_grid = np.meshgrid(
                obstacle_x, world_y[lane_mask] - center_y, indexing="ij"
            )
            heights_m[:, lane_mask] = _rubble_surface(x_grid, relative_y_grid)
            parameters = {
                "maximum_bump_height_m": RUBBLE_MAX_HEIGHT_M,
                "seed": RUBBLE_SEED,
                "bump_count": 170,
            }
        else:
            raise ValueError(f"Unsupported lane: {name}")

        lane_metadata.append(
            {
                "name": name,
                "y_min_m": center_y - LANE_WIDTH_M / 2.0,
                "y_max_m": center_y + LANE_WIDTH_M / 2.0,
                "y_center_m": center_y,
                "obstacle_x_start_m": 0.0,
                "obstacle_x_end_m": OBSTACLE_LENGTH_M,
                **parameters,
            }
        )

    # Heightfields use normalized float32 samples and a world-space z range.
    height_min = 0.0
    height_max = float(heights_m.max())
    normalized = heights_m / height_max
    binary_data = np.ascontiguousarray(normalized.T, dtype=np.float32)
    binary_header = np.asarray(binary_data.shape, dtype=np.int32).tobytes()

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    BINARY_PATH.write_bytes(binary_header + binary_data.tobytes())

    half_x = (x_count - 1) * HORIZONTAL_SCALE_M / 2.0
    half_y = (y_count - 1) * HORIZONTAL_SCALE_M / 2.0
    height_samples_mm = np.rint(heights_m * 1000.0).astype(np.int16)
    metadata = {
        "generator": "mujoco.generate_terrain_course",
        "profile": "three_lane_visual_course",
        "horizontal_scale_m": HORIZONTAL_SCALE_M,
        "binary_shape": list(binary_data.shape),
        "height_min_m": height_min,
        "height_max_m": height_max,
        "hfield_size": [half_x, half_y, height_max, 0.1],
        "hfield_position": [world_x_min + half_x, 0.0, height_min],
        "world_bounds": {
            "x": [world_x_min, world_x_min + course_length_m],
            "y": [world_y_min, world_y_min + course_width_m],
        },
        "spawn_ground_position": [-5.0, 0.0, 0.0],
        "staging_area": {"x": [world_x_min, 0.0], "surface": "plane"},
        "runout_area": {
            "x": [OBSTACLE_LENGTH_M, world_x_min + course_length_m],
            "surface": "plane",
        },
        "lanes": lane_metadata,
        "height_field_sha256": hashlib.sha256(
            height_samples_mm.tobytes()
        ).hexdigest(),
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
    for lane in lane_metadata:
        print(f"  y={lane['y_center_m']:+4.1f} m  {lane['name']}")


if __name__ == "__main__":
    main()
