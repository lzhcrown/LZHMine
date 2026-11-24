"""Fixed WF platform geometry for inspecting a deployed student in Isaac Gym."""

import math

from legged_gym.terrains.wf import default_wf_sub_terrains


def configure_fixed_platform(terrain, shape, height, num_envs, width=None):
    """Override the restored play terrain, returning its quantized height in metres."""
    if shape is None and height is None and width is None:
        return None
    if shape not in ("block", "pit") or height is None:
        raise ValueError("Supply both --platform_shape {block,pit} and --platform_height METRES")
    if terrain.mesh_type not in ("trimesh", "heightfield") or terrain.generator_profile != "legbot_wf":
        raise ValueError("Fixed platform play requires a WF trimesh or heightfield")
    if not math.isfinite(height) or height <= 0:
        raise ValueError("--platform_height must be finite and positive")
    samples = round(height / terrain.vertical_scale)
    if not 1 <= samples <= 32767:
        raise ValueError("--platform_height must fit a nonzero signed 16-bit terrain sample")
    kind = "platform_" + shape
    specs = {name: dict(spec, proportion=0.0) for name, spec in terrain.wf_sub_terrains.items()}
    # Older maps store scattered obstacles or a pit under 'high_platform'.
    if kind not in specs:
        specs[kind] = default_wf_sub_terrains()[kind]
    if width is not None:
        if not math.isfinite(width) or width <= 0:
            raise ValueError("--platform_width must be finite and positive")
        spacing = specs[kind].get("horizontal_scale", terrain.horizontal_scale)
        side_cells = round(width / spacing)
        usable_cells = (int(min(terrain.terrain_length, terrain.terrain_width) / spacing) + 1
                        - 2 * (int(terrain.sub_terrain_border_width / spacing) + 1))
        actual_width = side_cells * spacing
        if not 0 < side_cells < usable_cells - 1 or (terrain.terrain_length - actual_width) / 4 < 0.5:
            raise ValueError("--platform_width must fit the WF tile and leave a clear approach")
        if shape == "pit" and actual_width < 2 + 2 * spacing:
            raise ValueError("--platform_width must cover the WF pit spawn height sampling window")
        specs[kind]["platform_width"] = actual_width
    specs[kind].update(proportion=1.0, platform_height_range=[height, height], round_height=True)
    terrain.wf_sub_terrains = specs
    terrain.terrain_proportions = [spec["proportion"] for spec in specs.values()]
    terrain.curriculum = False
    terrain.num_rows = 1
    terrain.num_cols = min(terrain.num_cols, num_envs)
    terrain.max_init_terrain_level = 0
    terrain.yaw_init_range = 0.0
    terrain.spawn_xy_range = 0.0
    return samples * terrain.vertical_scale
