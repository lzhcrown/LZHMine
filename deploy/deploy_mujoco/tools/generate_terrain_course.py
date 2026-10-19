"""Generate the seven-lane MuJoCo evaluation course using only NumPy.

Stairs use separate solid boxes, and smooth ramps use convex wedge meshes, so
neither geometry depends on the heightfield resolution. Only the rough lane
uses the deterministic heightfield. Run this file to rebuild all three assets:
the heightfield binary, its metadata, and the included MJCF geometry.
"""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


DEPLOY_DIR = Path(__file__).resolve().parents[1]
MJCF_DIR = DEPLOY_DIR / "nezha" / "mjcf"
ASSET_DIR = MJCF_DIR / "assets"
BINARY_PATH = ASSET_DIR / "nezha_terrain_course.bin"
METADATA_PATH = ASSET_DIR / "nezha_terrain_course.json"
GEOMETRY_PATH = MJCF_DIR / "course_geometry.xml"

HORIZONTAL_SCALE_M = 0.05
STAGING_LENGTH_M = 10.0
OBSTACLE_LENGTH_M = 10.0
RUNOUT_LENGTH_M = 5.0
LANE_WIDTH_M = 5.0
SIDE_MARGIN_M = 2.5
STAIR_HEIGHTS_M = (0.15, 0.20, 0.25)
STAIR_TREAD_M = 0.30
STAIR_COUNT = 5
STAIR_PLATFORM_M = 1.0
STAIR_START_X_M = 1.0
RAMP_ANGLES_DEG = (15.0, 25.0, 30.0)
RAMP_START_X_M = 1.0
RAMP_HORIZONTAL_LENGTH_M = 3.5
RAMP_PLATFORM_M = 1.0
GROUND_FRICTION = (1.0, 0.01, 0.001)
RAMP_FRICTION = (0.35, 0.01, 0.001)
RAMP_SOLREF = (0.0125, 1.0)
RUBBLE_MAX_HEIGHT_M = 0.20
RUBBLE_SEED = 20260904
RUBBLE_BUMP_COUNT = 170
SPEED_BUMP_CENTERS_X_M = (1.5, 3.5, 5.5, 7.5, 9.0)
SPEED_BUMP_HEIGHTS_M = (0.12, 0.16, 0.20, 0.16, 0.20)
SPEED_BUMP_LENGTH_M = 0.80

LANES = (
    ("stairs_15cm", -15.0),
    ("stairs_20cm", -10.0),
    ("stairs_25cm", -5.0),
    ("ramp_15deg", 0.0),
    ("ramp_25deg", 5.0),
    ("ramp_30deg", 10.0),
    ("rubble_speed_bumps", 15.0),
)


def _numbers(values):
    return " ".join(format(float(value), ".12g") for value in values)


def _indent_xml(element, level=0):
    """Keep generated MJCF readable, including on Python 3.8."""
    prefix = "\n" + "  " * level
    if len(element):
        element.text = prefix + "  "
        for child in element:
            _indent_xml(child, level + 1)
        child.tail = prefix
    if level:
        element.tail = prefix


def _add_geom(worldbody, name, geom_type, position, material, friction, **attrs):
    return ET.SubElement(
        worldbody,
        "geom",
        name=name,
        type=geom_type,
        pos=_numbers(position),
        material=material,
        contype="2",
        conaffinity="1",
        condim="3",
        friction=_numbers(friction),
        **attrs,
    )


def _add_box(worldbody, name, center_y, x_start, x_end, height, material,
             friction=GROUND_FRICTION, **attrs):
    _add_geom(
        worldbody, name, "box",
        ((x_start + x_end) / 2.0, center_y, height / 2.0),
        material, friction,
        size=_numbers(((x_end - x_start) / 2.0, LANE_WIDTH_M / 2.0, height / 2.0)),
        **attrs,
    )
    return {
        "geom_name": name,
        "x_range_m": [x_start, x_end],
        "height_m": height,
    }


def _stairs_geometry(worldbody, name, center_y, height):
    """Five risers per side, four 30 cm treads, and exactly 1 m at the top.

    The fifth upward riser reaches the top platform. It does not introduce an
    additional 30 cm tread contiguous with the platform. This makes the top
    surface exactly 1 m, and the total obstacle length 3.4 m.
    """
    surfaces = []
    x = STAIR_START_X_M
    for level in range(1, STAIR_COUNT):
        surfaces.append(_add_box(
            worldbody, f"{name}_up_{level}", center_y, x, x + STAIR_TREAD_M,
            level * height, "coursestair_light" if level % 2 else "coursestair_dark",
        ))
        x += STAIR_TREAD_M
    platform_start = x
    surfaces.append(_add_box(
        worldbody, f"{name}_platform", center_y, x, x + STAIR_PLATFORM_M,
        STAIR_COUNT * height, "coursestair_top",
    ))
    x += STAIR_PLATFORM_M
    for level in range(STAIR_COUNT - 1, 0, -1):
        surfaces.append(_add_box(
            worldbody, f"{name}_down_{level}", center_y, x, x + STAIR_TREAD_M,
            level * height, "coursestair_light" if level % 2 else "coursestair_dark",
        ))
        x += STAIR_TREAD_M
    return {
        "surface_type": "box_stairs",
        "step_height_m": height,
        "tread_depth_m": STAIR_TREAD_M,
        "step_count": STAIR_COUNT,
        "riser_count_per_side": STAIR_COUNT,
        "intermediate_tread_count_per_side": STAIR_COUNT - 1,
        "platform_length_m": STAIR_PLATFORM_M,
        "platform_x_range_m": [platform_start, platform_start + STAIR_PLATFORM_M],
        "top_height_m": STAIR_COUNT * height,
        "geometry_x_range_m": [STAIR_START_X_M, x],
        "friction": list(GROUND_FRICTION),
        "geom_names": [surface["geom_name"] for surface in surfaces],
        "surfaces": surfaces,
    }


def _ramp_geometry(asset, worldbody, name, center_y, angle_deg):
    height = float(np.tan(np.deg2rad(angle_deg)) * RAMP_HORIZONTAL_LENGTH_M)
    half_length = RAMP_HORIZONTAL_LENGTH_M / 2.0
    half_width = LANE_WIDTH_M / 2.0
    # A closed convex triangular prism: a truly planar incline with no sampling
    # steps. Descending geometry reuses this mesh rotated 180 degrees about z.
    vertices = (
        (-half_length, -half_width, 0), (half_length, -half_width, 0),
        (half_length, -half_width, height), (-half_length, half_width, 0),
        (half_length, half_width, 0), (half_length, half_width, height),
    )
    faces = ((0, 1, 2), (3, 5, 4), (0, 3, 4), (0, 4, 1),
             (1, 4, 5), (1, 5, 2), (0, 2, 5), (0, 5, 3))
    mesh_name = f"{name}_wedge"
    ET.SubElement(
        asset, "mesh", name=mesh_name,
        vertex=_numbers(value for vertex in vertices for value in vertex),
        face=" ".join(str(value) for face in faces for value in face),
    )
    platform_start = RAMP_START_X_M + RAMP_HORIZONTAL_LENGTH_M
    platform_end = platform_start + RAMP_PLATFORM_M
    ramp_end = platform_end + RAMP_HORIZONTAL_LENGTH_M
    contact = {"priority": "1", "solref": _numbers(RAMP_SOLREF)}
    _add_geom(
        worldbody, f"{name}_up", "mesh",
        (RAMP_START_X_M + half_length, center_y, 0),
        "courseramp", RAMP_FRICTION, mesh=mesh_name, **contact,
    )
    _add_box(
        worldbody, f"{name}_platform", center_y, platform_start, platform_end,
        height, "courseramp", friction=RAMP_FRICTION, **contact,
    )
    _add_geom(
        worldbody, f"{name}_down", "mesh",
        (platform_end + half_length, center_y, 0),
        "courseramp", RAMP_FRICTION, mesh=mesh_name, quat="0 0 0 1", **contact,
    )
    return {
        "surface_type": "smooth_wedge_ramp",
        "slope_degrees": angle_deg,
        "ramp_horizontal_length_m": RAMP_HORIZONTAL_LENGTH_M,
        "ramp_surface_length_m": float(RAMP_HORIZONTAL_LENGTH_M / np.cos(np.deg2rad(angle_deg))),
        "platform_length_m": RAMP_PLATFORM_M,
        "top_height_m": height,
        "up_x_range_m": [RAMP_START_X_M, platform_start],
        "platform_x_range_m": [platform_start, platform_end],
        "down_x_range_m": [platform_end, ramp_end],
        "geometry_x_range_m": [RAMP_START_X_M, ramp_end],
        "friction": list(RAMP_FRICTION),
        "contact_priority": 1,
        "contact_dimension": 3,
        "solref": list(RAMP_SOLREF),
        "geom_names": [f"{name}_up", f"{name}_platform", f"{name}_down"],
    }


def _rubble_surface(x_grid, y_grid):
    """Deterministic roughness plus five rounded transverse speed bumps."""
    rng = np.random.default_rng(RUBBLE_SEED)
    rubble = np.zeros_like(x_grid)
    for _ in range(RUBBLE_BUMP_COUNT):
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

    # Raised cosines have zero slope at both ends. Taking max, rather than
    # adding bumps, preserves the specified 20 cm peak height.
    for center, height in zip(SPEED_BUMP_CENTERS_X_M, SPEED_BUMP_HEIGHTS_M):
        phase = (x_grid - center) / (SPEED_BUMP_LENGTH_M / 2.0)
        bump = np.where(np.abs(phase) <= 1.0,
                        0.5 * height * (1.0 + np.cos(np.pi * phase)), 0.0)
        rubble = np.maximum(rubble, bump)

    x_fade = np.minimum(
        np.clip((x_grid - 0.25) / 0.50, 0.0, 1.0),
        np.clip((9.75 - x_grid) / 0.50, 0.0, 1.0),
    )
    y_fade = np.clip((LANE_WIDTH_M / 2.0 - np.abs(y_grid)) / 0.25, 0.0, 1.0)
    return np.minimum(rubble * x_fade * y_fade, RUBBLE_MAX_HEIGHT_M)


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

    geometry = ET.Element("mujocoinclude")
    geometry.append(ET.Comment("Generated by tools/generate_terrain_course.py; edit its constants and regenerate."))
    asset = ET.SubElement(geometry, "asset")
    for material_name, rgba in (
        ("coursestair_light", "0.12 0.42 0.64 1"),
        ("coursestair_dark", "0.075 0.29 0.48 1"),
        ("coursestair_top", "0.20 0.55 0.74 1"),
    ):
        ET.SubElement(asset, "material", name=material_name, rgba=rgba,
                      specular="0.20", shininess="0.30")
    ET.SubElement(asset, "material", name="courseramp", rgba="0.14 0.48 0.60 1",
                  specular="0.5", shininess="0.65", reflectance="0.12")
    worldbody = ET.SubElement(geometry, "worldbody")

    lane_metadata = []
    for index, (name, center_y) in enumerate(LANES):
        if index < len(STAIR_HEIGHTS_M):
            parameters = _stairs_geometry(worldbody, name, center_y, STAIR_HEIGHTS_M[index])
        elif index < len(STAIR_HEIGHTS_M) + len(RAMP_ANGLES_DEG):
            angle = RAMP_ANGLES_DEG[index - len(STAIR_HEIGHTS_M)]
            parameters = _ramp_geometry(asset, worldbody, name, center_y, angle)
        else:
            lane_mask = np.abs(world_y - center_y) <= LANE_WIDTH_M / 2.0
            x_grid, relative_y_grid = np.meshgrid(
                world_x, world_y[lane_mask] - center_y, indexing="ij"
            )
            heights_m[:, lane_mask] = _rubble_surface(x_grid, relative_y_grid)
            parameters = {
                "surface_type": "rubble_with_rounded_speed_bumps",
                "maximum_bump_height_m": RUBBLE_MAX_HEIGHT_M,
                "seed": RUBBLE_SEED,
                "bump_count": RUBBLE_BUMP_COUNT,
                "speed_bump_centers_x_m": list(SPEED_BUMP_CENTERS_X_M),
                "speed_bump_heights_m": list(SPEED_BUMP_HEIGHTS_M),
                "speed_bump_length_m": SPEED_BUMP_LENGTH_M,
                "edge_fade_width_m": 0.25,
                "friction": list(GROUND_FRICTION),
                "geom_names": ["training_terrain"],
            }
        lane_metadata.append({
            "name": name,
            "width_m": LANE_WIDTH_M,
            "y_min_m": center_y - LANE_WIDTH_M / 2.0,
            "y_max_m": center_y + LANE_WIDTH_M / 2.0,
            "y_center_m": center_y,
            "obstacle_x_start_m": 0.0,
            "obstacle_x_end_m": OBSTACLE_LENGTH_M,
            **parameters,
        })

    # Background heightfield is exactly zero under stairs and ramps. Their
    # collision surfaces come solely from the boxes and convex wedges above.
    height_min = 0.0
    height_max = float(heights_m.max())
    binary_data = np.ascontiguousarray((heights_m / height_max).T, dtype=np.float32)
    binary_header = np.asarray(binary_data.shape, dtype=np.int32).tobytes()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    BINARY_PATH.write_bytes(binary_header + binary_data.tobytes())

    half_x = (x_count - 1) * HORIZONTAL_SCALE_M / 2.0
    half_y = (y_count - 1) * HORIZONTAL_SCALE_M / 2.0
    hfield_size = [half_x, half_y, height_max, 0.1]
    hfield_position = [world_x_min + half_x, 0.0, height_min]
    ET.SubElement(asset, "hfield", name="terrain_course",
                  file="../mjcf/assets/nezha_terrain_course.bin", size=_numbers(hfield_size))
    _add_geom(worldbody, "training_terrain", "hfield", hfield_position,
              "courseground", GROUND_FRICTION, hfield="terrain_course")
    _indent_xml(geometry)
    geometry_bytes = ET.tostring(geometry, encoding="utf-8") + b"\n"
    GEOMETRY_PATH.write_bytes(geometry_bytes)

    height_samples_mm = np.rint(heights_m * 1000.0).astype(np.int16)
    metadata = {
        "generator": "deploy.deploy_mujoco.tools.generate_terrain_course",
        "profile": "seven_lane_stairs_smooth_ramps_speed_bumps",
        "horizontal_scale_m": HORIZONTAL_SCALE_M,
        "binary_shape": list(binary_data.shape),
        "height_min_m": height_min,
        "height_max_m": height_max,
        "heightfield_scope": "flat ground and rubble lane only; stairs and ramps use separate solid geoms",
        "obstacle_height_max_m": max(lane.get("top_height_m", height_max) for lane in lane_metadata),
        "hfield_size": hfield_size,
        "hfield_position": hfield_position,
        "geometry_include": GEOMETRY_PATH.name,
        "geometry_sha256": hashlib.sha256(geometry_bytes).hexdigest(),
        "world_bounds": {
            "x": [world_x_min, world_x_min + course_length_m],
            "y": [world_y_min, world_y_min + course_width_m],
        },
        "spawn_ground_position": [-5.0, LANES[0][1], 0.0],
        "staging_area": {"x": [world_x_min, 0.0], "surface": "plane"},
        "runout_area": {
            "x": [OBSTACLE_LENGTH_M, world_x_min + course_length_m],
            "surface": "plane",
        },
        "lanes": lane_metadata,
        "height_field_sha256": hashlib.sha256(height_samples_mm.tobytes()).hexdigest(),
        "binary_data_sha256": hashlib.sha256(binary_data.tobytes()).hexdigest(),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {BINARY_PATH}: {course_length_m:.1f} x {course_width_m:.1f} m, "
          f"heightfield [{height_min:.3f}, {height_max:.3f}] m")
    print(f"wrote {GEOMETRY_PATH}")
    print(f"wrote {METADATA_PATH}")
    for lane in lane_metadata:
        print(f"  y={lane['y_center_m']:+5.1f} m  {lane['name']}")


if __name__ == "__main__":
    main()
