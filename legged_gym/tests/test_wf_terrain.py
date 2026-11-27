"""CPU-only WF terrain regressions against Isaac Lab v2.3.2 reference data.

Golden hashes came from upstream hf_terrains.py and the original
height_field_to_mesh wrapper, using legbot_lab daff9fb's TERRAIN_CFG,
NumPy seed 17, and little-endian int16 array bytes. Tests need no Isaac Lab,
Isaac Gym, GPU, or network access.
"""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from legged_gym.terrains.wf import (
    WFTerrain,
    column_types,
    default_wf_sub_terrains,
    high_platform_wf_sub_terrains,
    make_height_field,
)


def terrain_cfg(**overrides):
    values = dict(
        wf_sub_terrains=default_wf_sub_terrains(),
        mesh_type="heightfield", terrain_length=9.0, terrain_width=9.0,
        horizontal_scale=0.1, vertical_scale=0.005, border_size=25.0,
        sub_terrain_border_width=0.5, num_rows=10, num_cols=20,
        curriculum=True, difficulty_range=(0.0, 1.0), seed=17,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture(autouse=True)
def preserve_numpy_random_state():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


# Upstream oracle:
# https://github.com/isaac-sim/IsaacLab/tree/v2.3.2/source/isaaclab/isaaclab/terrains/height_field
# Difficulties include both endpoints and values just below the upper endpoint
# to cover discretization rather than only visual similarity.
DIFFICULTIES = (0.0, 0.25, 0.555, 0.999, 1.0)
GOLDEN_HEIGHT_FIELDS = {
    "wave": (
        "f3b649ae93437df41a25b5fd11984801be4bf0fcd8af17856eccc9a4394da142",
        "db0ddcdccef3404dd22fcea9b4f4bf6b66a8ba184f334b03c9cb876c19754ead",
        "6b5671323f0a864a680a685d98ffb6ba4f6e891a658cf4526ea6bf8083167d93",
        "e300a129f6534909c832ef55d14d212db2a0756fbcad320be3d7e8f2ca44f035",
        "607277147e634babe6aa38e460f330cad42ecdb51580f05910597fd11fb59a4c",
    ),
    "rough_slope": (
        "5b01d5068a4607242f7f9860fc46b39f958b0dd2a9c1701d670f02b9ce690c61",
        "947464fbc1f6c55c3eb48007a8ffc570fd8dce2cc43ecea55f2fe1fadad7a622",
        "0b5d982490e3edbf91a94258b4014742c40a2ab51144a0f37b4eaf11cadefca5",
        "1abfae30b5f1f0525a6b656c4114375a9a9f391ed8bc0ca3d17ba97b3ad02482",
        "1abfae30b5f1f0525a6b656c4114375a9a9f391ed8bc0ca3d17ba97b3ad02482",
    ),
    "slope_up": (
        "dab49c779a6b92aa7bb7c0b05b4ffa623f3cc62419d7df45c8a932ae0bff9b3c",
        "b59c593b5b8eb62c849a4991627c9758980029adee97d1568a3c287d7137c5ce",
        "da310feb9938ac7285b5bccc799754c8ecc6d840772dc712d48e58f30334f5cf",
        "377c4cd03cc799b0be36547489280ddb0e593f672b6a3f2b8cd769fca6ce8ce3",
        "377c4cd03cc799b0be36547489280ddb0e593f672b6a3f2b8cd769fca6ce8ce3",
    ),
    "slope_down": (
        "32ce5379344ec37329b0d501d669cd34e1d79a16a2bf1a82577d0d99c78d41b3",
        "d6cb0392897083ea538666c54826b04242988d91bb4f072e8a9625774fb46f42",
        "e29d02dd65f01a1ff23c384d7cc4bc8857891985cf255e4b431decc6aabfb228",
        "b7a9a7aa71fb9854da093fdb3c7ddc2ffc269c877fb83a468cba21acead8709b",
        "b7a9a7aa71fb9854da093fdb3c7ddc2ffc269c877fb83a468cba21acead8709b",
    ),
    "stairs_up": (
        "7a8da673e87a263839517bb3427bcab704b8d6890de88f62dbde1d4d5d9eefd4",
        "99a76c1b4e0ab78634d71cdd7d958d7f1de6c3c6918f0cda5b9b58c4ed4fcbba",
        "4f78710d28c4802dfaaea618f51875fc3f771194d623a38661a683be7fbeaf56",
        "c25772ae43e9153cf3334e7d0d4edae5c8cf640a8c47228b1c927b60a463785e",
        "c25772ae43e9153cf3334e7d0d4edae5c8cf640a8c47228b1c927b60a463785e",
    ),
    "stairs_down": (
        "f16af92ea305029bd7a2915c74866678b35363d6b525ae5b158b8760f4d375bd",
        "65510a53283dacdca3f23da19a2a9bcb9e5af03834a64672959cc6b040e7c0a4",
        "7c5d119c7e559bccb8f931a70859f45238be5b3277d9fe966d625ff409566f1d",
        "854ad55cc22b44d9a4d4df695e6197d28e78ffe75c905f586b7f93cdfc8fc058",
        "854ad55cc22b44d9a4d4df695e6197d28e78ffe75c905f586b7f93cdfc8fc058",
    ),
    "obstacles": (
        "412a4e47353b482bffa003367ed02ad55a3949aac4f95230e82fa5d782a993f7",
        "039a955a8ae1a749cd645c65f62f3ec8e969a32e95df275316b2c868320c0082",
        "8a8d65b46a2f2050a288586703325290023972827c070002e53755548a4e36a2",
        "2eee93f0ff51b7105f123fb3d9f49633ef1e9c81c0897f2a279e28e47162630a",
        "9e047cb2db295e504aca61fce76078744a506c7fcd84abf6171550e382004495",
    ),
}


@pytest.mark.parametrize("name,difficulty,expected", [
    (name, difficulty, expected)
    for name, hashes in GOLDEN_HEIGHT_FIELDS.items()
    for difficulty, expected in zip(DIFFICULTIES, hashes)
])
def test_height_fields_match_upstream_golden(name, difficulty, expected):
    np.random.seed(17)
    cfg = terrain_cfg()
    if name == "obstacles":  # Original specification stored by earlier checkpoints.
        cfg.wf_sub_terrains = {"obstacles": dict(
            proportion=0.20, slope_threshold=0.25, obstacle_height_mode="choice",
            obstacle_width_range=[1.0, 2.0], obstacle_height_range=[0.05, 0.275],
            num_obstacles=20, platform_width=3.0,
        )}
    elif name in ("stairs_up", "stairs_down"):
        cfg.wf_sub_terrains[name] = dict(
            inverted=(name == "stairs_up"), step_height_range=[0.05, 0.257],
            step_width=0.31, platform_width=3.0,
        )
    heights = make_height_field(name, difficulty, cfg)
    assert heights.shape == (91, 91)
    assert heights.dtype == np.int16
    assert not np.any(heights[:6])
    assert not np.any(heights[-6:])
    assert not np.any(heights[:, :6])
    assert not np.any(heights[:, -6:])
    assert hashlib.sha256(heights.astype("<i2").tobytes()).hexdigest() == expected


@pytest.mark.parametrize("name,sign", [("stairs_up", -1), ("stairs_down", 1)])
@pytest.mark.parametrize("difficulty", [0.0, 0.97, 1.04, 1.999])
def test_all_four_stair_faces_have_35cm_treads(name, sign, difficulty):
    cfg = terrain_cfg()
    field = make_height_field(name, difficulty, cfg)
    spec = cfg.wf_sub_terrains[name]
    assert spec["step_width"] == .35
    assert spec["horizontal_scale"] == .05
    assert field.shape == (181, 181)
    step_height = sign * int((.05 + (.257 - .05) * difficulty) / cfg.vertical_scale)
    for cross_section in (field[:, 90], field[90, :]):
        for toward_center in (cross_section[:91], cross_section[90:][::-1]):
            rise = np.diff(toward_center)
            risers = np.flatnonzero(rise)
            assert len(risers) > 2
            np.testing.assert_array_equal(rise[risers], step_height)
            widths = np.diff(risers)
            np.testing.assert_array_equal(widths, 7)
            np.testing.assert_allclose(widths * spec["horizontal_scale"], .35)


@pytest.mark.parametrize("specs", [default_wf_sub_terrains, high_platform_wf_sub_terrains])
def test_platforms_share_the_previous_obstacle_columns(specs):
    cfg = terrain_cfg(wf_sub_terrains=specs())
    assert [spec["proportion"] for spec in cfg.wf_sub_terrains.values()] == [
        0.05, 0.10, 0.10, 0.05, 0.25, 0.10, 0.10, 0.10, 0.0, 0.0, 0.15,
    ]
    assert column_types(cfg) == (
        ["wave"] + ["slope_up"] * 2 + ["slope_down"] * 2 + ["rough_slope"]
        + ["stairs_up"] * 5 + ["stairs_down"] * 2 + ["platform_pit"] * 2
        + ["platform_block"] * 2 + ["flat"] * 3
    )
    assert not np.any(make_height_field("flat", 1.0, cfg))
    # A caller can edit a run's config without changing subsequent runs.
    cfg.wf_sub_terrains["wave"]["proportion"] = 0
    assert specs()["wave"]["proportion"] == 0.05


@pytest.mark.parametrize("specs,height_law", [
    (default_wf_sub_terrains, lambda d: int((0.05 + (0.275 - 0.05) * d) / 0.005)),
    (high_platform_wf_sub_terrains, lambda d: round((0.20 + (0.70 - 0.20) * d) / 0.005)),
])
@pytest.mark.parametrize("difficulty", [0.0, 0.25, 0.555, 0.999, 1.0, 1.55, 1.999])
def test_centered_single_step_in_each_tile_retains_height_law(specs, height_law, difficulty):
    cfg = terrain_cfg(wf_sub_terrains=specs())
    # 3.2 m square completely inside the tile with level margins on all sides.
    expected = np.zeros((91, 91), dtype=np.int16)
    expected[29:61, 29:61] = height_law(difficulty)
    np.testing.assert_array_equal(make_height_field("platform_block", difficulty, cfg), expected)
    np.testing.assert_array_equal(make_height_field("platform_pit", difficulty, cfg), -expected)


def test_original_g12_pit_snapshot_replays_the_same_geometry():
    cfg = terrain_cfg(wf_sub_terrains={"high_platform": dict(
        proportion=0.20, slope_threshold=0.25, pit_width=3.2, platform_height_range=[0.20, 0.70],
    )})
    expected = np.zeros((91, 91), dtype=np.int16)
    expected[29:61, 29:61] = -100  # 50 cm
    np.testing.assert_array_equal(make_height_field("high_platform", 0.6, cfg), expected)


@pytest.mark.parametrize("specs", [default_wf_sub_terrains, high_platform_wf_sub_terrains])
def test_spawn_on_pit_floor_or_ground_before_block_with_level_spawn_jitter(specs):
    cfg = terrain_cfg(wf_sub_terrains=specs(), num_rows=3)
    terrain = WFTerrain(cfg, 64, mesh_converter=None)
    assert np.all(terrain.tile_types[:, 13:15] == "platform_pit")
    assert np.all(terrain.tile_types[:, 15:17] == "platform_block")
    for row in range(cfg.num_rows):
        for col in range(13, 17):
            name = terrain.tile_types[row, col]
            x, y, z = terrain.env_origins[row, col]
            heights = make_height_field(name, terrain.difficulties[row, col], cfg)
            x0, y0 = terrain.border + row * 180, terrain.border + col * 180
            np.testing.assert_array_equal(terrain.height_field_raw[x0:x0+181:2, y0:y0+181:2], heights)
            assert y == (col + 0.5) * 9
            if name == "platform_pit":
                assert x == (row + 0.5) * 9
                assert z == heights.min() * cfg.vertical_scale < 0
            else:
                assert x == pytest.approx(row * 9 + 1.45)
                assert z == 0 < heights.max()
            xi = round((x - row * 9) / cfg.horizontal_scale)
            assert np.all(heights[xi-5:xi+6, 40:51] * cfg.vertical_scale == z)


def test_curriculum_difficulty_uses_column_major_independent_rng():
    np.random.seed(17)
    terrain = WFTerrain(terrain_cfg(), 64, mesh_converter=None)
    uniform = np.random.default_rng(17).uniform(size=(20, 10)).T
    expected = (np.arange(10)[:, None] + uniform) / 10
    np.testing.assert_array_equal(terrain.difficulties, expected)
    np.testing.assert_array_equal(terrain.tile_types, np.tile(column_types(terrain.cfg), (10, 1)))
    assert np.all(terrain.difficulties >= np.arange(10)[:, None] / 10)
    assert np.all(terrain.difficulties < (np.arange(10)[:, None] + 1) / 10)


def test_v5_keeps_v1_geometry_at_each_reachable_curriculum_level():
    common = dict(wf_sub_terrains=high_platform_wf_sub_terrains(), num_rows=20,
                  difficulty_reference_rows=10, border_size=0.)
    np.random.seed(17)
    v1 = WFTerrain(terrain_cfg(**common), 1, None)
    np.random.seed(17)
    v5 = WFTerrain(terrain_cfg(ordinary_max_level=9, **common), 1, None)
    np.testing.assert_array_equal(v5.tile_types, v1.tile_types)
    np.testing.assert_array_equal(v5.difficulties[:10], v1.difficulties[:10])
    np.testing.assert_array_equal(v5.height_field_raw[:10 * v5.length_per_env_pixels + 1],
                                  v1.height_field_raw[:10 * v1.length_per_env_pixels + 1])
    for col, kind in enumerate(v5.column_types):
        if kind in ("platform_block", "platform_pit"):
            np.testing.assert_array_equal(v5.difficulties[:, col], v1.difficulties[:, col])
            start = col * v5.width_per_env_pixels
            stop = start + v5.width_per_env_pixels + 1
            np.testing.assert_array_equal(v5.height_field_raw[:, start:stop], v1.height_field_raw[:, start:stop])
        else:
            assert np.all(v5.difficulties[10:, col] < 1.)


def test_v5_random_play_caps_ordinary_difficulty_by_actual_tile_type():
    terrain = WFTerrain(terrain_cfg(
        wf_sub_terrains=high_platform_wf_sub_terrains(), num_rows=20, difficulty_reference_rows=10,
        ordinary_max_level=9, curriculum=False, border_size=0.,
    ), 1, None)
    platforms = np.isin(terrain.tile_types, ("platform_block", "platform_pit"))
    assert np.all(terrain.difficulties[~platforms] < 1.)
    for kind in ("platform_block", "platform_pit"):
        assert terrain.difficulties[terrain.tile_types == kind].max() > 1.


def test_full_map_has_shared_edges_outer_border_and_correct_origins():
    np.random.seed(17)
    terrain = WFTerrain(terrain_cfg(), 64, mesh_converter=None)
    assert terrain.cfg.horizontal_scale == .05
    assert terrain.height_field_raw.shape == (2801, 4601)
    assert not np.any(terrain.height_field_raw[:500])
    assert not np.any(terrain.height_field_raw[-500:])
    assert not np.any(terrain.height_field_raw[:, :500])
    assert not np.any(terrain.height_field_raw[:, -500:])
    for row, col in ((0, 0), (3, 5), (9, 9), (9, 19)):
        x, y = 500 + row * 180, 500 + col * 180
        tile = terrain.height_field_raw[x:x + 181, y:y + 181]
        assert terrain.env_origins[row, col, 0] == (row + 0.5) * 9
        assert terrain.env_origins[row, col, 1] == (col + 0.5) * 9
        assert terrain.env_origins[row, col, 2] == tile[70:110:2, 70:110:2].max() * 0.005
    # Positive and inverted stairs must place robots on their center platforms.
    assert terrain.env_origins[9, 6, 2] < 0
    assert terrain.env_origins[9, 11, 2] > 0


def test_same_seeds_reproduce_both_difficulties_and_stochastic_geometry():
    np.random.seed(2026)
    first = WFTerrain(terrain_cfg(num_rows=2), 64, mesh_converter=None)
    np.random.seed(2026)
    second = WFTerrain(terrain_cfg(num_rows=2), 64, mesh_converter=None)
    np.testing.assert_array_equal(first.difficulties, second.difficulties)
    np.testing.assert_array_equal(first.height_field_raw, second.height_field_raw)
    np.random.seed(0)
    different_roughness = WFTerrain(terrain_cfg(num_rows=2), 64, mesh_converter=None)
    np.testing.assert_array_equal(first.difficulties, different_roughness.difficulties)
    assert not np.array_equal(first.height_field_raw, different_roughness.height_field_raw)


def test_default_difficulty_seed_uses_global_numpy_state():
    np.random.seed(2026)
    terrain = WFTerrain(terrain_cfg(num_rows=1, seed=None), 64, mesh_converter=None)
    expected = np.random.default_rng(2026).uniform(size=20)[None, :]
    np.testing.assert_array_equal(terrain.difficulties, expected)


def test_mesh_uses_native_tile_resolution_and_sensor_grid_samples_the_same_surface():
    calls = []

    def record_converter(heights, horizontal_scale, vertical_scale, slope_threshold):
        calls.append((heights, horizontal_scale, vertical_scale, slope_threshold))
        return np.zeros((3, 3), dtype=np.float32), np.array([[0, 1, 2]], dtype=np.uint32)

    np.random.seed(17)
    cfg = terrain_cfg(mesh_type="trimesh", num_rows=2)
    terrain = WFTerrain(cfg, 64, record_converter)
    assert terrain.cfg.horizontal_scale == .10
    assert len(calls) == 34  # 17 non-flat columns and 2 rows
    np.random.seed(17)
    fine_map = WFTerrain(terrain_cfg(mesh_type="heightfield", num_rows=2), 64, None)
    assert fine_map.cfg.horizontal_scale == .05
    for i, (heights, spacing, vs, threshold) in enumerate(calls):
        col, row = divmod(i, cfg.num_rows)
        is_stair = 6 <= col < 13
        assert spacing == (.05 if is_stair else .10)
        assert heights.shape == ((181, 181) if is_stair else (91, 91))
        assert vs == .005 and threshold == (10.0 if col < 6 else .25)
        x, y = terrain.border + row * 90, terrain.border + col * 90
        sensor = terrain.height_field_raw[x:x + 91, y:y + 91]
        native_stride = 2 if is_stair else 1
        np.testing.assert_array_equal(sensor, heights[::native_stride, ::native_stride])
        x, y = fine_map.border + row * 180, fine_map.border + col * 180
        collision = fine_map.height_field_raw[x:x + 181, y:y + 181]
        fine_stride = 1 if is_stair else 2
        np.testing.assert_array_equal(collision[::fine_stride, ::fine_stride], heights)
    assert terrain.vertices.dtype == np.float32
    assert terrain.triangles.dtype == np.uint32
    assert terrain.triangles.max() < len(terrain.vertices)
    assert terrain.vertices[:, 0].min() == 0
    assert terrain.vertices[:, 0].max() == 68
    assert terrain.vertices[:, 1].min() == 0
    assert terrain.vertices[:, 1].max() == 230
