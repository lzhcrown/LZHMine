# Adapted from wheel_gym_CQ/wheel_legged_gym/evaluation/presets.py
"""Stable identifiers and dimensions for evaluation, separate from training."""

PATCH_TERRAIN_TYPES = (
    "flat",
    "stairs_concave_35cm",
    "stairs_concave_25cm",
    "stairs_concave_30cm",
    "stairs_convex_35cm",
    "stairs_convex_25cm",
    "stairs_convex_30cm",
    "platform_45cm",
    "rough",
    "slope",
)
PATCH_STAIR_TREAD_M = 0.3
PATCH_STEPS_PER_SIDE = 8
PATCH_STAIR_PLATFORM_M = 3.0
PATCH_STAIR_HEIGHTS_M = (0.35, 0.25, 0.30)
PATCH_HIGH_PLATFORM_SIZE_M = 5.0
PATCH_HIGH_PLATFORM_HEIGHT_M = 0.45
PATCH_SLOPE_ANGLE_DEG = 20.0
PATCH_SLOPE_LENGTH_M = 6.5
PATCH_SLOPE_WIDTH_M = 5.6
# Preserve all eight risers and the central platform when changing tread width.
PATCH_SIZE_M = 2 * PATCH_STEPS_PER_SIDE * PATCH_STAIR_TREAD_M + PATCH_STAIR_PLATFORM_M
PATCH_PLATFORM_LENGTH_M = 1.5
PATCH_CORRIDOR_WIDTH_M = PATCH_SIZE_M + 2.0
