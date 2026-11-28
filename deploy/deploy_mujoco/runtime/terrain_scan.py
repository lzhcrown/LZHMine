"""Sample the current MuJoCo collision terrain in the CTS height-scan convention."""

import numpy as np

from rsl_rl.modules.teacher_map_attention import height_grid_xy


class TerrainHeightScanner:
    def __init__(self, mujoco, model, height_offset):
        self.mujoco = mujoco
        self.model = model
        self.offset = height_offset
        self.grid = height_grid_xy()[0].numpy()
        world = (model.geom_bodyid == 0) & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
        ground_ids = np.flatnonzero(world)
        if not ground_ids.size:
            raise ValueError("A terrain student requires collision ground in the MuJoCo scene")
        self.groups = np.zeros(6, dtype=np.uint8)
        self.groups[model.geom_group[world]] = 1
        if self.groups[model.geom_group[~world]].any():
            raise ValueError("Ground collision geoms must have ray groups separate from robot and visual geoms")
        # Cast from above every static obstacle, including elevated platforms.
        self.ray_z = np.max(model.geom_pos[world, 2] + model.geom_rbound[world]) + 1.0
        self.lower = np.zeros(2)
        self.spacing = np.full(2, 0.1)
        self.samples = None
        self.sole_hfield = False
        fields = ground_ids[model.geom_type[world] == mujoco.mjtGeom.mjGEOM_HFIELD]
        if fields.size:
            if len(fields) != 1 or not np.allclose(model.geom_quat[fields[0]], [1, 0, 0, 0]):
                raise ValueError("CTS terrain sampling requires a single axis-aligned heightfield")
            geom = fields[0]
            field = model.geom_dataid[geom]
            rows, cols = model.hfield_nrow[field], model.hfield_ncol[field]
            start = model.hfield_adr[field]
            self.samples = model.hfield_data[start:start + rows * cols].reshape(rows, cols)
            half_size = model.hfield_size[field, :2]
            self.lower = model.geom_pos[geom, :2] - half_size
            self.spacing = 2 * half_size / np.array([cols - 1, rows - 1])
            self.base_z = model.geom_pos[geom, 2]
            self.scale_z = model.hfield_size[field, 2]
            self.ray_z = max(self.ray_z, self.base_z + self.scale_z + 1.0)
            self.sole_hfield = len(ground_ids) == 1
        self.plane_z = None
        if (len(ground_ids) == 1 and model.geom_type[ground_ids[0]] == mujoco.mjtGeom.mjGEOM_PLANE
                and np.allclose(model.geom_quat[ground_ids[0]], [1, 0, 0, 0])):
            self.plane_z = model.geom_pos[ground_ids[0], 2]

    def scan(self, data, position, quat_wxyz):
        """Return x-fastest raw base-relative heights using Isaac's yaw-only quaternion and vertex minimum."""
        if self.plane_z is not None:
            return np.full(len(self.grid), position[2] - self.offset - self.plane_z, dtype=np.float32)
        w, z = np.asarray(quat_wxyz, dtype=np.float32)[[0, 3]]
        norm = max(np.sqrt(w * w + z * z), 1e-9)
        w, z = w / norm, z / norm
        # Follow Isaac's float32 quat_apply operation order even at cell boundaries.
        # t = 2 * cross(q.xyz, point), rotated = point + w * t + cross(q.xyz, t).
        t = np.stack((-z * self.grid[:, 1], z * self.grid[:, 0]), axis=1) * 2
        xy = self.grid + w * t + np.stack((-z * t[:, 1], z * t[:, 0]), axis=1)
        xy += np.asarray(position[:2], dtype=np.float32)
        index = np.floor((xy - self.lower.astype(np.float32)) / self.spacing.astype(np.float32)).astype(int)
        if self.samples is not None:
            index = np.clip(index, 0, np.array(self.samples.shape[::-1]) - 2)
        # Each Isaac scan point uses the lower of (x,y), (x+1,y) and (x,y+1).
        vertices = index[:, None] + np.array([[0, 0], [1, 0], [0, 1]])
        if self.sole_hfield:
            ground = self.base_z + self.scale_z * self.samples[vertices[..., 1], vertices[..., 0]]
        else:
            # The seven-lane course also models stairs and ramps as solid box/mesh geoms.
            unique, inverse = np.unique(vertices.reshape(-1, 2), axis=0, return_inverse=True)
            heights = np.empty(len(unique))
            direction = np.array([0., 0., -1.])
            hit = np.empty(1, dtype=np.int32)
            for i, (x, y) in enumerate(unique * self.spacing + self.lower):
                distance = self.mujoco.mj_ray(self.model, data, np.array([x, y, self.ray_z]),
                                              direction, self.groups, True, -1, hit)
                if distance < 0:
                    if self.samples is None:
                        raise ValueError("Height scan extends beyond the MuJoCo collision ground")
                    col, row = unique[i]
                    heights[i] = self.base_z + self.scale_z * self.samples[row, col]
                else:
                    heights[i] = self.ray_z - distance
            ground = heights[inverse].reshape(len(self.grid), 3)
        return (position[2] - self.offset - ground.min(axis=1)).astype(np.float32)
