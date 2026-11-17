"""Height-grid helpers adapted from wheel_gym_CQ evaluation/terrain.py."""
import math

import numpy as np


def _number(cfg, name, default=None, minimum=0., strict=False):
    value = float(getattr(cfg, name, default))
    if not math.isfinite(value) or value < minimum or (strict and value == minimum):
        comparison = ">" if strict else ">="
        raise ValueError("%s must be finite and %s %s" % (name, comparison, minimum))
    return value


def _grid_cells(distance, scale, name):
    cells = int(round(distance / scale))
    if not math.isclose(cells * scale, distance, rel_tol=0., abs_tol=1e-7):
        raise ValueError("%s must be an integer multiple of horizontal_scale" % name)
    return cells


class EvaluationTerrain:
    def ground_heights(self, xy):
        """Sample the quantized height grid, exactly flat on both spawn platforms.

        Bilinear values away from the platforms describe the heightfield; they
        are not a raycast against vertically corrected triangle-mesh stair faces.
        """
        points = np.asarray(xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("xy must be a finite (N, 2) array")
        grid = (points + self.cfg.border_size) / self.horizontal_scale
        maximum = np.asarray(self.height_field_raw.shape) - 1
        if np.any(grid < -1e-7) or np.any(grid > maximum + 1e-7):
            raise ValueError("Height query lies outside the evaluation terrain mesh")
        grid = np.clip(grid, 0, maximum)
        lower = np.floor(grid).astype(np.int64)
        lower = np.minimum(lower, maximum - 1)
        fraction = grid - lower
        ix, iy = lower[:, 0], lower[:, 1]
        dx, dy = fraction[:, 0], fraction[:, 1]
        samples = self.height_field_raw
        return ((1 - dx) * (1 - dy) * samples[ix, iy] + dx * (1 - dy) * samples[ix + 1, iy]
                + (1 - dx) * dy * samples[ix, iy + 1] + dx * dy * samples[ix + 1, iy + 1]) * self.vertical_scale

