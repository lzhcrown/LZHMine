"""Map-only representation shared by pretraining and the CTS policy."""

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .moe import L2Norm, MLP, SimNorm
from .teacher_map_attention import height_grid_xy, validate_height_grid
from rsl_rl.utils.cts_observations import observation_term_slice, validate_observation_config


HEIGHT_ROWS = 11
HEIGHT_COLS = 17
HEIGHT_CELLS = HEIGHT_ROWS * HEIGHT_COLS
TERRAIN_LATENT_DIM = 16


def _scan_config():
    return {
        "rows_y": HEIGHT_ROWS, "cols_x": HEIGHT_COLS, "spacing_m": 0.1,
        "height_grid_order": "xy", "height_offset": 0.5,
        "clip": [-1.0, 1.0], "scale": 2.5,
        "quantity": "base_z_minus_offset_minus_surface_z",
    }


class TerrainEncoder(nn.Module):
    """Encode a clipped, scaled CTS scan. True mask entries hide their heights."""

    def __init__(self):
        super().__init__()
        self.height_rows = HEIGHT_ROWS
        self.height_cols = HEIGHT_COLS
        self.height_cells = HEIGHT_CELLS
        self.features = nn.Sequential(
            nn.Conv2d(2, 8, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ELU(),
        )
        # Each of four learned queries pools the 3 x 5 spatial feature grid.
        self.position = nn.Parameter(torch.empty(1, 3 * 5, 16))
        nn.init.normal_(self.position, std=0.02)
        self.attention_queries = nn.Linear(16, 4, bias=False)
        self.to_latent = nn.Linear(4 * 16, TERRAIN_LATENT_DIM)

    def forward(self, scan: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if scan.ndim != 2 or scan.shape[1] != self.height_cells:
            raise ValueError("TerrainEncoder requires a [batch, 187] height scan")
        if mask is None:
            mask_channel = torch.zeros_like(scan)
        else:
            if mask.dtype != torch.bool or mask.shape != scan.shape:
                raise ValueError("Terrain masks must be boolean and match the scan shape")
            scan = scan.masked_fill(mask, 0.0)
            mask_channel = mask.to(scan.dtype)
        image = torch.stack((scan, mask_channel), dim=1).reshape(-1, 2, self.height_rows, self.height_cols)
        tokens = self.features(image).flatten(2).transpose(1, 2) + self.position
        weights = self.attention_queries(torch.tanh(tokens)).transpose(1, 2).softmax(dim=-1)
        pooled = torch.bmm(weights, tokens).flatten(1)
        return F.normalize(self.to_latent(pooled), dim=-1)


def terrain_encoder_checkpoint(encoder):
    """Package only the deployable stage-one encoder and its scan convention."""
    return {
        "architecture": "nezha_terrain_encoder", "format_version": 1,
        "scan_config": _scan_config(), "latent_dim": TERRAIN_LATENT_DIM,
        "encoder_state_dict": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
    }


class FrozenTerrainEncoder(TerrainEncoder):
    """Stage-one geometry stays fixed throughout locomotion learning."""

    def __init__(self):
        super().__init__()
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def load_pretrained(self, path):
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if (not isinstance(artifact, dict) or artifact.get("architecture") != "nezha_terrain_encoder"
                or artifact.get("format_version") != 1 or artifact.get("scan_config") != _scan_config()
                or artifact.get("latent_dim") != TERRAIN_LATENT_DIM):
            raise ValueError("Incompatible pretrained terrain encoder or height scan convention")
        state = artifact.get("encoder_state_dict")
        expected = self.state_dict()
        if (not isinstance(state, dict) or state.keys() != expected.keys()
                or any(not isinstance(value, torch.Tensor) or value.shape != expected[key].shape
                       or value.dtype != expected[key].dtype or not torch.isfinite(value).all()
                       for key, value in state.items())):
            raise ValueError("Invalid pretrained terrain encoder weights")
        self.load_state_dict(state, strict=True)


def pretrained_scan_slice(schema, privileged_dim):
    if schema is None:
        raise ValueError("The frozen terrain encoder requires the observation schema")
    validate_observation_config(schema)
    student_fields = {term["name"] for term in schema["proprioceptive_terms"]}
    if student_fields.intersection({"height_scan", "skill_id", "terrain_id"}):
        raise ValueError("G13 requires a blind student proprioceptive history without height_scan or skill_id")
    fields = {term["name"]: term for term in schema["privileged_terms"]}
    scan = fields.get("height_scan", {})
    expected = _scan_config()
    if (schema.get("height_grid_order") != expected["height_grid_order"]
            or schema.get("height_offset") != expected["height_offset"]
            or scan.get("dim") != HEIGHT_CELLS or scan.get("scale") != expected["scale"]
            or scan.get("clip") != expected["clip"] or scan.get("noise", 0.0) != 0):
        raise ValueError("The frozen terrain encoder requires the pretrained height_scan layout and scaling")
    if fields.get("skill_id", {}).get("dim") != 1:
        raise ValueError("The frozen terrain encoder requires a privileged scalar skill_id")
    if sum(term["dim"] for term in fields.values()) != privileged_dim:
        raise ValueError("The frozen terrain encoder schema disagrees with num_critic_obs")
    return observation_term_slice(schema["privileged_terms"], "height_scan")


class FrozenTerrainTeacherEncoder(nn.Sequential):
    """Fuse non-map privileged fields and a detached, pretrained geometry code."""

    def __init__(self, schema, privileged_dim, latent_dim, hidden_dims, activation, norm_type):
        scan = pretrained_scan_slice(schema, privileged_dim)
        super().__init__(
            MLP(privileged_dim - HEIGHT_CELLS + TERRAIN_LATENT_DIM, latent_dim, hidden_dims, activation),
            L2Norm() if norm_type == "l2norm" else SimNorm(),
        )
        self.height_start, self.height_end = scan.start, scan.stop
        self.register_buffer("xy", height_grid_xy(), persistent=False)
        self.terrain_encoder = FrozenTerrainEncoder()

    def validate_grid(self, height_points):
        validate_height_grid(height_points, self.xy)

    def load_pretrained(self, path):
        self.terrain_encoder.load_pretrained(path)

    def forward(self, privileged):
        with torch.no_grad():
            terrain = self.terrain_encoder(privileged[:, self.height_start:self.height_end])
        inputs = torch.cat((privileged[:, :self.height_start], privileged[:, self.height_end:], terrain), dim=1)
        return self[1](self[0](inputs))
