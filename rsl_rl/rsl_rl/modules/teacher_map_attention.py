"""Spatial encoding of the Nezha teacher's privileged 17 by 11 height scan."""

import torch
from torch import nn

from rsl_rl.utils.cts_observations import validate_observation_config


def height_grid_xy():
    """Match _init_height_points: x varies within each row of constant y."""
    x = (torch.arange(17, dtype=torch.float32) - 8) * 0.1
    y = (torch.arange(11, dtype=torch.float32) - 5) * 0.1
    return torch.stack((x.repeat(11), y.repeat_interleave(17)), dim=-1).unsqueeze(0)


def validate_height_grid(height_points, expected_xy):
    """Check the simulator's actual sampling coordinates before any rollout."""
    points_xy = height_points[:1, :, :2]
    if points_xy.shape != expected_xy.shape or not torch.allclose(
            points_xy.to(expected_xy), expected_xy, atol=1e-6):
        raise ValueError("Teacher terrain encoding requires Nezha's x-fastest 17-by-11, 0.1 m grid")


class TeacherMapAttention(nn.Module):
    """Read local terrain with a query made from current proprioception and task."""

    def __init__(self, schema, privileged_dim, latent_dim):
        super().__init__()
        if schema is None:
            raise ValueError("Teacher map attention requires the observation schema")
        validate_observation_config(schema)
        if schema.get("height_grid_order") != "xy":
            raise ValueError("Teacher map attention requires an x-fastest ('xy') grid")

        fields = {}
        offset = 0
        for term in schema["privileged_terms"]:
            fields[term["name"]] = range(offset, offset + term["dim"])
            offset += term["dim"]
        if offset != privileged_dim or len(fields.get("height_scan", ())) != 17 * 11:
            raise ValueError("Teacher map attention requires a named 187-cell privileged height_scan")
        self.height_start = fields["height_scan"].start

        query_names = [term["name"] for term in schema["proprioceptive_terms"]]
        if "skill_id" in fields and "skill_id" not in query_names:
            query_names.append("skill_id")
        if "height_scan" in query_names or any(name not in fields for name in query_names):
            raise ValueError("Teacher map queries require non-map proprioceptive fields in privileged_terms")
        query_indices = [index for name in query_names for index in fields[name]]
        self.register_buffer("query_indices", torch.tensor(query_indices, dtype=torch.long), persistent=False)

        self.register_buffer("xy", height_grid_xy(), persistent=False)

        # Use a compact single-query cross-attention over all 187 map cells.
        # The scan uses the same clipped/scaled height values as the baseline teacher.
        embed_dim = 16
        self.height_features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ELU(),
            nn.Conv2d(8, embed_dim - 3, kernel_size=3, padding=1), nn.ELU(),
        )
        self.query = nn.Linear(len(query_indices), embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.to_latent = nn.Linear(embed_dim, latent_dim, bias=False)
        nn.init.zeros_(self.to_latent.weight)

    def validate_grid(self, height_points):
        validate_height_grid(height_points, self.xy)

    def forward(self, privileged):
        height = privileged[:, self.height_start:self.height_start + 187]
        local = self.height_features(height.reshape(-1, 1, 11, 17)).flatten(2).transpose(1, 2)
        tokens = torch.cat((local, self.xy.expand(height.size(0), -1, -1), height.unsqueeze(-1)), dim=-1)
        query = self.query(privileged.index_select(1, self.query_indices)).unsqueeze(1)
        pooled, _ = self.attention(query, tokens, tokens, need_weights=False)
        return self.to_latent(pooled.squeeze(1))


class MapAttentionTeacherEncoder(nn.Sequential):
    """Keep the pretrained MLP and norm at their original checkpoint indices."""

    def __init__(self, baseline, attention):
        super().__init__(baseline[0], baseline[1])
        self.map_attention = attention

    def forward(self, privileged):
        return self[1](self[0](privileged) + self.map_attention(privileged))
