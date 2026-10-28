"""Stateless TorchScript export of the WF-CTS student and shared actor."""

import copy
import json
import os

import torch
from torch import nn


class CTSStudentExporter(nn.Module):
    """Input is already preprocessed, term-major history; no internal cache."""

    __constants__ = ["actor_is_moe"]

    def __init__(self, actor_critic, observation_config=None, control_config=None):
        super().__init__()
        if observation_config is None:
            observation_config = getattr(actor_critic, "observation_config", None)
        if observation_config is None:
            raise ValueError("CTS export requires the training observation_config")
        dimensions = [term["dim"] for term in observation_config["proprioceptive_terms"]]
        if dimensions != actor_critic.observation_dims:
            raise ValueError("Export observation terms differ from the policy")
        if observation_config["history_length"] != actor_critic.history_length:
            raise ValueError("Export history length differs from the policy")
        if observation_config["history_order"] != "oldest_first" or observation_config["history_layout"] != "term_major":
            raise ValueError("CTS export requires oldest-first, term-major history")
        self.student_encoder = copy.deepcopy(actor_critic.student_moe_encoder).cpu()
        self.actor = copy.deepcopy(actor_critic.actor).cpu()
        self.register_buffer("current_obs_indices", actor_critic.current_obs_indices.detach().clone().cpu())
        self.num_actor_obs = int(actor_critic.num_actor_obs)
        self.num_actions = int(actor_critic.num_actions)
        self.num_experts = int(actor_critic.expert_num)
        self.actor_type = actor_critic.actor_type
        self.actor_is_moe = self.actor_type == "moe"
        self.actor_expert_num = int(actor_critic.actor_expert_num)
        self.history_length = int(actor_critic.history_length)
        self.observation_config_json = json.dumps(observation_config, sort_keys=True)
        self.control_config_json = json.dumps(control_config or {}, sort_keys=True)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.dim() != 2 or history.shape[1] != self.num_actor_obs:
            raise ValueError("CTS student expects a batch of flattened observation histories")
        latent, _ = self.student_encoder(history)
        current = history.index_select(-1, self.current_obs_indices)
        return self.actor(torch.cat((latent, current), dim=-1))

    @torch.jit.export
    def get_expert_probabilities(self, history: torch.Tensor) -> torch.Tensor:
        # Keep the established deployment API referring to student experts.
        return self.student_encoder.moe.gating_network(history)

    @torch.jit.export
    def get_actor_expert_probabilities(self, history: torch.Tensor) -> torch.Tensor:
        if self.actor_is_moe:
            latent, _ = self.student_encoder(history)
            current = history.index_select(-1, self.current_obs_indices)
            return self.actor.get_expert_probabilities(torch.cat((latent, current), dim=-1))
        return history.new_empty((history.shape[0], 0))

    @torch.jit.export
    def get_actor_type(self) -> str:
        return self.actor_type

    @torch.jit.export
    def get_num_actor_experts(self) -> int:
        return self.actor_expert_num

    @torch.jit.export
    def get_architecture(self) -> str:
        return "nezha_moe_cts"

    @torch.jit.export
    def get_num_actor_obs(self) -> int:
        return self.num_actor_obs

    @torch.jit.export
    def get_num_actions(self) -> int:
        return self.num_actions

    @torch.jit.export
    def get_num_experts(self) -> int:
        return self.num_experts

    @torch.jit.export
    def get_history_length(self) -> int:
        return self.history_length

    @torch.jit.export
    def get_observation_config_json(self) -> str:
        return self.observation_config_json

    @torch.jit.export
    def get_control_config_json(self) -> str:
        return self.control_config_json

    def export(self, path):
        path = os.path.abspath(os.fspath(path))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.eval()
        scripted = torch.jit.script(self)
        scripted.save(path)
        metadata = {
            "architecture": "nezha_moe_cts",
            "format_version": 1,
            "num_actor_obs": self.num_actor_obs,
            "num_actions": self.num_actions,
            "num_experts": self.num_experts,
            "actor_type": self.actor_type,
            "actor_expert_num": self.actor_expert_num,
            "observation_config": json.loads(self.observation_config_json),
            "control_config": json.loads(self.control_config_json),
        }
        with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        return path
