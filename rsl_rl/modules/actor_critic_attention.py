# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import warnings
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any

from rsl_rl.networks import AttentionEncoder, MLP, EmpiricalNormalization
from .actor_critic import ActorCritic


class ActorCriticAttention(ActorCritic):
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        grid_idx: torch.Tensor,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] = [256, 256, 256],
        critic_hidden_dims: list[int] = [512, 256, 128],
        exteroception_dims: tuple[int, int] = (25, 16),
        attention_hidden_dim: int = 64,
        attention_heads: int = 8,
        activation: str = "elu",
        init_noise_std: float = 0.1,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticAttention.__init__ got unexpected arguments, which will be ignored: "
                + str(kwargs.keys()),
            )
        super(ActorCritic, self).__init__()

        assert actor_obs_normalization == critic_obs_normalization, "Actor and critic obs normalization must be the same for attention model."

        self.exteroception_dims = exteroception_dims

        # get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs_1d = 0
        self.actor_obs_groups_1d = []
        self.actor_obs_groups_extero = []

        for obs_group in obs_groups["policy"]:
            if len(obs[obs_group].shape) > 2 or (len(obs[obs_group].shape) == 2 and obs[obs_group].shape[-1] == exteroception_dims[0] * exteroception_dims[1]):
                self.actor_obs_groups_extero.append(obs_group)
            elif len(obs[obs_group].shape) == 2:
                self.actor_obs_groups_1d.append(obs_group)
                num_actor_obs_1d += obs[obs_group].shape[-1]
            else:
                raise ValueError(f"Invalid observation shape for {obs_group}: {obs[obs_group].shape}")

        num_critic_obs_1d = 0
        self.critic_obs_groups_1d = []
        self.critic_obs_groups_extero = []

        for obs_group in obs_groups["critic"]:
            if len(obs[obs_group].shape) > 2 or (len(obs[obs_group].shape) == 2 and obs[obs_group].shape[-1] == exteroception_dims[0] * exteroception_dims[1]):
                self.critic_obs_groups_extero.append(obs_group)
            elif len(obs[obs_group].shape) == 2:
                self.critic_obs_groups_1d.append(obs_group)
                num_critic_obs_1d += obs[obs_group].shape[-1]
            else:
                raise ValueError(f"Invalid observation shape for {obs_group}: {obs[obs_group].shape}")

        # student encoder + MLP
        self.encoder = AttentionEncoder(
            num_proprio_obs=num_actor_obs_1d,
            exteroception_dims=exteroception_dims,
            hidden_dim=attention_hidden_dim,
            num_heads=attention_heads,
            grid_idx=grid_idx
        )
        mlp_input_dim = num_actor_obs_1d + attention_hidden_dim

        self.state_dependent_std = state_dependent_std

        if self.state_dependent_std:
            self.actor = MLP(mlp_input_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(mlp_input_dim, num_actions, actor_hidden_dims, activation)

        # student observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs_1d)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        print(f"Attention encoder: {self.encoder}")
        print(f"Actor MLP: {self.actor}")

        self.critic = MLP(mlp_input_dim, 1, critic_hidden_dims, activation)

        # critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = self.actor_obs_normalizer  # share the same normalizer, as we will use the same encoder
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if isinstance(init_noise_std, dict) or init_noise_std is None:
            init_noise_std = 1.0  # default
        else:
            init_noise_std = float(init_noise_std)

        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # action distribution
        self.distribution = None
        Normal.set_default_validate_args(False)

    def _update_distribution(self, mlp_obs: torch.Tensor, extero_obs: dict[str, torch.Tensor]) -> None:
        if self.actor_obs_groups_extero:
            extero_tensors = [extero_obs[group] for group in self.actor_obs_groups_extero]
            extero = torch.cat(extero_tensors, dim=-1)
            out_enc = self.encoder(mlp_obs, extero).squeeze(0)
        else:
            # Fallback if no exteroception
            out_enc = mlp_obs

        super()._update_distribution(out_enc)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        mlp_obs, extero_obs = self.get_actor_obs(obs)
        mlp_obs = self.actor_obs_normalizer(mlp_obs)
        self._update_distribution(mlp_obs, extero_obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        mlp_obs, extero_obs = self.get_actor_obs(obs)
        mlp_obs = self.actor_obs_normalizer(mlp_obs)

        if self.actor_obs_groups_extero:
            extero_tensors = [extero_obs[group] for group in self.actor_obs_groups_extero]
            extero = torch.cat(extero_tensors, dim=-1)
            out_enc = self.encoder(mlp_obs, extero).squeeze(0)
        else:
            out_enc = mlp_obs

        if self.state_dependent_std:
            return self.actor(out_enc)[..., 0, :]
        else:
            return self.actor(out_enc)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        mlp_obs, extero_obs = self.get_critic_obs(obs)
        mlp_obs = self.critic_obs_normalizer(mlp_obs)

        if self.critic_obs_groups_extero:
            extero_tensors = [extero_obs[group] for group in self.critic_obs_groups_extero]
            extero = torch.cat(extero_tensors, dim=-1)
            out_enc = self.encoder(mlp_obs, extero).squeeze(0)
        else:
            out_enc = mlp_obs

        return self.critic(out_enc)

    def get_actor_obs(self, obs: TensorDict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        obs_list_1d = [obs[obs_group] for obs_group in self.actor_obs_groups_1d]
        obs_dict_extero = {}
        for obs_group in self.actor_obs_groups_extero:
            obs_dict_extero[obs_group] = obs[obs_group]
        return torch.cat(obs_list_1d, dim=-1) if obs_list_1d else torch.empty(obs.shape[0], 0, device=obs.device), obs_dict_extero

    def get_critic_obs(self, obs: TensorDict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        obs_list_1d = [obs[obs_group] for obs_group in self.critic_obs_groups_1d]
        obs_dict_extero = {}
        for obs_group in self.critic_obs_groups_extero:
            obs_dict_extero[obs_group] = obs[obs_group]
        return torch.cat(obs_list_1d, dim=-1) if obs_list_1d else torch.empty(obs.shape[0], 0, device=obs.device), obs_dict_extero

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs, _ = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs, _ = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict, strict=True):
        if any("encoder_s" in key for key in state_dict.keys()):
            warnings.warn(
                "The state_dict contains keys for 'encoder_s', which indicates that the checkpoint was saved with an older version of the code. The keys will be renamed to 'encoder'."
            )
            state_dict = {k.replace("encoder_s.", "encoder."): v for k, v in state_dict.items()}

        if any("student" in key or "teacher" in key for key in state_dict.keys()):  # loading parameters from distillation training
            actor_state_dict = {}
            actor_obs_normalizer_state_dict = {}
            encoder_state_dict = {k: v for k, v in state_dict.items() if "encoder" in k}

            for key, value in state_dict.items():
                if "student" in key:
                    actor_state_dict[key.replace("student.", "actor.")] = value
                if "student_obs_normalizer" in key:
                    actor_obs_normalizer_state_dict[key.replace("student_obs_normalizer.", "actor_obs_normalizer.")] = value

            if len(actor_obs_normalizer_state_dict) > 0:
                assert self.actor_obs_normalization, \
                    "The checkpoint contains a student_obs_normalizer, but the current model does not use actor \
                    obs normalization. Please enable actor_obs_normalization in the config to load the normalizer parameters."
            else:
                assert not self.actor_obs_normalization, \
                    "The checkpoint does not contain a student_obs_normalizer, but the current model uses actor \
                    obs normalization. Please disable actor_obs_normalization in the config to load the model without normalizer parameters."

            self.actor.load_state_dict(actor_state_dict, strict=strict)
            self.actor_obs_normalizer.load_state_dict(actor_obs_normalizer_state_dict, strict=strict)
            self.encoder.load_state_dict(encoder_state_dict, strict=strict)

            warnings.warn(
                "Loading actor parameters from distillation training. The critic network will be randomly initialized (except for the shared encoder)."
            )
            return True
        else:
            warnings.warn(
                "Loading actor-critic parameters from RL training. If you want to load parameters from distillation training, please make sure you are choosing the correct checkpoint."
            )
            return super().load_state_dict(state_dict, strict=strict)
