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

from rsl_rl.networks import MLP, EmpiricalNormalization, AttentionEncoder


class StudentTeacherAttention(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        grid_idx: torch.Tensor,
        student_obs_normalization: bool = False,
        teacher_obs_normalization: bool = False,
        student_hidden_dims: list[int] = [256, 256, 256],
        teacher_hidden_dims: list[int] = [512, 256, 128],
        exteroception_dims: tuple[int, int] = (25, 16),
        attention_hidden_dim: int = 64,
        attention_heads: int = 8,
        activation: str = "elu",
        init_noise_std: float = 0.1,
        noise_std_type: str = "scalar",
        **kwargs: dict[str, Any],
    ):
        if kwargs:
            print(
                "StudentTeacherAttention.__init__ got unexpected arguments, which will be ignored: "
                + str(kwargs.keys()),
            )
        super().__init__()

        self.loaded_teacher = False  # indicates if teacher has been loaded
        self.exteroception_dims = exteroception_dims

        # get the observation dimensions
        self.obs_groups = obs_groups
        num_student_obs_1d = 0
        self.student_obs_groups_1d = []
        self.student_obs_groups_extero = []

        for obs_group in obs_groups["policy"]:
            if len(obs[obs_group].shape) > 2 or (len(obs[obs_group].shape) == 2 and obs[obs_group].shape[-1] == exteroception_dims[0] * exteroception_dims[1]):
                self.student_obs_groups_extero.append(obs_group)
            elif len(obs[obs_group].shape) == 2:
                self.student_obs_groups_1d.append(obs_group)
                num_student_obs_1d += obs[obs_group].shape[-1]
            else:
                raise ValueError(f"Invalid observation shape for {obs_group}: {obs[obs_group].shape}")

        num_teacher_obs = 0
        for obs_group in obs_groups["teacher"]:
            if len(obs[obs_group].shape) == 2:
                num_teacher_obs += obs[obs_group].shape[-1]
            else:
                raise ValueError(f"Invalid observation shape for {obs_group}: {obs[obs_group].shape}")

        assert self.student_obs_groups_extero, "No exteroception observations found in student policy. Ensure they are configured in a separate group or match exteroception_dims."

        # student encoder + MLP
        self.encoder = AttentionEncoder(
            num_proprio_obs=num_student_obs_1d,
            exteroception_dims=exteroception_dims,
            hidden_dim=attention_hidden_dim,
            num_heads=attention_heads,
            grid_idx=grid_idx
        )
        mlp_input_dim_s = num_student_obs_1d + 2 * attention_hidden_dim
        self.student = MLP(mlp_input_dim_s, num_actions, student_hidden_dims, activation)

        # student observation normalization
        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization:
            self.student_obs_normalizer = EmpiricalNormalization(num_student_obs_1d)
        else:
            self.student_obs_normalizer = torch.nn.Identity()

        print(f"Student encoder: {self.encoder}")
        print(f"Student MLP: {self.student}")

        self.teacher = MLP(num_teacher_obs, num_actions, teacher_hidden_dims, activation)

        # teacher observation normalization
        self.teacher_obs_normalization = teacher_obs_normalization
        if teacher_obs_normalization:
            self.teacher_obs_normalizer = EmpiricalNormalization(num_teacher_obs)
        else:
            self.teacher_obs_normalizer = torch.nn.Identity()

        print(f"Teacher MLP: {self.teacher}")

        # action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones=None, hidden_states=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        # compute mean
        mean = self.student(obs)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]):
        mlp_obs, extero_obs = self.get_student_obs(obs)
        mlp_obs = self.student_obs_normalizer(mlp_obs)

        extero_tensors = [extero_obs[group] for group in self.student_obs_groups_extero]
        extero = torch.cat(extero_tensors, dim=-1)

        out_enc = self.encoder(mlp_obs, extero).squeeze(0)
        self.update_distribution(out_enc)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict):
        mlp_obs, extero_obs = self.get_student_obs(obs)
        mlp_obs = self.student_obs_normalizer(mlp_obs)

        extero_tensors = [extero_obs[group] for group in self.student_obs_groups_extero]
        extero = torch.cat(extero_tensors, dim=-1)

        out_enc = self.encoder(mlp_obs, extero).squeeze(0)
        return self.student(out_enc)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]):
        obs_teacher = self.get_teacher_obs(obs)
        obs_teacher = self.teacher_obs_normalizer(obs_teacher)
        with torch.no_grad():
            return self.teacher(obs_teacher)

    def get_student_obs(self, obs: TensorDict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        obs_list_1d = [obs[obs_group] for obs_group in self.student_obs_groups_1d]
        obs_dict_extero = {}
        for obs_group in self.student_obs_groups_extero:
            obs_dict_extero[obs_group] = obs[obs_group]
        return torch.cat(obs_list_1d, dim=-1) if obs_list_1d else torch.empty(obs.shape[0], 0, device=obs.device), obs_dict_extero

    def get_teacher_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = []
        for obs_group in self.obs_groups["teacher"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_hidden_states(self):
        pass

    def detach_hidden_states(self, dones=None):
        pass

    def train(self, mode=True):
        super().train(mode)
        # make sure teacher is in eval mode
        self.teacher.eval()
        self.teacher_obs_normalizer.eval()

    def update_normalization(self, obs: TensorDict):
        if self.student_obs_normalization:
            student_obs_1d, _ = self.get_student_obs(obs)
            self.student_obs_normalizer.update(student_obs_1d)

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the student and teacher networks.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters.
        """

        # check if state_dict contains teacher and student or just teacher parameters
        if any("actor" in key for key in state_dict.keys()):  # loading parameters from rl training
            # rename keys to match teacher and remove critic parameters
            teacher_state_dict = {}
            teacher_obs_normalizer_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    teacher_state_dict[key.replace("actor.", "")] = value
                if "actor_obs_normalizer." in key:
                    teacher_obs_normalizer_state_dict[key.replace("actor_obs_normalizer.", "")] = value
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            if len(teacher_obs_normalizer_state_dict) > 0:
                self.teacher_obs_normalizer.load_state_dict(teacher_obs_normalizer_state_dict, strict=strict)

            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return False  # training does not resume
        elif any("student" in key for key in state_dict.keys()):  # loading parameters from distillation training
            super().load_state_dict(state_dict, strict=strict)
            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return True  # training resumes
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")
