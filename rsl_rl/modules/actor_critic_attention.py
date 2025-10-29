# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import warnings
from torch.distributions import Normal

from rsl_rl.networks import AttentionEncoder, MLP, EmpiricalNormalization


class ActorCriticAttention(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        grid_idx: torch.Tensor,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] = [256, 256, 256],
        critic_hidden_dims: list[int] = [512, 256, 128], # TODO: single source of truth
        exteroception_dims: tuple[int, int] = (25, 16), # TODO: single source of truth
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
        super().__init__()

        assert actor_obs_normalization == critic_obs_normalization, "Actor and critic obs normalization must be the same for attention model."

        exteroception_offset = -exteroception_dims[0] * exteroception_dims[1]

        # get the observation dimensions
        self.obs_groups = obs_groups
        num_student_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCriticAttention module only supports 1D observations."
            num_student_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCriticAttention module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        # student encoder + MLP
        self.encoder = AttentionEncoder(
            num_student_obs,
            exteroception_offset=exteroception_offset,
            exteroception_dims=exteroception_dims,
            hidden_dim=attention_hidden_dim,
            num_heads=attention_heads,
            grid_idx=grid_idx
        )
        num_query = exteroception_offset if exteroception_offset > 0 else num_student_obs + exteroception_offset
        mlp_input_dim = num_query + attention_hidden_dim

        self.state_dependent_std = state_dependent_std

        if self.state_dependent_std:
            self.actor = MLP(mlp_input_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(mlp_input_dim, num_actions, actor_hidden_dims, activation)

        # student observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_student_obs)
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

        # action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def update_distribution(self, obs):
        if self.state_dependent_std:
            # compute mean and standard deviation
            mean_and_std = self.actor(obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            # compute mean
            mean = self.actor(obs)
            # compute standard deviation
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        out_enc = self.encoder(obs).squeeze(0)
        self.update_distribution(out_enc)
        return self.distribution.sample()

    def act_inference(self, obs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        out_enc = self.encoder(obs).squeeze(0)
        return self.actor(out_enc)

    def evaluate(self, obs, **kwargs):
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)

        out_enc = self.encoder(obs).squeeze(0)

        return self.critic(out_enc)

    def get_actor_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_hidden_states(self):
        raise NotImplementedError

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        if any("encoder_s" in key for key in state_dict.keys()):
            # rename keys to match encoder
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

        else:
            warnings.warn(
                "Loading actor-critic parameters from RL training. If you want to load parameters from distillation training, please make sure you are choosing the correct checkpoint."
            )
            super().load_state_dict(state_dict, strict=strict)

        return True
