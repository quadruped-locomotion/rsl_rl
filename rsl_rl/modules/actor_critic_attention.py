# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation

# torch.autograd.set_detect_anomaly(True)

class AttentionEncoder(nn.Module):
    def __init__(
        self,
        num_obs: int,
        exteroception_offset: int,
        exteroception_dims: tuple[int, int],
        grid_idx: torch.Tensor,
        hidden_dim: int = 64,
        activation: str = "elu",
        conv_params: dict = {"kernel_size": 5, "stride": 1, "padding": "same"}  # Default parameters for convolution,
    ):
        """Attention-based encoder for proprioception and exteroception data.

        The encoder can be used to model both the actor and critic networks in an actor-critic architecture.

        Args:
            num_obs (int): Dimension of the proprioception input.
            exteroception_offset (int): Starting index of the exteroception in the input vector. We assume that the input will be a concatenation of proprioception and exteroception data.
            exteroception_dims (tuple[int, int]): Dimensions of the exteroception input (dim1, dim2).
            hidden_dim (int, optional): Dimension of the hidden layers. Defaults to 128.
            activation (str, optional): Activation function to use. Defaults to "elu".
            conv_params (dict, optional): Parameters for the convolutional layer. Defaults to {'kernel_size': 5, 'stride': 1}.

        Raises:
            AssertionError: If the exteroception dimensions are not divisible by the kernel size.
        """
        super().__init__()
        self.num_obs = num_obs

        self.exteroception_offset = exteroception_offset
        self.exteroception_dims = exteroception_dims
        self.grid_idx = grid_idx  # The grid indices of the ray pattern

        assert grid_idx.shape[0] == exteroception_dims[0] * exteroception_dims[1], \
            f"Grid indices shape {grid_idx.shape} does not match exteroception dimensions {exteroception_dims}."
        
        assert conv_params["padding"] == "same", \
            "Padding must be set to 'same' to ensure the output dimensions match the input dimensions \
            for the convolutional layers."

        self.activation = resolve_nn_activation(activation)
        self.proprioception_encoder = nn.Linear(exteroception_offset, hidden_dim)

        # Convolutional encoder for exteroception
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, **conv_params),
            self.activation,
            nn.Conv2d(16, hidden_dim - 2, **conv_params),
            self.activation,
            nn.Flatten(-2),  # Flatten the last two dimensions to get patches
        )

        self.num_patches = exteroception_dims[0] * exteroception_dims[1]

        self.attention = nn.MultiheadAttention(
            embed_dim = hidden_dim,
            num_heads = 16,
            batch_first = True,
        )

        self.att_scores: torch.Tensor | None = None  # Placeholder for attention scores



    def forward(self, input: torch.Tensor, need_weights: bool = False) -> torch.Tensor:
        proprioception: torch.Tensor = input[:, :self.exteroception_offset]  # (num_envs, num_proprioception_obs)
        exteroception: torch.Tensor = input[:, self.exteroception_offset:]  # (num_envs, num_exteroception_obs)
        num_envs = input.shape[0]

        # Fold exteroception by the number of history steps
        exteroception = exteroception.view(
            exteroception.shape[0],
            1, # Height measurement channel
            self.exteroception_dims[0], # Width
            self.exteroception_dims[1], # Length
        )  # (num_envs, num_exteroception_history, dim1, dim2)

        # Proprioception encoding
        proprio_encoded = self.activation(self.proprioception_encoder(proprioception))

        # Exteroception encoding
        extero_encoded = self.conv(exteroception) # (num_envs, hidden_dim, num_patches)
        extero_encoded = extero_encoded.permute(0, 2, 1)  # (num_envs, num_patches, hidden_dim)
        extero_encoded = torch.cat([self.grid_idx.expand(num_envs, -1, -1), extero_encoded], -1) # (num_envs, num_patches, hidden_dim + 2), add grid indices to the exteroception encoding

        # Unsqueeze to add a target sequence length dimension for attention
        proprio_encoded = proprio_encoded.unsqueeze(1)  # (num_envs, 1, hidden_dim)

        # Compute attention
        att_output, self.att_scores = self.attention(
            query = proprio_encoded,  # Query: (num_envs, 1, hidden_dim)
            key   = extero_encoded,   # Key (num_envs, num_patches, hidden_dim)
            value = extero_encoded,   # Value (num_envs, num_patches, hidden_dim)
            need_weights=need_weights
        ) # Output shape: (num_envs, 1, hidden_dim), (att_scores shape: (num_envs, 1, num_patches)

        out = torch.cat([att_output.squeeze(), proprioception], 1)
        # Final output shape: (num_envs, hidden_dim + num_proprioception_obs)

        return out



class ActorCriticAttention(nn.Module):
    is_recurrent = False
    def __init__(
        self,
        num_actor_obs: int,
        exteroception_offset: int,
        exteroception_dims: tuple[int, int],
        grid_idx: torch.Tensor,
        num_actions: int,
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticAttention.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        if exteroception_offset < 0:
            exteroception_offset += num_actor_obs  # Allow negative indexing
        assert 0 < exteroception_offset < num_actor_obs, "exteroception_offset must be a valid index within the input vector."

        self.encoder = AttentionEncoder(num_actor_obs, exteroception_offset, exteroception_dims, grid_idx, hidden_dim=64)

        # The encoder outputs a tensor of shape (num_envs, 64 + exteroception_offset)
        self.actor = nn.Sequential(
            self.encoder,
            activation,
            nn.Linear(64 + exteroception_offset, 256),
            activation,
            nn.Linear(256, 256),
            activation,
            nn.Linear(256, num_actions),
        )

        # The critic also uses the same encoder, but outputs a single value
        # The input to the critic is the same as the actor, so we can reuse the encoder
        self.critic = nn.Sequential(
            self.encoder,
            activation,
            nn.Linear(64 + exteroception_offset, 256),
            activation,
            nn.Linear(256, 256),
            activation,
            nn.Linear(256, 1),
        )

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)


    def reset(self, dones=None):
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

    def update_distribution(self, observations):
        # compute mean
        mean = self.actor(observations)
        self.att_scores = self.actor[0].att_scores  # Store attention scores for potential use

        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
            )
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

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

        super().load_state_dict(state_dict, strict=strict)
        return True
