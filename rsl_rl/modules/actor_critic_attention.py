# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation

torch.autograd.set_detect_anomaly(True)

class AttentionEncoder(nn.Module):
    def __init__(
        self,
        num_obs: int,
        exteroception_offset: int,
        exteroception_dims: tuple[int, int],
        out_dim: int,
        num_exteroception_history: int = 2,
        hidden_dim: int = 64,
        activation: str = "elu",
        conv_params: dict = {"kernel_size": 3, "stride": 3},
    ):
        """Attention-based encoder for proprioception and exteroception data.

        The encoder can be used to model both the actor and critic networks in an actor-critic architecture.

        Args:
            num_obs (int): Dimension of the proprioception input.
            exteroception_offset (int): Starting index of the exteroception in the input vector. We assume that the input will be a concatenation of proprioception and exteroception data.
            exteroception_dims (tuple[int, int]): Dimensions of the exteroception input (dim1, dim2).
            out_dim (int): Dimension of the output. This is typically the number of actions for the actor or the value dimension for the critic.
            hidden_dim (int, optional): Dimension of the hidden layers. Defaults to 128.
            activation (str, optional): Activation function to use. Defaults to "elu".
            conv_params (dict, optional): Parameters for the convolutional layer. Defaults to {'kernel_size': 3, 'stride': 3}.

        Raises:
            AssertionError: If the exteroception dimensions are not divisible by the kernel size.
        """
        super().__init__()
        self.num_obs = num_obs
        
        if exteroception_offset < 0:
            exteroception_offset += num_obs  # Allow negative indexing
        assert 0 < exteroception_offset < num_obs, "exteroception_offset must be a valid index within the input vector."

        self.exteroception_offset = exteroception_offset
        self.exteroception_dims = exteroception_dims
        self.num_exteroception_history = num_exteroception_history

        self.out_dim = out_dim

        self.activation = resolve_nn_activation(activation)
        self.proprioception_encoder = nn.Linear(exteroception_offset, hidden_dim) # * 2

        assert num_exteroception_history > 0, "Exteroception history must be greater than 0."

        # Compute patch size for exteroception
        extero_conv = nn.Conv2d(
            num_exteroception_history,
            hidden_dim,
            groups=1,
            padding="valid",
            **conv_params,
        )

        # number of patches
        assert (
            exteroception_dims[-1] % conv_params["kernel_size"] == 0
        ), "Exteroception dims must be divisible by kernel size"
        assert (
            exteroception_dims[-2] % conv_params["kernel_size"] == 0
        ), "Exteroception dims must be divisible by kernel size"

        # Dummy input to compute the output dimension
        dummy_input = torch.zeros(
            7, num_exteroception_history, exteroception_dims[-2], exteroception_dims[-1]
        )
        dummy_output = extero_conv(dummy_input)
        num_patches = dummy_output.shape[2] * dummy_output.shape[3]

        self.exteroception_encoder = nn.Sequential(
            extero_conv,
            self.activation,
            nn.Flatten(-2),  # Flatten the last two dimensions to get patches
        )  # Output shape: (num_env, hidden_dim, num_patches)

        self.position_encoder = nn.Embedding(num_patches, hidden_dim) # *4

        self.att_Q = nn.Linear(hidden_dim, hidden_dim) # 2, 2
        self.att_K = nn.Linear(hidden_dim, hidden_dim) # 4, 2
        self.att_V = nn.Linear(hidden_dim, hidden_dim) #  # 4, 2

        self.ffn = nn.Linear(hidden_dim, hidden_dim) # 2, 1
        self.output_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        proprioception: torch.Tensor = input[:, :self.exteroception_offset]  # (num_envs, num_proprioception_obs)
        exteroception: torch.Tensor = input[:, self.exteroception_offset:]  # (num_envs, num_exteroception_obs)

        # Fold exteroception by the number of history steps
        exteroception = exteroception.view(
            exteroception.shape[0],
            self.num_exteroception_history,
            self.exteroception_dims[0],
            self.exteroception_dims[1],
        )  # (num_envs, num_exteroception_history, dim1, dim2)

        # Proprioception encoding
        proprio_encoded = self.activation(self.proprioception_encoder(proprioception))

        # Exteroception encoding
        extero_encoded = self.exteroception_encoder(exteroception) # (num_envs, hidden_dim, num_patches)
        extero_encoded = extero_encoded.permute(0, 2, 1)  # (num_envs, num_patches, hidden_dim)

        # Create position embeddings
        num_envs = proprioception.shape[0]
        num_patches = extero_encoded.shape[1]  # Number of patches from exteroception encoding
        assert num_patches > 0, "Exteroception encoding must produce patches."

        position_indices = (
            torch.arange(num_patches, device=proprio_encoded.device)
            .unsqueeze(0)
            .expand(num_envs, -1)
        )  # (num_envs, num_patches)
        
        extero_encoded = extero_encoded + self.position_encoder(position_indices)  # (num_envs, num_patches, hidden_dim)

        # Attention mechanism
        proprio_Q = self.att_Q(proprio_encoded).unsqueeze(1).unsqueeze(1) # (num_envs, num_Q_heads = 1, target_seq_len = 1, hidden_dim)
        extero_K = self.att_K(extero_encoded).unsqueeze(1) # (num_envs, num_KV_heads = 1, num_patches, hidden_dim)
        extero_V = self.att_V(extero_encoded).unsqueeze(1) # (num_envs, num_KV_heads = 1, num_patches, hidden_dim)

        # Compute attention scores
        out = torch.nn.functional.scaled_dot_product_attention(proprio_Q, extero_K, extero_V, dropout_p=0.0)  # (num_envs, num_Q_heads = 1, target_seq_len = 1, hidden_dim)
        out = out.squeeze()  # (num_envs, hidden_dim)

        # Combine proprioception and exteroception encodings
        out = out + proprio_encoded  # (num_envs, hidden_dim)
        out = nn.functional.normalize(out, dim=-1)  # Normalize the output across the last dimension
        out = self.activation(self.ffn(out))  # Apply activation function
        out = self.output_layer(out)  # (num_envs, out_dim)
        return out  # Final output shape: (num_envs, out_dim)
        


class ActorCriticAttention(nn.Module):
    is_recurrent = False
    def __init__(
        self,
        num_actor_obs: int,
        exteroception_offset: int,
        exteroception_dims: tuple[int, int],
        num_critic_obs: int,
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

        # Policy
        self.actor = AttentionEncoder(num_actor_obs, exteroception_offset, exteroception_dims, num_actions, num_exteroception_history=2, hidden_dim=128)

        # Value function
        self.critic = AttentionEncoder(num_critic_obs, exteroception_offset, exteroception_dims, 1, num_exteroception_history=2, hidden_dim=128)

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
