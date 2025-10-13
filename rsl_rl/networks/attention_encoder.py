from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation


class AttentionEncoder(nn.Module):
    def __init__(
        self,
        num_obs: int,
        exteroception_offset: int,
        exteroception_dims: tuple[int, int],
        grid_idx: torch.Tensor,
        hidden_dim: int,
        num_heads: int,
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

        self.exteroception_offset: int = exteroception_offset
        self.exteroception_dims = exteroception_dims
        self.grid_idx = grid_idx  # The grid indices of the ray pattern

        assert grid_idx.shape[0] == exteroception_dims[0] * exteroception_dims[1], \
            f"Grid indices shape {grid_idx.shape} does not match exteroception dimensions {exteroception_dims}."
        
        assert conv_params["padding"] == "same", \
            "Padding must be set to 'same' to ensure the output dimensions match the input dimensions \
            for the convolutional layers."

        self.activation = resolve_nn_activation(activation)

        assert abs(exteroception_offset) < num_obs, \
            f"Exteroception offset {exteroception_offset} must be less than the total observation dimension {num_obs}."

        num_proprio_obs = num_obs + exteroception_offset if exteroception_offset < 0 else exteroception_offset

        self.proprioception_encoder = nn.Linear(num_proprio_obs, hidden_dim)

        self.position_encoder = nn.Embedding(
            num_embeddings = exteroception_dims[0] * exteroception_dims[1],
            embedding_dim = hidden_dim)

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
            num_heads = num_heads,
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
        extero_encoded = extero_encoded.permute(0, 2, 1)  # (num_envs, num_patches, hidden_dim - 2)
        extero_encoded = torch.cat([self.grid_idx.expand(num_envs, -1, -1), extero_encoded], -1) # (num_envs, num_patches, hidden_dim), add grid indices to the exteroception encoding

        # # Add positional encoding to exteroception
        # position_idx = torch.arange(
        #     self.num_patches, device=extero_encoded.device, dtype=torch.long
        # )  # (num_patches,)
        # position_encoded = self.position_encoder(position_idx)  # (num_patches, hidden_dim)
        # position_encoded = position_encoded.unsqueeze(0).expand(num_envs, -1, -1)  # (num_envs, num_patches, hidden_dim)
        # # Add positional
        # extero_encoded = extero_encoded + position_encoded  # (num_envs, num_patches, hidden_dim)

        # Unsqueeze to add a target sequence length dimension for attention
        proprio_encoded = proprio_encoded.unsqueeze(1)  # (num_envs, 1, hidden_dim)

        # Compute attention
        att_output, self.att_scores = self.attention(
            query = proprio_encoded,  # Query: (num_envs, 1, hidden_dim)
            key   = extero_encoded,   # Key (num_envs, num_patches, hidden_dim)
            value = extero_encoded,   # Value (num_envs, num_patches, hidden_dim)
            need_weights=need_weights
        ) # Output shape: (num_envs, 1, hidden_dim), (att_scores shape: (num_envs, 1, num_patches)

        att_output = self.activation(att_output) # Apply activation to the attention output

        out = torch.cat([att_output.squeeze((1,2)), proprioception], 1)
        # Final output shape: (num_envs, hidden_dim + num_proprioception_obs)

        return out
