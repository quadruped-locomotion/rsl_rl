from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation


class AttentionEncoder(nn.Module):
    def __init__(
        self,
        num_proprio_obs: int,
        exteroception_dims: tuple[int, int],
        grid_idx: torch.Tensor,
        hidden_dim: int,
        num_heads: int,
        activation: str = "elu",
        conv_params: dict = {"kernel_size": 5, "stride": 1, "padding": "same"}
    ):
        """Dual-Path Attention encoder for proprioception and exteroception data.

        This encoder uses a parallel Global path and a Local Attention path.
        Features are concatenated to ensure both paths are trained effectively.

        Args:
            num_proprio_obs (int): Dimension of the 1D proprioceptive input.
            exteroception_dims (tuple[int, int]): Dimensions of the exteroception input (dim1, dim2).
            grid_idx (torch.Tensor): The grid indices of the ray pattern.
            hidden_dim (int): Dimension of the hidden layers.
            num_heads (int): Number of heads for the MultiheadAttention.
            activation (str): Activation function to use.
            conv_params (dict): Parameters for the convolutional layer.
        """
        super().__init__()
        self.num_proprio_obs = num_proprio_obs
        self.exteroception_dims = exteroception_dims
        self.num_patches = exteroception_dims[0] * exteroception_dims[1]

        self.activation = resolve_nn_activation(activation)

        # 1. Global Exteroception Path (Stable Summary)
        self.global_extero_encoder = nn.Sequential(
            nn.Linear(self.num_patches, hidden_dim),
            self.activation,
            nn.LayerNorm(hidden_dim)
        )
        # Branch Dropout: Forces the model to rely on Attention occasionally
        self.branch_dropout = nn.Dropout(p=0.2)

        # 2. Local Exteroception Path (Attention Feature Extraction)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, **conv_params),
            self.activation,
            nn.Conv2d(16, hidden_dim, **conv_params),
            self.activation,
        )
        self.position_encoder = nn.Embedding(self.num_patches, hidden_dim)
        self.extero_norm = nn.LayerNorm(hidden_dim)

        # 3. Proprioception processing (The Query)
        self.proprioception_encoder = nn.Linear(num_proprio_obs, hidden_dim)
        self.proprio_norm = nn.LayerNorm(hidden_dim)

        # 4. Cross-Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim = hidden_dim,
            num_heads = num_heads,
            batch_first = True,
        )

        # Buffer for patch indices
        self.register_buffer("patch_indices", torch.arange(self.num_patches))

        self.att_scores: torch.Tensor | None = None
        self.bfloat16()

    def forward(self, proprioception: torch.Tensor, exteroception: torch.Tensor, need_weights: bool = False) -> torch.Tensor:
        proprioception = proprioception.bfloat16()
        exteroception = exteroception.bfloat16()
        num_envs = proprioception.shape[0]

        # Fold exteroception into grid: (num_envs, history, height, width)
        exteroception_grid = exteroception.view(
            num_envs,
            -1,
            self.exteroception_dims[0],
            self.exteroception_dims[1],
        )

        # --- Path A: Global Terrain Summary ---
        # Apply dropout to the global branch to force Attention to learn
        global_terrain = self.global_extero_encoder(exteroception_grid[:, 0].flatten(1))
        global_terrain = self.branch_dropout(global_terrain)

        # --- Path B: Local Attention ---
        # 1. Patch Extraction
        extero_features = self.conv(exteroception_grid) # (num_envs, hidden_dim, H, W)
        extero_features = extero_features.flatten(2).permute(0, 2, 1) # (num_envs, num_patches, hidden_dim)

        # 2. Positional Embedding
        pos_embeddings = self.position_encoder(self.patch_indices) # (num_patches, hidden_dim)
        extero_encoded = self.extero_norm(extero_features + pos_embeddings.unsqueeze(0))

        # 3. Proprioception Query
        proprio_encoded = self.proprio_norm(self.activation(self.proprioception_encoder(proprioception)))
        proprio_encoded = proprio_encoded.unsqueeze(1) # (num_envs, 1, hidden_dim)

        # 4. Cross-Attention Calculation
        att_output, self.att_scores = self.attention(
            query = proprio_encoded,
            key   = extero_encoded,
            value = extero_encoded,
            need_weights=need_weights
        )
        att_output = att_output.squeeze(1)

        # --- Concatenation ---
        # We concatenate both representations so the policy MLP can use both
        # Result: [Global_Summary, Attention_Focus, Proprioception]
        out = torch.cat([global_terrain, att_output, proprioception], dim=-1)

        return out.float()
