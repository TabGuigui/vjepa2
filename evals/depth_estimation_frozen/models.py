import importlib
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class LinearDepthProbe(nn.Module):
    """DINO-style dense linear probe on frozen patch tokens.

    This intentionally stays linear: patch tokens are normalized with a learned
    BatchNorm and projected to one depth value per patch. The patch grid is then
    bilinearly upsampled to the target depth resolution.
    """

    def __init__(
        self,
        embed_dim: int,
        grid_size: int,
        output_size: int,
        temporal_pool: str = "mean",
        activation: str = "softplus",
    ):
        super().__init__()
        if temporal_pool not in {"mean", "last"}:
            raise ValueError(f"unsupported temporal_pool={temporal_pool}")
        if activation not in {"none", "relu", "softplus"}:
            raise ValueError(f"unsupported activation={activation}")
        self.grid_size = grid_size
        self.output_size = output_size
        self.temporal_pool = temporal_pool
        self.activation = activation
        self.norm = nn.BatchNorm1d(embed_dim, affine=True)
        self.proj = nn.Linear(embed_dim, 1)

    def _spatial_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, dim = tokens.shape
        spatial_tokens = self.grid_size * self.grid_size
        if num_tokens % spatial_tokens != 0:
            raise ValueError(
                f"num_tokens={num_tokens} is not divisible by grid_size^2={spatial_tokens}; "
                "check resolution and patch_size"
            )
        temporal_tokens = num_tokens // spatial_tokens
        tokens = tokens.view(batch_size, temporal_tokens, spatial_tokens, dim)
        if self.temporal_pool == "last":
            return tokens[:, -1]
        return tokens.mean(dim=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self._spatial_tokens(tokens)
        depth = self.proj(self.norm(tokens.transpose(1, 2)).transpose(1, 2))
        depth = depth.transpose(1, 2).reshape(tokens.shape[0], 1, self.grid_size, self.grid_size)
        depth = F.interpolate(depth, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)
        if self.activation == "softplus":
            depth = F.softplus(depth)
        elif self.activation == "relu":
            depth = F.relu(depth)
        return depth


def init_module(
    module_name,
    device,
    frames_per_clip,
    resolution,
    checkpoint,
    model_kwargs,
    wrapper_kwargs,
):
    model = (
        importlib.import_module(f"{module_name}")
        .init_module(
            frames_per_clip=frames_per_clip,
            resolution=resolution,
            checkpoint=checkpoint,
            model_kwargs=model_kwargs,
            wrapper_kwargs=wrapper_kwargs,
        )
        .to(device)
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    logger.info(model)
    return model


def init_probe(
    embed_dim: int,
    grid_size: int,
    output_size: int,
    temporal_pool: str,
    activation: str,
    device: torch.device,
    num_probes: int,
):
    probes = [
        LinearDepthProbe(
            embed_dim=embed_dim,
            grid_size=grid_size,
            output_size=output_size,
            temporal_pool=temporal_pool,
            activation=activation,
        ).to(device)
        for _ in range(num_probes)
    ]
    logger.info(probes[0])
    return probes
