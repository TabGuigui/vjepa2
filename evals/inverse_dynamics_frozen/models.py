import importlib
import logging

import torch
import torch.nn as nn

from src.models.attentive_pooler import AttentivePooler

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class InverseDynamicsProbe(nn.Module):
    """Attentive inverse-dynamics probe on frozen V-JEPA tokens."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        depth: int,
        action_dim: int,
        action_steps: int = 4,
        use_activation_checkpointing: bool = True,
    ):
        super().__init__()
        if action_steps <= 0:
            raise ValueError("action_steps must be positive")
        if action_dim % action_steps != 0:
            raise ValueError(f"action_dim={action_dim} must be divisible by action_steps={action_steps}")
        self.action_dim = action_dim
        self.action_steps = action_steps
        self.step_dim = action_dim // action_steps
        self.pooler = AttentivePooler(
            num_queries=action_steps,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            use_activation_checkpointing=use_activation_checkpointing,
        )
        self.regressor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, self.step_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.pooler(tokens)
        x = self.regressor(x)
        return x.flatten(1)


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
    num_heads: int,
    num_blocks: int,
    action_dim: int,
    action_steps: int,
    device: torch.device,
    num_probes: int,
):
    probes = [
        InverseDynamicsProbe(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=num_blocks,
            action_dim=action_dim,
            action_steps=action_steps,
            use_activation_checkpointing=True,
        ).to(device)
        for _ in range(num_probes)
    ]
    logger.info(probes[0])
    return probes
