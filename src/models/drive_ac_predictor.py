# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial

import torch
import torch.nn as nn

from src.models.utils.modules import ACBlock as Block
from src.models.utils.modules import build_action_block_causal_attention_mask
from src.utils.tensors import trunc_normal_


class VisionTransformerPredictorDriveAC(nn.Module):
    """Driving action-conditioned latent dynamics predictor.

    Stage-2 future predictor for AD-JEPA. The frozen encoder provides a mixed
    latent sequence:

      context latents: frame0-1, frame2-3 from standard tubelets
      future latents: frame4, frame5, frame6, frame7 from duplicated-frame tubelets

    The predictor consumes source latents and xy actions, then predicts the
    future latent for each action step. For the default fpc=8 setting:

      source latents: [z01, z23, z4, z5, z6]  # 5 steps
      actions:        [a3->4, a4->5, a5->6, a6->7]  # 4 steps
      predictions:    [z4, z5, z6, z7]
    """

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        embed_dim=768,
        predictor_embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=True,
        use_silu=False,
        wide_silu=True,
        is_frame_causal=True,
        use_activation_checkpointing=False,
        use_rope=True,
        action_embed_dim=2,
        context_steps=2,
        future_steps=4,
        **kwargs,
    ):
        super().__init__()
        self.is_frame_causal = is_frame_causal
        self.context_steps = int(context_steps)
        self.future_steps = int(future_steps)
        self.max_source_steps = self.context_steps + self.future_steps - 1

        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.action_encoder = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)

        if type(img_size) is int:
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        self.grid_height = img_size[0] // self.patch_size
        self.grid_width = img_size[1] // self.patch_size
        self.use_activation_checkpointing = use_activation_checkpointing

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.uniform_power = uniform_power
        self.use_rope = use_rope
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=self.grid_height,
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

        attn_mask = None
        if self.is_frame_causal:
            attn_mask = build_action_block_causal_attention_mask(
                self.max_source_steps,
                self.grid_height,
                self.grid_width,
                add_tokens=1,
            )
        self.attn_mask = attn_mask

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def forward(self, source_tokens, actions):
        """Predict future visual latents conditioned on xy actions.

        Args:
            source_tokens: Source visual tokens with shape [B, S * H * W, C].
                For the default T=6 latent plan, S=5:
                [z01, z23, z4, z5, z6].
            actions: Per-step xy actions with shape [B, A, 2].
                For the default plan, A=4:
                [frame3->4, frame4->5, frame5->6, frame6->7].

        Returns:
            Predicted future tokens with shape [B, A * H * W, C].
        """
        x = self.predictor_embed(source_tokens)
        B, N_ctxt, D = x.size()
        tokens_per_frame = self.grid_height * self.grid_width
        if N_ctxt % tokens_per_frame != 0:
            raise ValueError(f"Expected token count divisible by {tokens_per_frame}, got {N_ctxt}")
        source_steps = N_ctxt // tokens_per_frame
        action_steps = actions.size(1)

        expected_source_steps = self.context_steps + action_steps - 1
        if source_steps != expected_source_steps:
            raise ValueError(
                f"Expected source temporal length {expected_source_steps} for "
                f"{self.context_steps=} and {action_steps=}, got {source_steps}"
            )
        if action_steps > self.future_steps:
            raise ValueError(f"Expected at most {self.future_steps} action steps, got {action_steps}")

        cond_tokens = 1
        action_tokens = self.action_encoder(actions)
        num_context_only_steps = source_steps - action_steps
        context_tokens = action_tokens.new_zeros(B, num_context_only_steps, D)
        action_tokens = torch.cat([context_tokens, action_tokens], dim=1).unsqueeze(2)

        x = x.view(B, source_steps, tokens_per_frame, D)
        x = torch.cat([action_tokens, x], dim=2).flatten(1, 2)

        attn_mask = None
        if self.attn_mask is not None:
            attn_mask = self.attn_mask[: x.size(1), : x.size(1)].to(x.device, non_blocking=True)

        for blk in self.predictor_blocks:
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk,
                    x,
                    mask=None,
                    attn_mask=attn_mask,
                    T=source_steps,
                    H=self.grid_height,
                    W=self.grid_width,
                    action_tokens=cond_tokens,
                    use_reentrant=False,
                )
            else:
                x = blk(
                    x,
                    mask=None,
                    attn_mask=attn_mask,
                    T=source_steps,
                    H=self.grid_height,
                    W=self.grid_width,
                    action_tokens=cond_tokens,
                )

        x = x.view(B, source_steps, cond_tokens + tokens_per_frame, D)
        x = x[:, :, cond_tokens:, :]
        x = x[:, num_context_only_steps:, :, :].flatten(1, 2)
        x = self.predictor_norm(x)
        x = self.predictor_proj(x)
        return x


def vit_drive_ac_predictor(**kwargs):
    model = VisionTransformerPredictorDriveAC(
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
