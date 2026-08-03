# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys

import torch

import src.models.drive_ac_predictor as vit_drive_ac_pred
import src.models.vision_transformer as video_vit
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.schedulers import CosineWDSchedule, WSDSchedule

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def _load_state(module, state_dict, strict=False):
    state_dict = {
        k.replace("module.", "").replace("backbone.", "").replace("_orig_mod.", ""): v
        for k, v in state_dict.items()
    }
    unwrapped = module.module if hasattr(module, "module") else module
    return unwrapped.load_state_dict(state_dict, strict=strict)


def load_pretrained(
    r_path,
    encoder=None,
    predictor=None,
    context_encoder_key="encoder",
    load_predictor=False,
    load_encoder=True,
):
    logger.info(f"Loading pretrained model from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    epoch = checkpoint.get("epoch", -1)

    if load_encoder:
        pretrained_dict = checkpoint[context_encoder_key]
        msg = _load_state(encoder, pretrained_dict, strict=False)
        logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

    if load_predictor:
        pretrained_dict = checkpoint["predictor"]
        msg = _load_state(predictor, pretrained_dict, strict=False)
        logger.info(f"loaded pretrained predictor from epoch {epoch} with msg: {msg}")

    del checkpoint
    return encoder, predictor


def load_checkpoint(
    r_path,
    encoder,
    predictor,
    opt=None,
    scaler=None,
):
    logger.info(f"Loading checkpoint from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    epoch = checkpoint["epoch"]

    msg = _load_state(encoder, checkpoint["encoder"], strict=False)
    logger.info(f"loaded encoder from epoch {epoch} with msg: {msg}")

    msg = _load_state(predictor, checkpoint["predictor"], strict=False)
    logger.info(f"loaded predictor from epoch {epoch} with msg: {msg}")

    if opt is not None and "opt" in checkpoint:
        opt.load_state_dict(checkpoint["opt"])

    if scaler is not None and checkpoint.get("scaler", None) is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    logger.info(f"loaded optimizer state from epoch {epoch}")
    logger.info(f"read-path: {r_path}")
    del checkpoint
    return encoder, predictor, opt, scaler, epoch


def init_video_model(
    device,
    patch_size=16,
    max_num_frames=8,
    tubelet_size=2,
    model_name="vit_giant_xformers",
    crop_size=(256, 512),
    pred_depth=12,
    pred_num_heads=None,
    pred_embed_dim=1024,
    uniform_power=False,
    use_sdpa=False,
    use_rope=True,
    use_silu=False,
    use_pred_silu=False,
    wide_silu=True,
    pred_is_frame_causal=True,
    use_activation_checkpointing=False,
    action_embed_dim=2,
    context_steps=2,
    future_steps=4,
):
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        use_rope=use_rope,
    )

    predictor = vit_drive_ac_pred.__dict__["vit_drive_ac_predictor"](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        action_embed_dim=action_embed_dim,
        depth=pred_depth,
        is_frame_causal=pred_is_frame_causal,
        num_heads=encoder.num_heads if pred_num_heads is None else pred_num_heads,
        uniform_power=uniform_power,
        use_rope=use_rope,
        use_sdpa=use_sdpa,
        use_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        context_steps=context_steps,
        future_steps=future_steps,
    )

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    logger.info(predictor)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Encoder number of trainable parameters: {count_parameters(encoder)}")
    logger.info(f"Predictor number of trainable parameters: {count_parameters(predictor)}")

    return encoder, predictor


def init_opt(
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    anneal,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
):
    param_groups = [
        {
            "params": (p for n, p in predictor.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
        },
        {
            "params": (p for n, p in predictor.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
    ]

    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    scheduler = WSDSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        anneal_steps=int(anneal * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler
