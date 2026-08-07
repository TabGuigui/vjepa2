import logging
import math
import os
import pprint
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from evals.depth_estimation_frozen.dataloader import make_dataloader
from evals.depth_estimation_frozen.models import init_module, init_probe
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import AllReduce, init_distributed
from src.utils.logging import AverageMeter, CSVLogger

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def main(args_eval, resume_preempt=False):
    val_only = args_eval.get("val_only", False)
    pretrain_folder = args_eval.get("folder", None)
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    eval_tag = args_eval.get("tag", None)

    args_pretrain = args_eval.get("model_kwargs")
    checkpoint = args_pretrain.get("checkpoint")
    module_name = args_pretrain.get("module_name")
    args_model = args_pretrain.get("pretrain_kwargs")
    args_wrapper = args_pretrain.get("wrapper_kwargs")

    args_exp = args_eval.get("experiment")
    args_probe = args_exp.get("probe")
    temporal_pool = args_probe.get("temporal_pool", "mean")
    activation = args_probe.get("activation", "softplus")

    args_data = args_exp.get("data")
    train_csv = args_data.get("dataset_train")
    val_csv = args_data.get("dataset_val")
    resolution = args_data.get("resolution", 256)
    frames_per_clip = args_data.get("frames_per_clip", 4)
    depth_scale = args_data.get("depth_scale", 1.0)
    min_depth = args_data.get("min_depth", 1e-3)
    max_depth = args_data.get("max_depth", 80.0)
    num_workers = args_data.get("num_workers", 8)
    pin_mem = args_data.get("pin_memory", True)

    patch_size = args_model["encoder"].get("patch_size", 16)
    grid_size = resolution // patch_size
    if resolution % patch_size != 0:
        raise ValueError(f"resolution={resolution} must be divisible by patch_size={patch_size}")

    args_opt = args_exp.get("optimization")
    batch_size = args_opt.get("batch_size", 16)
    num_epochs = args_opt.get("num_epochs", 20)
    use_bfloat16 = args_opt.get("use_bfloat16", True)
    opt_kwargs = [
        dict(
            ref_wd=kwargs.get("weight_decay"),
            final_wd=kwargs.get("final_weight_decay"),
            start_lr=kwargs.get("start_lr"),
            ref_lr=kwargs.get("lr"),
            final_lr=kwargs.get("final_lr"),
            warmup=kwargs.get("warmup"),
        )
        for kwargs in args_opt.get("multihead_kwargs")
    ]

    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    folder = os.path.join(pretrain_folder, "depth_estimation_frozen")
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")

    if rank == 0:
        csv_logger = CSVLogger(
            log_file,
            ("%d", "epoch"),
            ("%.6f", "train_loss"),
            ("%.6f", "train_rmse"),
            ("%.6f", "train_abs_rel"),
            ("%.6f", "train_delta1"),
            ("%.6f", "val_loss"),
            ("%.6f", "val_rmse"),
            ("%.6f", "val_abs_rel"),
            ("%.6f", "val_delta1"),
        )

    encoder = init_module(
        module_name=module_name,
        frames_per_clip=frames_per_clip,
        resolution=resolution,
        checkpoint=checkpoint,
        model_kwargs=args_model,
        wrapper_kwargs=args_wrapper,
        device=device,
    )
    probes = init_probe(
        embed_dim=encoder.embed_dim,
        grid_size=grid_size,
        output_size=resolution,
        temporal_pool=temporal_pool,
        activation=activation,
        device=device,
        num_probes=len(opt_kwargs),
    )
    probes = [DistributedDataParallel(p, static_graph=True) for p in probes]

    train_loader, train_sampler = make_dataloader(
        csv_path=train_csv,
        batch_size=batch_size,
        frames_per_clip=frames_per_clip,
        resolution=resolution,
        depth_scale=depth_scale,
        min_depth=min_depth,
        max_depth=max_depth,
        world_size=world_size,
        rank=rank,
        training=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )
    val_loader, _ = make_dataloader(
        csv_path=val_csv,
        batch_size=batch_size,
        frames_per_clip=frames_per_clip,
        resolution=resolution,
        depth_scale=depth_scale,
        min_depth=min_depth,
        max_depth=max_depth,
        world_size=world_size,
        rank=rank,
        training=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )
    ipe = len(train_loader)
    logger.info(f"Dataloader created... iterations per epoch: {ipe}")

    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        probes=probes,
        opt_kwargs=opt_kwargs,
        iterations_per_epoch=ipe,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16,
    )

    start_epoch = 0
    if resume_checkpoint and os.path.exists(latest_path):
        probes, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=latest_path,
            probes=probes,
            opt=optimizer,
            scaler=scaler,
            val_only=val_only,
        )
        for _ in range(start_epoch * ipe):
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

    def save_checkpoint(epoch):
        save_dict = {
            "probes": [p.state_dict() for p in probes],
            "opt": [o.state_dict() for o in optimizer],
            "scaler": None if scaler is None else [s.state_dict() for s in scaler],
            "epoch": epoch,
            "batch_size": batch_size,
            "world_size": world_size,
        }
        if rank == 0:
            torch.save(save_dict, latest_path)

    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Epoch {epoch + 1}")
        train_sampler.set_epoch(epoch)
        if val_only:
            train_metrics = {"loss": -1.0, "rmse": -1.0, "abs_rel": -1.0, "delta1": -1.0}
        else:
            train_metrics = run_one_epoch(
                device=device,
                training=True,
                encoder=encoder,
                probes=probes,
                scaler=scaler,
                optimizer=optimizer,
                scheduler=scheduler,
                wd_scheduler=wd_scheduler,
                data_loader=train_loader,
                use_bfloat16=use_bfloat16,
            )
        val_metrics = run_one_epoch(
            device=device,
            training=False,
            encoder=encoder,
            probes=probes,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=val_loader,
            use_bfloat16=use_bfloat16,
        )
        logger.info(
            "[%5d] train loss %.6f rmse %.6f abs_rel %.6f d1 %.6f | val loss %.6f rmse %.6f abs_rel %.6f d1 %.6f"
            % (
                epoch + 1,
                train_metrics["loss"],
                train_metrics["rmse"],
                train_metrics["abs_rel"],
                train_metrics["delta1"],
                val_metrics["loss"],
                val_metrics["rmse"],
                val_metrics["abs_rel"],
                val_metrics["delta1"],
            )
        )
        if rank == 0:
            csv_logger.log(
                epoch + 1,
                train_metrics["loss"],
                train_metrics["rmse"],
                train_metrics["abs_rel"],
                train_metrics["delta1"],
                val_metrics["loss"],
                val_metrics["rmse"],
                val_metrics["abs_rel"],
                val_metrics["delta1"],
            )
        if val_only:
            return
        save_checkpoint(epoch + 1)


def masked_l1_loss(pred, target, valid):
    denom = valid.sum().clamp_min(1.0)
    return ((pred - target).abs() * valid).sum() / denom


def depth_metrics(pred, target, valid):
    pred = pred.float().clamp_min(1e-6)
    target = target.float().clamp_min(1e-6)
    valid = valid.float()
    denom = valid.sum().clamp_min(1.0)
    error = (pred - target) * valid
    rmse = torch.sqrt(((error**2).sum() / denom).clamp_min(0.0))
    abs_rel = (((pred - target).abs() / target) * valid).sum() / denom
    ratio = torch.maximum(pred / target, target / pred)
    delta1 = ((ratio < 1.25).float() * valid).sum() / denom
    return rmse, abs_rel, delta1


def run_one_epoch(
    device,
    training,
    encoder,
    probes,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
):
    for probe in probes:
        probe.train(mode=training)

    loss_meters = [AverageMeter() for _ in probes]
    rmse_meters = [AverageMeter() for _ in probes]
    abs_rel_meters = [AverageMeter() for _ in probes]
    delta1_meters = [AverageMeter() for _ in probes]
    data_time = AverageMeter()

    for itr, (clips, targets, valid_masks) in enumerate(data_loader):
        itr_start = time.time()
        if training:
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

        clips = clips.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        valid_masks = valid_masks.to(device, non_blocking=True)
        batch_size = targets.shape[0]
        data_time.update((time.time() - itr_start) * 1000.0)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            with torch.no_grad():
                tokens = encoder(clips)
            preds = [probe(tokens) for probe in probes]
            losses = [masked_l1_loss(pred, targets, valid_masks) for pred in preds]

        with torch.no_grad():
            for meter_loss, meter_rmse, meter_abs_rel, meter_delta1, pred, loss in zip(
                loss_meters, rmse_meters, abs_rel_meters, delta1_meters, preds, losses
            ):
                rmse, abs_rel, delta1 = depth_metrics(pred, targets, valid_masks)
                meter_loss.update(float(AllReduce.apply(loss.detach())), batch_size)
                meter_rmse.update(float(AllReduce.apply(rmse.detach())), batch_size)
                meter_abs_rel.update(float(AllReduce.apply(abs_rel.detach())), batch_size)
                meter_delta1.update(float(AllReduce.apply(delta1.detach())), batch_size)

        if training:
            if use_bfloat16:
                [s.scale(loss).backward() for s, loss in zip(scaler, losses)]
                [s.step(opt) for s, opt in zip(scaler, optimizer)]
                [s.update() for s in scaler]
            else:
                [loss.backward() for loss in losses]
                [opt.step() for opt in optimizer]
            [opt.zero_grad() for opt in optimizer]

        if itr % 10 == 0 or itr == len(data_loader) - 1:
            best = int(np.argmin([m.avg for m in loss_meters]))
            logger.info(
                "[%5d] loss %.6f rmse %.6f abs_rel %.6f d1 %.6f [data %.1f ms] [mem %.2e]"
                % (
                    itr,
                    loss_meters[best].avg,
                    rmse_meters[best].avg,
                    abs_rel_meters[best].avg,
                    delta1_meters[best].avg,
                    data_time.avg,
                    torch.cuda.max_memory_allocated() / 1024.0**2,
                )
            )

    best = int(np.argmin([m.avg for m in loss_meters]))
    return {
        "loss": loss_meters[best].avg,
        "rmse": rmse_meters[best].avg,
        "abs_rel": abs_rel_meters[best].avg,
        "delta1": delta1_meters[best].avg,
    }


def init_opt(probes, iterations_per_epoch, opt_kwargs, num_epochs, use_bfloat16=False):
    optimizers, schedulers, wd_schedulers, scalers = [], [], [], []
    for probe, kwargs in zip(probes, opt_kwargs):
        param_groups = [
            {
                "params": (p for _, p in probe.named_parameters()),
                "mc_warmup_steps": int(kwargs.get("warmup") * iterations_per_epoch),
                "mc_start_lr": kwargs.get("start_lr"),
                "mc_ref_lr": kwargs.get("ref_lr"),
                "mc_final_lr": kwargs.get("final_lr"),
                "mc_ref_wd": kwargs.get("ref_wd"),
                "mc_final_wd": kwargs.get("final_wd"),
            }
        ]
        optimizers += [torch.optim.AdamW(param_groups)]
        schedulers += [WarmupCosineLRSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch))]
        wd_schedulers += [CosineWDSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch))]
        scalers += [torch.cuda.amp.GradScaler() if use_bfloat16 else None]
    return optimizers, scalers, schedulers, wd_schedulers


class WarmupCosineLRSchedule:
    def __init__(self, optimizer, T_max):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        for group in self.optimizer.param_groups:
            ref_lr = group.get("mc_ref_lr")
            final_lr = group.get("mc_final_lr")
            start_lr = group.get("mc_start_lr")
            warmup_steps = group.get("mc_warmup_steps")
            t_max = max(1, self.T_max - warmup_steps)
            if self._step < warmup_steps:
                progress = float(self._step) / float(max(1, warmup_steps))
                new_lr = start_lr + progress * (ref_lr - start_lr)
            else:
                progress = float(self._step - warmup_steps) / float(t_max)
                new_lr = max(final_lr, final_lr + (ref_lr - final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)))
            group["lr"] = new_lr


class CosineWDSchedule:
    def __init__(self, optimizer, T_max):
        self.optimizer = optimizer
        self.T_max = max(1, T_max)
        self._step = 0.0

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        for group in self.optimizer.param_groups:
            ref_wd = group.get("mc_ref_wd")
            final_wd = group.get("mc_final_wd")
            new_wd = final_wd + (ref_wd - final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))
            group["weight_decay"] = max(final_wd, new_wd) if final_wd <= ref_wd else min(final_wd, new_wd)


def load_checkpoint(device, r_path, probes, opt, scaler, val_only=False):
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    logger.info(f"read-path: {r_path}")
    msg = [probe.load_state_dict(pd) for probe, pd in zip(probes, checkpoint["probes"])]
    if val_only:
        logger.info(f"loaded probe with msg: {msg}")
        return probes, opt, scaler, 0
    epoch = checkpoint["epoch"]
    logger.info(f"loaded probe from epoch {epoch} with msg: {msg}")
    [o.load_state_dict(pd) for o, pd in zip(opt, checkpoint["opt"])]
    if scaler is not None:
        [s.load_state_dict(pd) for s, pd in zip(scaler, checkpoint["scaler"])]
    return probes, opt, scaler, epoch
