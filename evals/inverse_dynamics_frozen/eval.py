import logging
import math
import os
import pprint
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from evals.inverse_dynamics_frozen.dataloader import make_dataloader
from evals.inverse_dynamics_frozen.models import init_module, init_probe
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
    num_probe_blocks = args_probe.get("num_probe_blocks", 1)
    num_heads = args_probe.get("num_heads", 16)
    action_dim = args_probe.get("action_dim", 8)
    action_steps = args_probe.get("action_steps", 4)

    args_data = args_exp.get("data")
    train_csv = args_data.get("dataset_train")
    val_csv = args_data.get("dataset_val")
    resolution = args_data.get("resolution", 256)
    frames_per_clip = args_data.get("frames_per_clip", 8)
    action_start_col = args_data.get("action_start_col", 1)
    source_frames_per_clip = args_data.get("source_frames_per_clip", frames_per_clip)
    input_frame_start = args_data.get("input_frame_start", 0)
    num_workers = args_data.get("num_workers", 8)
    pin_mem = args_data.get("pin_memory", True)

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

    folder = os.path.join(pretrain_folder, "inverse_dynamics_frozen")
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
            ("%.6f", "train_l1"),
            ("%.6f", "train_rmse"),
            ("%.6f", "train_ade"),
            ("%.6f", "train_fde"),
            ("%.6f", "val_loss"),
            ("%.6f", "val_l1"),
            ("%.6f", "val_rmse"),
            ("%.6f", "val_ade"),
            ("%.6f", "val_fde"),
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
        num_heads=num_heads,
        num_blocks=num_probe_blocks,
        action_dim=action_dim,
        action_steps=action_steps,
        device=device,
        num_probes=len(opt_kwargs),
    )
    probes = [DistributedDataParallel(p, static_graph=True) for p in probes]

    train_loader, train_sampler = make_dataloader(
        csv_path=train_csv,
        batch_size=batch_size,
        frames_per_clip=frames_per_clip,
        resolution=resolution,
        action_dim=action_dim,
        action_start_col=action_start_col,
        source_frames_per_clip=source_frames_per_clip,
        input_frame_start=input_frame_start,
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
        action_dim=action_dim,
        action_start_col=action_start_col,
        source_frames_per_clip=source_frames_per_clip,
        input_frame_start=input_frame_start,
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
            train_metrics = {"loss": -1.0, "l1": -1.0, "rmse": -1.0, "ade": -1.0, "fde": -1.0}
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
            "[%5d] train loss %.6f l1 %.6f rmse %.6f ade %.6f fde %.6f | val loss %.6f l1 %.6f rmse %.6f ade %.6f fde %.6f"
            % (
                epoch + 1,
                train_metrics["loss"],
                train_metrics["l1"],
                train_metrics["rmse"],
                train_metrics["ade"],
                train_metrics["fde"],
                val_metrics["loss"],
                val_metrics["l1"],
                val_metrics["rmse"],
                val_metrics["ade"],
                val_metrics["fde"],
            )
        )
        if rank == 0:
            csv_logger.log(
                epoch + 1,
                train_metrics["loss"],
                train_metrics["l1"],
                train_metrics["rmse"],
                train_metrics["ade"],
                train_metrics["fde"],
                val_metrics["loss"],
                val_metrics["l1"],
                val_metrics["rmse"],
                val_metrics["ade"],
                val_metrics["fde"],
            )
        if val_only:
            return
        save_checkpoint(epoch + 1)


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

    criterion = torch.nn.SmoothL1Loss()
    loss_meters = [AverageMeter() for _ in probes]
    l1_meters = [AverageMeter() for _ in probes]
    rmse_meters = [AverageMeter() for _ in probes]
    ade_meters = [AverageMeter() for _ in probes]
    fde_meters = [AverageMeter() for _ in probes]
    data_time = AverageMeter()

    for itr, (clips, targets) in enumerate(data_loader):
        itr_start = time.time()
        if training:
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

        clips = clips.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = targets.shape[0]
        data_time.update((time.time() - itr_start) * 1000.0)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            with torch.no_grad():
                tokens = encoder(clips)
            preds = [probe(tokens) for probe in probes]
            losses = [criterion(pred, targets) for pred in preds]

        with torch.no_grad():
            for meter_loss, meter_l1, meter_rmse, meter_ade, meter_fde, pred, loss in zip(
                loss_meters, l1_meters, rmse_meters, ade_meters, fde_meters, preds, losses
            ):
                pred = pred.float()
                target = targets.float()
                error = pred - target
                l1 = error.abs().mean()
                rmse = torch.sqrt((error**2).mean())
                pred_traj = pred.view(batch_size, -1, 2).cumsum(dim=1)
                target_traj = target.view(batch_size, -1, 2).cumsum(dim=1)
                point_error = torch.linalg.norm(pred_traj - target_traj, dim=-1)
                ade = point_error.mean()
                fde = point_error[:, -1].mean()
                meter_loss.update(float(AllReduce.apply(loss.detach())), batch_size)
                meter_l1.update(float(AllReduce.apply(l1.detach())), batch_size)
                meter_rmse.update(float(AllReduce.apply(rmse.detach())), batch_size)
                meter_ade.update(float(AllReduce.apply(ade.detach())), batch_size)
                meter_fde.update(float(AllReduce.apply(fde.detach())), batch_size)

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
                "[%5d] loss %.6f l1 %.6f rmse %.6f ade %.6f fde %.6f [data %.1f ms] [mem %.2e]"
                % (
                    itr,
                    loss_meters[best].avg,
                    l1_meters[best].avg,
                    rmse_meters[best].avg,
                    ade_meters[best].avg,
                    fde_meters[best].avg,
                    data_time.avg,
                    torch.cuda.max_memory_allocated() / 1024.0**2,
                )
            )

    best = int(np.argmin([m.avg for m in loss_meters]))
    return {
        "loss": loss_meters[best].avg,
        "l1": l1_meters[best].avg,
        "rmse": rmse_meters[best].avg,
        "ade": ade_meters[best].avg,
        "fde": fde_meters[best].avg,
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
