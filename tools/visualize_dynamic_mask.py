#!/usr/bin/env python
import argparse
from pathlib import Path
import random
import sys

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.vjepa.transforms import make_transforms
from src.masks.multiseq_multiblock3d import MaskCollator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mask-index", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def read_video(path, num_frames, size):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"failed to open video: {path}")

    frames = []
    while len(frames) < num_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (size[1], size[0]), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()

    if len(frames) < num_frames:
        raise ValueError(f"video has {len(frames)} readable frames, need {num_frames}")
    return frames


def frames_to_rgb_buffer(frames):
    rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    return np.stack(rgb)


def apply_train_transform(frames, cfg, crop_size):
    data_aug = cfg.get("data_aug", {})
    transform = make_transforms(
        random_horizontal_flip=data_aug.get("random_horizontal_flip", True),
        random_resize_aspect_ratio=data_aug.get("random_resize_aspect_ratio", (3 / 4, 4 / 3)),
        random_resize_scale=data_aug.get("random_resize_scale", (0.3, 1.0)),
        reprob=data_aug.get("reprob", 0.0),
        auto_augment=data_aug.get("auto_augment", False),
        motion_shift=data_aug.get("motion_shift", False),
        crop_size=crop_size,
    )
    return transform(frames_to_rgb_buffer(frames))


def transformed_tensor_to_bgr_frames(tensor, auto_augment):
    if auto_augment:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1, 1)
        rgb = (tensor.detach().cpu() * std + mean) * 255.0
    else:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1, 1) * 255.0
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1, 1) * 255.0
        rgb = tensor.detach().cpu() * std + mean

    rgb = rgb.clamp(0, 255).byte().permute(1, 2, 3, 0).numpy()
    return [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in rgb]


def mask_to_volume(mask, duration, height, width):
    volume = torch.zeros(duration * height * width, dtype=torch.bool)
    volume[mask.cpu()] = True
    return volume.view(duration, height, width)


def overlay_masks(frames, mask_volumes, tubelet_size, patch_size, alpha):
    out = []
    patch_h, patch_w = patch_size
    colors = [
        torch.tensor([0, 0, 255], dtype=torch.float32),    # red
        torch.tensor([0, 255, 0], dtype=torch.float32),    # green
        torch.tensor([255, 0, 0], dtype=torch.float32),    # blue
        torch.tensor([0, 255, 255], dtype=torch.float32),  # yellow
        torch.tensor([255, 0, 255], dtype=torch.float32),  # magenta
        torch.tensor([255, 255, 0], dtype=torch.float32),  # cyan
    ]

    for frame_idx, frame in enumerate(frames):
        painted = torch.from_numpy(frame.copy()).float()
        for mask_idx, mask_volume in enumerate(mask_volumes):
            t_idx = min(frame_idx // tubelet_size, mask_volume.size(0) - 1)
            token_mask = mask_volume[t_idx]
            overlay_color = colors[mask_idx % len(colors)]
            for y in range(token_mask.size(0)):
                for x in range(token_mask.size(1)):
                    if not token_mask[y, x]:
                        continue
                    y0, y1 = y * patch_h, (y + 1) * patch_h
                    x0, x1 = x * patch_w, (x + 1) * patch_w
                    painted[y0:y1, x0:x1] = (
                        painted[y0:y1, x0:x1] * (1.0 - alpha) + overlay_color * alpha
                    )
        out.append(painted.clamp(0, 255).byte().numpy())
    return out


def write_video(path, frames, fps):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        writer.write(frame)
    writer.release()
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"failed to write video: {output}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    crop_size = data_cfg["crop_size"]
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    else:
        crop_size = tuple(crop_size)
    patch_size = data_cfg["patch_size"]
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    else:
        patch_size = tuple(patch_size)

    fpc = data_cfg["dataset_fpcs"][0]
    tubelet_size = data_cfg["tubelet_size"]
    frames = read_video(args.video, fpc, crop_size)
    video_tensor = apply_train_transform(frames, cfg, crop_size)
    vis_frames = transformed_tensor_to_bgr_frames(
        video_tensor,
        auto_augment=cfg.get("data_aug", {}).get("auto_augment", False),
    )

    collator = MaskCollator(
        cfgs_mask=cfg["mask"],
        dataset_fpcs=[fpc],
        crop_size=crop_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
    )
    sample = ([video_tensor], 0, [list(range(fpc))])
    collated = collator([sample])[0]
    masks_pred = collated[2]

    duration = fpc // tubelet_size
    grid_h = crop_size[0] // patch_size[0]
    grid_w = crop_size[1] // patch_size[1]

    if args.mask_index is None:
        selected_indices = list(range(len(masks_pred)))
    else:
        mask_index = args.mask_index
        if mask_index < 0:
            mask_index = len(masks_pred) + mask_index
        if mask_index < 0 or mask_index >= len(masks_pred):
            raise IndexError(f"mask-index {args.mask_index} out of range for {len(masks_pred)} masks")
        selected_indices = [mask_index]

    mask_volumes = [
        mask_to_volume(masks_pred[mask_idx][0], duration, grid_h, grid_w)
        for mask_idx in selected_indices
    ]
    overlayed = overlay_masks(vis_frames, mask_volumes, tubelet_size, patch_size, args.alpha)
    write_video(args.output, overlayed, data_cfg.get("fps", 2))
    print(f"wrote {args.output}")
    for mask_idx, mask_volume in zip(selected_indices, mask_volumes):
        print(f"mask_index={mask_idx}, masked_tokens={int(mask_volume.sum().item())}/{duration * grid_h * grid_w}")


if __name__ == "__main__":
    main()
