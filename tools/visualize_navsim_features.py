import argparse
import csv
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from PIL import Image

import src.models.vision_transformer as video_vit


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="/data/chainflow-vla/outputs/vjepa2/lamp-fpc4-camf0-256x512-vitg-navsim-50wclip/latest.pt",
    )
    parser.add_argument(
        "--csv",
        default="/data/chainflow-vla/opendv_vjepa_fpc4_camf0/train.csv",
    )
    parser.add_argument("--index", type=int, default=1000)
    parser.add_argument("--output-dir", default="/data/chainflow-vla/outputs/vjepa2/lamp_feature_viz")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--tubelet-size", type=int, default=2)
    parser.add_argument("--model-name", default="vit_giant_xformers")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--overlay-fps", type=float, default=2.0)
    return parser.parse_args()


def read_csv_video(csv_path, index):
    with open(csv_path, "r") as f:
        rows = list(csv.reader(f, delimiter=" "))
    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")
    return rows[index % len(rows)][0]


def load_video(path, frames, height, width):
    vr = VideoReader(path, ctx=cpu(0))
    if len(vr) == 0:
        raise RuntimeError(f"Empty video: {path}")
    indices = np.linspace(0, len(vr) - 1, num=frames).astype(np.int64)
    video = vr.get_batch(indices).asnumpy()
    resized = [cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in video]
    return np.stack(resized, axis=0)


def preprocess(video):
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32) * 255.0
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32) * 255.0
    x = torch.from_numpy(video).float().permute(3, 0, 1, 2)
    x = x.view(3, -1).permute(1, 0).sub_(mean).div_(std).permute(1, 0)
    x = x.view(3, video.shape[0], video.shape[1], video.shape[2])
    return x.unsqueeze(0)


def build_model(args):
    model = video_vit.__dict__[args.model_name](
        img_size=(args.height, args.width),
        patch_size=args.patch_size,
        num_frames=args.frames,
        tubelet_size=args.tubelet_size,
        uniform_power=True,
        use_sdpa=True,
        use_activation_checkpointing=False,
        use_rope=True,
    )
    return model


def load_encoder_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["encoder"]
    state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
    msg = model.load_state_dict(state, strict=True)
    print(f"Loaded encoder from {checkpoint_path}: {msg}")


def normalize_img(x):
    x = x - x.min()
    x = x / (x.max() + 1e-6)
    return (x * 255.0).clip(0, 255).astype(np.uint8)


def pca_rgb(features):
    x = features - features.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(x.float(), full_matrices=False)
    basis = vh[:3].T
    max_abs_indices = basis.abs().argmax(dim=0)
    signs = torch.sign(basis[max_abs_indices, torch.arange(basis.shape[1], device=basis.device)])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    basis = basis * signs
    rgb = x @ basis[:, :3]
    return normalize_img(rgb.cpu().numpy())


def fit_pca_basis(features):
    x = features - features.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(x.float(), full_matrices=False)
    basis = vh[:3].T
    max_abs_indices = basis.abs().argmax(dim=0)
    signs = torch.sign(basis[max_abs_indices, torch.arange(basis.shape[1], device=basis.device)])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return features.mean(dim=0, keepdim=True), basis * signs


def blend_overlay(image, overlay, alpha):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    image = image.astype(np.float32)
    overlay = overlay.astype(np.float32)
    return ((1.0 - alpha) * image + alpha * overlay).clip(0, 255).astype(np.uint8)


def write_rgb_video(path, frames, fps, width, height):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def save_temporal_feature_video(grid, video, args, output_dir):
    t_grid, h_grid, w_grid, channels = grid.shape
    flat = grid.reshape(-1, channels)
    pca_mean, pca_basis = fit_pca_basis(flat)
    all_rgb = ((flat - pca_mean) @ pca_basis[:, :3]).cpu().numpy()
    all_rgb = all_rgb - all_rgb.min()
    all_rgb = all_rgb / (all_rgb.max() + 1e-6)
    all_rgb = (all_rgb * 255.0).clip(0, 255).astype(np.uint8).reshape(t_grid, h_grid, w_grid, 3)

    overlay_frames = []
    pca_frames = []
    raw_frames = []
    for chunk_idx in range(t_grid):
        frame_idx = min((chunk_idx + 1) * args.tubelet_size - 1, video.shape[0] - 1)
        raw_frame = video[frame_idx]
        pca_grid = all_rgb[chunk_idx]
        pca_image = cv2.resize(pca_grid, (args.width, args.height), interpolation=cv2.INTER_CUBIC)
        overlay = blend_overlay(raw_frame, pca_image, args.overlay_alpha)

        raw_frames.append(raw_frame)
        pca_frames.append(pca_image)
        overlay_frames.append(overlay)

    write_rgb_video(output_dir / "temporal_raw.mp4", raw_frames, args.overlay_fps, args.width, args.height)
    write_rgb_video(output_dir / "temporal_pca.mp4", pca_frames, args.overlay_fps, args.width, args.height)
    write_rgb_video(output_dir / "temporal_pca_overlay.mp4", overlay_frames, args.overlay_fps, args.width, args.height)


def save_feature_images(features, video, args, output_dir):
    t_grid = args.frames // args.tubelet_size
    h_grid = args.height // args.patch_size
    w_grid = args.width // args.patch_size
    grid = features.reshape(t_grid, h_grid, w_grid, -1)
    grid_mean = grid.mean(dim=0)

    last_frame = video[-1]
    pca_grid = pca_rgb(grid_mean.reshape(-1, grid_mean.shape[-1])).reshape(h_grid, w_grid, 3)
    pca_nearest = cv2.resize(pca_grid, (args.width, args.height), interpolation=cv2.INTER_NEAREST)
    pca_smooth = cv2.resize(pca_grid, (args.width, args.height), interpolation=cv2.INTER_CUBIC)
    Image.fromarray(pca_nearest).save(output_dir / "pca_rgb.png")
    Image.fromarray(pca_smooth).save(output_dir / "pca_rgb_upsampled.png")
    Image.fromarray(blend_overlay(last_frame, pca_smooth, args.overlay_alpha)).save(output_dir / "pca_rgb_overlay.png")
    save_temporal_feature_video(grid, video, args, output_dir)

    center = grid_mean[h_grid // 2, w_grid // 2]
    sim = F.cosine_similarity(grid_mean, center[None, None, :], dim=-1)
    sim_img = normalize_img(sim.cpu().numpy())
    sim_img = cv2.applyColorMap(cv2.resize(sim_img, (args.width, args.height), interpolation=cv2.INTER_CUBIC), cv2.COLORMAP_JET)
    sim_img = cv2.cvtColor(sim_img, cv2.COLOR_BGR2RGB)
    Image.fromarray(sim_img).save(output_dir / "center_similarity.png")
    Image.fromarray(blend_overlay(last_frame, sim_img, args.overlay_alpha)).save(output_dir / "center_similarity_overlay.png")

    Image.fromarray(last_frame).save(output_dir / "input_last_frame.png")
    np.savez_compressed(
        output_dir / "features.npz",
        features=features.cpu().numpy(),
        grid_mean=grid_mean.cpu().numpy(),
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    video_path = read_csv_video(args.csv, args.index)
    print(f"Video: {video_path}")

    video = load_video(video_path, args.frames, args.height, args.width)
    x = preprocess(video).to(device)

    model = build_model(args).to(device).eval()
    load_encoder_weights(model, args.checkpoint)

    with torch.inference_mode():
        features = model(x)[0]

    save_feature_images(features, video, args, output_dir)
    print(f"Saved feature visualization to {output_dir}")


if __name__ == "__main__":
    main()
