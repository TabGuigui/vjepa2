import csv
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

try:
    from decord import VideoReader, cpu
except Exception:
    VideoReader = None
    cpu = None


DEFAULT_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".avi", ".mov", ".mkv"}


def _split_line(line: str) -> List[str]:
    line = line.strip()
    if not line:
        return []
    if "," in line:
        return next(csv.reader([line]))
    return line.split()


def _is_header(tokens: Sequence[str]) -> bool:
    if not tokens:
        return False
    return tokens[0].lower() in {"path", "image", "image_path", "video", "video_path", "rgb", "rgb_path"}


def _center_crop_square(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return frame[top : top + side, left : left + side]


def _load_video_decord(path: str, frames_per_clip: int) -> np.ndarray:
    vr = VideoReader(path, num_threads=1, ctx=cpu(0))
    num_frames = len(vr)
    if num_frames <= 0:
        raise RuntimeError(f"empty video: {path}")
    indices = np.linspace(0, num_frames - 1, frames_per_clip).round().astype(np.int64)
    return vr.get_batch(indices).asnumpy()


def _load_video_cv2(path: str, frames_per_clip: int) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames <= 0:
        cap.release()
        raise RuntimeError(f"empty video: {path}")
    indices = np.linspace(0, num_frames - 1, frames_per_clip).round().astype(np.int64)
    frames = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"failed to decode frame {index} from {path}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames, axis=0)


def load_rgb_clip(path: str, frames_per_clip: int) -> np.ndarray:
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return np.repeat(image[None], frames_per_clip, axis=0)
    if suffix in VIDEO_EXTENSIONS:
        if VideoReader is not None:
            try:
                return _load_video_decord(path, frames_per_clip)
            except Exception:
                pass
        return _load_video_cv2(path, frames_per_clip)
    raise RuntimeError(f"unsupported RGB path extension: {path}")


def load_depth(path: str, depth_scale: float) -> np.ndarray:
    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        depth = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        key = "depth" if "depth" in data else data.files[0]
        depth = data[key]
    else:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"failed to read depth: {path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth.astype(np.float32) / float(depth_scale)


class DepthCsvDataset(Dataset):
    """CSV dataset for frozen-backbone depth probing.

    Expected row format:
      /path/to/rgb_or_video /path/to/depth

    Depth may be .npy/.npz or image depth. For uint16 PNG depths, set depth_scale
    in config, e.g. 1000.0 for millimeters to meters.
    """

    def __init__(
        self,
        csv_path: str,
        frames_per_clip: int,
        resolution: int,
        depth_scale: float = 1.0,
        min_depth: float = 1e-3,
        max_depth: float = 80.0,
        normalize: Tuple[Sequence[float], Sequence[float]] = DEFAULT_NORMALIZATION,
        skip_broken: bool = True,
    ):
        self.csv_path = Path(csv_path)
        self.frames_per_clip = frames_per_clip
        self.resolution = resolution
        self.depth_scale = depth_scale
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.normalize = normalize
        self.skip_broken = skip_broken
        self.samples = self._read_samples()
        if len(self.samples) == 0:
            raise RuntimeError(f"No depth samples found in {csv_path}")

    def _read_samples(self):
        samples = []
        with self.csv_path.open("r") as handle:
            for line in handle:
                tokens = _split_line(line)
                if not tokens or _is_header(tokens):
                    continue
                if len(tokens) < 2:
                    continue
                samples.append((tokens[0], tokens[1]))
        return samples

    def __len__(self):
        return len(self.samples)

    def _transform_rgb(self, frames: np.ndarray) -> torch.Tensor:
        processed = []
        for frame in frames:
            frame = _center_crop_square(frame)
            frame = cv2.resize(frame, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
            tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            processed.append(tensor)
        video = torch.stack(processed, dim=1)  # C T H W
        mean = torch.tensor(self.normalize[0], dtype=video.dtype).view(3, 1, 1, 1)
        std = torch.tensor(self.normalize[1], dtype=video.dtype).view(3, 1, 1, 1)
        return (video - mean) / std

    def _transform_depth(self, depth: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        depth = _center_crop_square(depth)
        depth = cv2.resize(depth, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
        depth = torch.from_numpy(depth).float().unsqueeze(0)
        valid = torch.isfinite(depth) & (depth > self.min_depth) & (depth < self.max_depth)
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        return depth, valid.float()

    def __getitem__(self, index: int):
        rgb_path, depth_path = self.samples[index]
        try:
            frames = load_rgb_clip(rgb_path, self.frames_per_clip)
            depth = load_depth(depth_path, self.depth_scale)
            video = self._transform_rgb(frames)
            depth, valid = self._transform_depth(depth)
        except Exception:
            if not self.skip_broken:
                raise
            next_index = (index + 1) % len(self.samples)
            frames = load_rgb_clip(self.samples[next_index][0], self.frames_per_clip)
            depth = load_depth(self.samples[next_index][1], self.depth_scale)
            video = self._transform_rgb(frames)
            depth, valid = self._transform_depth(depth)
        return video, depth, valid


def make_dataloader(
    csv_path: str,
    batch_size: int,
    frames_per_clip: int,
    resolution: int,
    depth_scale: float = 1.0,
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
    world_size: int = 1,
    rank: int = 0,
    training: bool = True,
    num_workers: int = 8,
    pin_memory: bool = True,
):
    dataset = DepthCsvDataset(
        csv_path=csv_path,
        frames_per_clip=frames_per_clip,
        resolution=resolution,
        depth_scale=depth_scale,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=training)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=training,
        persistent_workers=num_workers > 0,
    )
    return loader, sampler
