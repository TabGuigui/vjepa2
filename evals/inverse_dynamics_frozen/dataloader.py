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
    return tokens[0].lower() in {"path", "video", "video_path", "fname", "filename"}


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


def load_video(path: str, frames_per_clip: int) -> np.ndarray:
    if VideoReader is not None:
        try:
            return _load_video_decord(path, frames_per_clip)
        except Exception:
            pass
    return _load_video_cv2(path, frames_per_clip)


class InverseDynamicsCsvDataset(Dataset):
    """CSV dataset for inverse-dynamics probing.

    Expected default row format:
      /path/to/clip.mp4 ax0 ay0 ax1 ay1 ...
    """

    def __init__(
        self,
        csv_path: str,
        frames_per_clip: int,
        resolution: int,
        action_dim: int,
        action_start_col: int = 1,
        source_frames_per_clip: int = None,
        input_frame_start: int = 0,
        normalize: Tuple[Sequence[float], Sequence[float]] = DEFAULT_NORMALIZATION,
        skip_broken: bool = True,
    ):
        self.csv_path = Path(csv_path)
        self.frames_per_clip = frames_per_clip
        self.resolution = resolution
        self.action_dim = action_dim
        self.action_start_col = action_start_col
        self.source_frames_per_clip = source_frames_per_clip or frames_per_clip
        self.input_frame_start = input_frame_start
        if self.input_frame_start < 0:
            raise ValueError("input_frame_start must be non-negative")
        if self.input_frame_start + self.frames_per_clip > self.source_frames_per_clip:
            raise ValueError(
                f"input window [{self.input_frame_start}, {self.input_frame_start + self.frames_per_clip}) "
                f"exceeds source_frames_per_clip={self.source_frames_per_clip}"
            )
        self.normalize = normalize
        self.skip_broken = skip_broken
        self.samples = self._read_samples()
        if len(self.samples) == 0:
            raise RuntimeError(f"No inverse-dynamics samples found in {csv_path}")

    def _read_samples(self):
        samples = []
        with self.csv_path.open("r") as handle:
            for line in handle:
                tokens = _split_line(line)
                if not tokens or _is_header(tokens):
                    continue
                video_path = tokens[0]
                action_tokens = tokens[self.action_start_col : self.action_start_col + self.action_dim]
                if len(action_tokens) != self.action_dim:
                    continue
                try:
                    action = [float(v) for v in action_tokens]
                except ValueError:
                    continue
                samples.append((video_path, torch.tensor(action, dtype=torch.float32)))
        return samples

    def __len__(self):
        return len(self.samples)

    def _transform(self, frames: np.ndarray) -> torch.Tensor:
        frames = frames[self.input_frame_start : self.input_frame_start + self.frames_per_clip]
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

    def __getitem__(self, index: int):
        video_path, action = self.samples[index]
        try:
            frames = load_video(video_path, self.source_frames_per_clip)
            video = self._transform(frames)
        except Exception:
            if not self.skip_broken:
                raise
            next_index = (index + 1) % len(self.samples)
            frames = load_video(self.samples[next_index][0], self.source_frames_per_clip)
            video = self._transform(frames)
            action = self.samples[next_index][1]
        return video, action


def make_dataloader(
    csv_path: str,
    batch_size: int,
    frames_per_clip: int,
    resolution: int,
    action_dim: int,
    action_start_col: int = 1,
    source_frames_per_clip: int = None,
    input_frame_start: int = 0,
    world_size: int = 1,
    rank: int = 0,
    training: bool = True,
    num_workers: int = 8,
    pin_memory: bool = True,
):
    dataset = InverseDynamicsCsvDataset(
        csv_path=csv_path,
        frames_per_clip=frames_per_clip,
        resolution=resolution,
        action_dim=action_dim,
        action_start_col=action_start_col,
        source_frames_per_clip=source_frames_per_clip,
        input_frame_start=input_frame_start,
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
