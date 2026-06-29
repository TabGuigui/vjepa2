import csv
import logging
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)


class DriveActionVideoDataset(Dataset):
    """Navsim fpc8 clips with xy actions stored in the CSV row.

    Expected row format:
      video_path ax_3_4 ay_3_4 ax_4_5 ay_4_5 ax_5_6 ay_5_6 ax_6_7 ay_6_7
    """

    def __init__(
        self,
        csv_path,
        frames_per_clip=8,
        context_frames=4,
        action_dim=2,
        action_steps=4,
        transform=None,
    ):
        self.csv_path = Path(csv_path)
        self.frames_per_clip = int(frames_per_clip)
        self.context_frames = int(context_frames)
        self.action_dim = int(action_dim)
        self.action_steps = int(action_steps)
        self.transform = transform
        self.samples = self._read_samples()
        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found in {csv_path}")

    def _read_samples(self):
        expected_action_values = self.action_dim * self.action_steps
        samples = []
        with self.csv_path.open("r", newline="") as handle:
            reader = csv.reader(handle, delimiter=" ")
            for line_idx, row in enumerate(reader, start=1):
                row = [item for item in row if item != ""]
                if len(row) == 0 or row[0].lower() in {"path", "video", "video_path", "fname", "filename"}:
                    continue
                if len(row) < 1 + expected_action_values:
                    logger.info(
                        "%s:%d expected at least %d columns, got %d",
                        self.csv_path,
                        line_idx,
                        1 + expected_action_values,
                        len(row),
                    )
                    continue
                video_path = row[0]
                try:
                    actions = np.asarray(row[1 : 1 + expected_action_values], dtype=np.float32)
                except ValueError:
                    continue
                actions = actions.reshape(self.action_steps, self.action_dim)
                samples.append((video_path, actions))
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_video(self, video_path):
        vr = VideoReader(video_path, num_threads=-1, ctx=cpu(0))
        if len(vr) < self.frames_per_clip:
            raise ValueError(f"Video is too short: {video_path}, need {self.frames_per_clip}, got {len(vr)}")
        indices = np.linspace(0, len(vr) - 1, self.frames_per_clip).round().astype(np.int64)
        return vr.get_batch(indices).asnumpy()

    def __getitem__(self, index):
        while True:
            video_path, actions = self.samples[index]
            try:
                clip = self._load_video(video_path)
                if self.transform is not None:
                    clip = self.transform(clip)
                return clip, torch.from_numpy(actions.copy())
            except Exception as exc:
                logger.info("Encountered exception when loading %s: %s", video_path, exc)
                index = np.random.randint(len(self.samples))


def make_dataloader(
    csv_path,
    batch_size,
    frames_per_clip,
    context_frames,
    action_dim,
    action_steps,
    transform,
    world_size=1,
    rank=0,
    training=True,
    num_workers=8,
    pin_memory=True,
):
    dataset = DriveActionVideoDataset(
        csv_path=csv_path,
        frames_per_clip=frames_per_clip,
        context_frames=context_frames,
        action_dim=action_dim,
        action_steps=action_steps,
        transform=transform,
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
