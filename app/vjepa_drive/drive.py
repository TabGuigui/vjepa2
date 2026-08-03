import csv
import logging
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from decord import VideoReader, cpu

logger = logging.getLogger(__name__)


def init_data(
    data_path,
    batch_size,
    frames_per_clip=8,
    tubelet_size=2,
    context_frames=4,
    action_dim=2,
    transform=None,
    rank=0,
    world_size=1,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    collator=None,
):
    dataset = DriveActionVideoDataset(
        data_path=data_path,
        frames_per_clip=frames_per_clip,
        tubelet_size=tubelet_size,
        context_frames=context_frames,
        action_dim=action_dim,
        transform=transform,
    )

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )

    logger.info("DriveActionVideoDataset data loader created")
    return data_loader, dist_sampler


class DriveActionVideoDataset(torch.utils.data.Dataset):
    """Video dataset with future xy actions stored directly in csv.

    CSV format:
        video_path ax0 ay0 ax1 ay1 ax2 ay2 ax3 ay3

    For fpc=8 and context_frames=4, this returns actions with shape [4, 2]:
        frame3->4, frame4->5, frame5->6, frame6->7.
    """

    def __init__(
        self,
        data_path,
        frames_per_clip=8,
        tubelet_size=2,
        context_frames=4,
        action_dim=2,
        transform=None,
    ):
        self.data_path = str(data_path)
        self.frames_per_clip = int(frames_per_clip)
        self.tubelet_size = int(tubelet_size)
        self.context_frames = int(context_frames)
        self.action_dim = int(action_dim)
        self.transform = transform

        if self.frames_per_clip % self.tubelet_size != 0:
            raise ValueError(
                f"frames_per_clip must be divisible by tubelet_size, got "
                f"{self.frames_per_clip} and {self.tubelet_size}"
            )
        if not (1 <= self.context_frames < self.frames_per_clip):
            raise ValueError(
                f"context_frames must be in [1, {self.frames_per_clip - 1}], got {self.context_frames}"
            )

        self.action_steps = self.frames_per_clip - self.context_frames
        self.samples = self._read_csv(Path(data_path))
        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {data_path}")

    def _read_csv(self, data_path):
        expected_values = self.action_steps * self.action_dim
        samples = []
        with data_path.open("r", newline="") as f:
            reader = csv.reader(f, delimiter=" ")
            for line_idx, row in enumerate(reader, start=1):
                row = [item for item in row if item != ""]
                if len(row) == 0:
                    continue
                if len(row) != 1 + expected_values:
                    raise ValueError(
                        f"{data_path}:{line_idx} expected {1 + expected_values} columns "
                        f"(path + {expected_values} action values), got {len(row)}"
                    )
                video_path = row[0]
                actions = np.asarray([float(v) for v in row[1:]], dtype=np.float32)
                actions = actions.reshape(self.action_steps, self.action_dim)
                samples.append((video_path, actions))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        loaded_sample = None
        while loaded_sample is None:
            video_path, actions = self.samples[index]
            try:
                buffer = self.loadvideo_decord(video_path)
                if self.transform is not None:
                    buffer = self.transform(buffer)
                loaded_sample = (
                    buffer,
                    torch.from_numpy(actions.copy()),
                    video_path,
                )
            except Exception as exc:
                logger.info(f"Encountered exception when loading drive video {video_path}: {exc}")
                index = np.random.randint(self.__len__())
        return loaded_sample

    def loadvideo_decord(self, video_path):
        vr = VideoReader(video_path, num_threads=-1, ctx=cpu(0))
        if len(vr) < self.frames_per_clip:
            raise ValueError(
                f"Video is too short: {video_path}, need {self.frames_per_clip}, got {len(vr)}"
            )

        indices = np.linspace(0, len(vr) - 1, self.frames_per_clip).round().astype(np.int64)
        buffer = vr.get_batch(indices).asnumpy()
        return buffer
