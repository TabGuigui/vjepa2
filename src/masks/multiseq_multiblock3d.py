# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from logging import getLogger
from multiprocessing import Value

import torch
import torch.nn.functional as F

_GLOBAL_SEED = 0
logger = getLogger()


class MaskCollator(object):

    def __init__(
        self,
        cfgs_mask,
        dataset_fpcs,
        crop_size=(224, 224),
        patch_size=(16, 16),
        tubelet_size=2,
    ):
        super(MaskCollator, self).__init__()

        self.mask_generators = dict()
        for fpc in dataset_fpcs:
            self.mask_generators[fpc] = []
            for m in cfgs_mask:
                mask_generator = _MaskGenerator(
                    crop_size=crop_size,
                    num_frames=fpc,
                    spatial_patch_size=patch_size,
                    temporal_patch_size=tubelet_size,
                    spatial_pred_mask_scale=m.get("spatial_scale"),
                    temporal_pred_mask_scale=m.get("temporal_scale"),
                    aspect_ratio=m.get("aspect_ratio"),
                    npred=m.get("num_blocks"),
                    max_context_frames_ratio=m.get("max_temporal_keep", 1.0),
                    max_keep=m.get("max_keep", None),
                    full_complement=m.get("full_complement", False),
                    pred_full_complement=m.get("pred_full_complement", False),
                    inv_block=m.get("inv_block", False),
                    dynamic_bias=m.get("dynamic_bias", None),
                    temporal_frame_mask=m.get("temporal_frame_mask", None),
                )
                self.mask_generators[fpc].append(mask_generator)

    def step(self):
        for fpc in self.mask_generators:
            for mask_generator in self.mask_generators[fpc]:
                mask_generator.step()

    def __call__(self, batch):

        # Batch: [buffer, label, clip_indices] for video
        # or [buffer, label] for images
        filtered_batches = {fpc: [] for fpc in self.mask_generators}
        for sample in batch:
            # Check if sample is from video dataset (has clip_indices) or image dataset
            if len(sample) >= 3 and isinstance(sample[-1], (list, tuple)):
                # Video sample: sample[-1] is clip_indices, sample[-1][-1] contains frame indices
                try:
                    fpc = len(sample[-1][-1])
                except (TypeError, IndexError):
                    # Fallback: assume single frame if structure is unexpected
                    fpc = 1
            else:
                # Image sample: single frame
                fpc = 1
            if fpc in filtered_batches:
                filtered_batches[fpc] += [sample]

        fpc_collations = []
        for fpc in filtered_batches:
            fpc_batch = filtered_batches[fpc]
            batch_size = len(fpc_batch)
            if batch_size == 0:
                continue
            collated_batch = torch.utils.data.default_collate(fpc_batch)
            collated_masks_pred, collated_masks_enc = [], []
            videos = collated_batch[0][0]
            for i, mask_generator in enumerate(self.mask_generators[fpc]):
                masks_enc, masks_pred = mask_generator(batch_size, videos=videos)
                collated_masks_enc.append(masks_enc)
                collated_masks_pred.append(masks_pred)
            fpc_collations += [
                (collated_batch, collated_masks_enc, collated_masks_pred)
            ]

        return fpc_collations


class _MaskGenerator(object):

    def __init__(
        self,
        crop_size=(224, 224),
        num_frames=16,
        spatial_patch_size=(16, 16),
        temporal_patch_size=2,
        spatial_pred_mask_scale=(0.2, 0.8),
        temporal_pred_mask_scale=(1.0, 1.0),
        aspect_ratio=(0.3, 3.0),
        npred=1,
        max_context_frames_ratio=1.0,
        max_keep=None,
        inv_block=False,
        full_complement=False,
        pred_full_complement=False,
        dynamic_bias=None,
        temporal_frame_mask=None,
    ):
        super(_MaskGenerator, self).__init__()
        if isinstance(crop_size, list):
            crop_size = tuple(crop_size)
        elif not isinstance(crop_size, tuple):
            crop_size = (crop_size,) * 2
        if isinstance(spatial_patch_size, list):
            spatial_patch_size = tuple(spatial_patch_size)
        elif not isinstance(spatial_patch_size, tuple):
            spatial_patch_size = (spatial_patch_size,) * 2
        self.crop_size = crop_size
        self.height, self.width = [
            crop_size[i] // spatial_patch_size[i] for i in (0, 1)
        ]
        self.duration = num_frames // temporal_patch_size
        self.full_complement = full_complement
        self.pred_full_complement = pred_full_complement

        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.aspect_ratio = aspect_ratio
        self.spatial_pred_mask_scale = spatial_pred_mask_scale
        self.temporal_pred_mask_scale = temporal_pred_mask_scale
        self.npred = npred
        self.max_context_duration = max(
            1, int(self.duration * max_context_frames_ratio)
        )  # maximum number of time-steps (frames) spanned by context mask
        self.max_keep = max_keep  # maximum number of patches to keep in context
        self._itr_counter = Value("i", -1)  # collator is shared across worker processes
        self.inv_block = inv_block
        dynamic_bias = dynamic_bias or {}
        self.dynamic_bias_enabled = dynamic_bias.get("enabled", False)
        self.dynamic_bias_alpha = dynamic_bias.get("alpha", 0.5)
        self.dynamic_bias_temperature = dynamic_bias.get("temperature", 0.2)
        self.dynamic_bias_eps = dynamic_bias.get("eps", 1.0e-6)

        temporal_frame_mask = temporal_frame_mask or {}
        self.temporal_frame_mask_enabled = temporal_frame_mask.get("enabled", False)
        self.temporal_frame_mask_num_slices = int(temporal_frame_mask.get("num_slices", 1))

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(
        self, generator, temporal_scale, spatial_scale, aspect_ratio_scale
    ):
        # -- Sample temporal block mask scale
        _rand = torch.rand(1, generator=generator).item()
        min_t, max_t = temporal_scale
        temporal_mask_scale = min_t + _rand * (max_t - min_t)
        t = max(1, int(self.duration * temporal_mask_scale))

        # -- Sample spatial block mask scale
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = spatial_scale
        spatial_mask_scale = min_s + _rand * (max_s - min_s)
        spatial_num_keep = int(self.height * self.width * spatial_mask_scale)

        # -- Sample block aspect-ratio
        _rand = torch.rand(1, generator=generator).item()
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(spatial_num_keep * aspect_ratio)))
        w = int(round(math.sqrt(spatial_num_keep / aspect_ratio)))
        h = min(h, self.height)
        w = min(w, self.width)

        return (t, h, w)

    def _compute_dynamic_heatmaps(self, videos):
        if (not self.dynamic_bias_enabled) or videos is None:
            return None
        if videos.ndim != 5:
            return None

        videos = videos.detach().float()
        if videos.size(2) < 2:
            return videos.new_zeros((videos.size(0), self.duration, self.height, self.width))

        motion = (videos[:, :, 1:] - videos[:, :, :-1]).abs().mean(dim=1)
        motion = motion.unsqueeze(1)
        motion = F.interpolate(
            motion,
            size=(self.duration, self.height, self.width),
            mode="trilinear",
            align_corners=False,
        ).squeeze(1)

        flat = motion.flatten(1)
        min_v = flat.min(dim=1).values.view(-1, 1, 1, 1)
        max_v = flat.max(dim=1).values.view(-1, 1, 1, 1)
        motion = (motion - min_v) / (max_v - min_v + self.dynamic_bias_eps)
        return motion

    def _sample_dynamic_anchor(self, b_size, dynamic_heatmap):
        t, h, w = b_size
        if dynamic_heatmap is None:
            return None

        d_span = self.duration - t + 1
        h_span = self.height - h + 1
        w_span = self.width - w + 1
        if d_span <= 0 or h_span <= 0 or w_span <= 0:
            return None

        block_scores = F.avg_pool3d(
            dynamic_heatmap.view(1, 1, self.duration, self.height, self.width),
            kernel_size=(t, h, w),
            stride=1,
        ).flatten()

        if not torch.isfinite(block_scores).all():
            return None
        if block_scores.max() <= self.dynamic_bias_eps:
            return None

        dynamic_prob = torch.softmax(
            block_scores / max(self.dynamic_bias_temperature, self.dynamic_bias_eps),
            dim=0,
        )
        uniform_prob = torch.full_like(dynamic_prob, 1.0 / dynamic_prob.numel())
        probs = (
            self.dynamic_bias_alpha * dynamic_prob
            + (1.0 - self.dynamic_bias_alpha) * uniform_prob
        )
        anchor = torch.multinomial(probs, 1).item()

        start = anchor // (h_span * w_span)
        rem = anchor % (h_span * w_span)
        top = rem // w_span
        left = rem % w_span
        return start, top, left

    def _sample_temporal_frame_mask(self):
        num_slices = max(1, min(self.temporal_frame_mask_num_slices, self.duration))
        time_ids = torch.randperm(self.duration)[:num_slices]
        mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
        mask[time_ids, :, :] = 0
        if self.max_context_duration < self.duration:
            mask[self.max_context_duration :, :, :] = 0
        return mask

    def _sample_block_mask(self, b_size, dynamic_heatmap=None):
        t, h, w = b_size
        anchor = self._sample_dynamic_anchor(b_size, dynamic_heatmap)
        if anchor is None:
            top = torch.randint(0, self.height - h + 1, (1,))
            left = torch.randint(0, self.width - w + 1, (1,))
            start = torch.randint(0, self.duration - t + 1, (1,))
        else:
            start, top, left = anchor

        mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
        mask[start : start + t, top : top + h, left : left + w] = 0

        # Context mask will only span the first X frames
        # (X=self.max_context_frames)
        if self.max_context_duration < self.duration:
            mask[self.max_context_duration :, :, :] = 0

        # --
        return mask

    def __call__(self, batch_size, videos=None):
        """
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample pred block size using seed
        # 2. sample several pred block locations for each image (w/o seed)
        # 3. return pred masks and complement (enc mask)
        """
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            temporal_scale=self.temporal_pred_mask_scale,
            spatial_scale=self.spatial_pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio,
        )

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_enc = min_keep_pred = self.duration * self.height * self.width
        dynamic_heatmaps = self._compute_dynamic_heatmaps(videos)
        for batch_idx in range(batch_size):

            empty_context = True
            while empty_context:

                mask_e = torch.ones(
                    (self.duration, self.height, self.width), dtype=torch.int32
                )
                if self.temporal_frame_mask_enabled:
                    mask_e *= self._sample_temporal_frame_mask()
                else:
                    for _ in range(self.npred):
                        dynamic_heatmap = None if dynamic_heatmaps is None else dynamic_heatmaps[batch_idx]
                        mask_e *= self._sample_block_mask(p_size, dynamic_heatmap=dynamic_heatmap)
                mask_e = mask_e.flatten()

                mask_p = torch.argwhere(mask_e == 0).squeeze()
                mask_e = torch.nonzero(mask_e).squeeze()

                empty_context = len(mask_e) == 0
                if not empty_context:
                    min_keep_pred = min(min_keep_pred, len(mask_p))
                    min_keep_enc = min(min_keep_enc, len(mask_e))
                    collated_masks_pred.append(mask_p)
                    collated_masks_enc.append(mask_e)

        if self.max_keep is not None:
            min_keep_enc = min(min_keep_enc, self.max_keep)

        collated_masks_enc = [cm[:min_keep_enc] for cm in collated_masks_enc]
        collated_masks_pred = [cm[:min_keep_pred] for cm in collated_masks_pred]
        if self.full_complement:  # predictor mask is just complement of encoder mask
            collated_masks_pred = [
                torch.tensor(
                    sorted(
                        list(
                            set(range(int(self.duration * self.height * self.width)))
                            - set(cm.tolist())
                        )
                    ),
                    dtype=cm.dtype,
                )
                for cm in collated_masks_enc
            ]
        elif self.pred_full_complement:
            collated_masks_enc = [
                torch.tensor(
                    sorted(
                        list(
                            set(range(int(self.duration * self.height * self.width)))
                            - set(cm.tolist())
                        )
                    ),
                    dtype=cm.dtype,
                )
                for cm in collated_masks_pred
            ]

        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)

        if self.inv_block:
            return collated_masks_pred, collated_masks_enc  # predict context from block
        else:
            return collated_masks_enc, collated_masks_pred
