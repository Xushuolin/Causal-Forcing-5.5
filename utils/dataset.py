from utils.lmdb_ import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.io import read_video
import numpy as np
import torch
import lmdb
import json
import random
from pathlib import Path
from PIL import Image
import os


class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class ODERegressionLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }





class LatentLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "clean_latent": torch.tensor(latents, dtype=torch.float32)[-1]
        }


class ShardingLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.envs = []
        self.index = []

        for fname in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, fname)
            env = lmdb.open(path,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            self.envs.append(env)

        self.latents_shape = [None] * len(self.envs)
        for shard_id, env in enumerate(self.envs):
            self.latents_shape[shard_id] = get_array_shape_from_lmdb(env, 'latents')
            for local_i in range(self.latents_shape[shard_id][0]):
                self.index.append((shard_id, local_i))

            # print("shard_id ", shard_id, " local_i ", local_i)

        self.max_pair = max_pair

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
            Outputs:
                - prompts: List of Strings
                - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        shard_id, local_idx = self.index[idx]

        latents = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "latents", np.float16, local_idx,
            shape=self.latents_shape[shard_id][1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "prompts", str, local_idx
        )

        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }



class LongHistoryLatentDataset(Dataset):
    """Window sampler for long precomputed latent videos.

    The manifest is a JSONL file with one object per source video. Required
    fields:
      - latent_path: path to a .pt or .npy tensor, absolute or relative to the
        manifest directory. Tensor shape must be [F, C, H, W]. A tensor with
        [S, F, C, H, W] is also accepted and the final S entry is used.
    Optional fields:
      - prompt: global caption string.
      - storyboard: list of {start_frame, end_frame, text} or {start, end, text}
        entries. The prompt nearest to the sampled window center is returned.
      - video_id: any stable identifier.
    """

    def __init__(
        self,
        manifest_path: str,
        sequence_length: int = 120,
        frame_stride: int = 1,
        window_sampling: str = "random",
        pad_short_videos: bool = True,
    ):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.sequence_length = sequence_length
        self.frame_stride = frame_stride
        self.window_sampling = window_sampling
        self.pad_short_videos = pad_short_videos
        self.samples = []

        with open(self.manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        if not self.samples:
            raise ValueError(f"No samples found in {manifest_path}")

    def __len__(self):
        return len(self.samples)

    def _resolve_path(self, latent_path: str) -> Path:
        path = Path(latent_path)
        if path.is_absolute():
            return path
        return self.root / path

    def _load_latent(self, latent_path: str) -> torch.Tensor:
        path = self._resolve_path(latent_path)
        if path.suffix == ".pt":
            latent = torch.load(path, map_location="cpu")
            if isinstance(latent, dict):
                latent = latent.get("clean_latent", latent.get("latents"))
        elif path.suffix == ".npy":
            latent = torch.from_numpy(np.load(path))
        else:
            raise ValueError(f"Unsupported latent file type: {path}")
        if latent is None:
            raise ValueError(f"No latent tensor found in {path}")
        latent = torch.as_tensor(latent, dtype=torch.float32)
        if latent.ndim == 5:
            latent = latent[-1]
        if latent.ndim != 4:
            raise ValueError(f"Expected [F,C,H,W] latent tensor, got {tuple(latent.shape)} from {path}")
        return latent

    def _select_start(self, num_frames: int, idx: int) -> int:
        needed = (self.sequence_length - 1) * self.frame_stride + 1
        max_start = max(0, num_frames - needed)
        if self.window_sampling == "center":
            return max_start // 2
        if self.window_sampling == "sequential":
            return idx % (max_start + 1)
        return random.randint(0, max_start)

    def _select_prompt(self, item: dict, window_start: int, window_end: int) -> str:
        storyboard = item.get("storyboard") or item.get("captions")
        if not storyboard:
            return item.get("prompt", item.get("caption", ""))

        center = (window_start + window_end) / 2
        best = None
        best_distance = float("inf")
        for segment in storyboard:
            text = segment.get("text", segment.get("caption", ""))
            start = segment.get("start_frame", segment.get("start", 0))
            end = segment.get("end_frame", segment.get("end", start))
            midpoint = (start + end) / 2
            distance = abs(center - midpoint)
            if distance < best_distance:
                best = text
                best_distance = distance
        return best or item.get("prompt", item.get("caption", ""))

    def __getitem__(self, idx):
        item = self.samples[idx]
        latent = self._load_latent(item["latent_path"])
        num_frames = latent.shape[0]
        needed = (self.sequence_length - 1) * self.frame_stride + 1
        if num_frames < needed and not self.pad_short_videos:
            raise ValueError(
                f"Video {item.get('video_id', idx)} has {num_frames} latent frames, needs {needed}"
            )

        start = self._select_start(num_frames, idx)
        frame_ids = torch.arange(start, start + needed, self.frame_stride)
        frame_ids = frame_ids.clamp(max=num_frames - 1)
        window = latent[frame_ids]
        prompt = self._select_prompt(item, start, min(start + needed, num_frames))

        return {
            "prompts": prompt,
            "clean_latent": window,
            "idx": idx,
            "video_id": item.get("video_id", str(idx)),
            "window_start": start,
        }


class LongHistoryVideoDataset(Dataset):
    """Raw-video window sampler for on-the-fly LR/HR Wan VAE encoding.

    This follows the paper-style data path: keep raw high-resolution/fps video,
    construct a lower-resolution/fps copy for the DiT input branch, and encode
    both branches with VAE inside the training step.
    """

    def __init__(
        self,
        manifest_path: str,
        hr_num_frames: int = 480,
        lr_num_frames: int = 240,
        hr_size: tuple[int, int] = (480, 832),
        lr_size: tuple[int, int] = (120, 208),
        window_sampling: str = "random",
        pad_short_videos: bool = True,
        temporal_span_frames: int | None = None,
        lr_sample_strategy: str = "uniform",
    ):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.hr_num_frames = hr_num_frames
        self.lr_num_frames = lr_num_frames
        self.temporal_span_frames = temporal_span_frames or hr_num_frames
        self.lr_sample_strategy = lr_sample_strategy
        self.hr_size = tuple(hr_size)
        self.lr_size = tuple(lr_size)
        self.window_sampling = window_sampling
        self.pad_short_videos = pad_short_videos
        self.samples = []

        with open(self.manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        if not self.samples:
            raise ValueError(f"No samples found in {manifest_path}")

    def __len__(self):
        return len(self.samples)

    def _resolve_path(self, video_path: str) -> Path:
        path = Path(video_path)
        if path.is_absolute():
            return path
        return self.root / path

    def _select_start(self, num_frames: int, idx: int) -> int:
        max_start = max(0, num_frames - self.temporal_span_frames)
        if self.window_sampling == "center":
            return max_start // 2
        if self.window_sampling == "sequential":
            return idx % (max_start + 1)
        return random.randint(0, max_start)

    def _select_prompt(self, item: dict, window_start: int, window_end: int) -> str:
        storyboard = item.get("storyboard") or item.get("captions")
        if not storyboard:
            return item.get("prompt", item.get("caption", ""))

        center = (window_start + window_end) / 2
        best = None
        best_distance = float("inf")
        for segment in storyboard:
            text = segment.get("text", segment.get("caption", ""))
            start = segment.get("start_frame", segment.get("start", 0))
            end = segment.get("end_frame", segment.get("end", start))
            midpoint = (start + end) / 2
            distance = abs(center - midpoint)
            if distance < best_distance:
                best = text
                best_distance = distance
        return best or item.get("prompt", item.get("caption", ""))

    def _normalize_and_resize(self, frames: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        frames = frames.float().permute(0, 3, 1, 2) / 127.5 - 1.0
        frames = F.interpolate(frames, size=size, mode="bilinear", align_corners=False)
        return frames.permute(1, 0, 2, 3).contiguous()

    def __getitem__(self, idx):
        item = self.samples[idx]
        video_path = self._resolve_path(item["video_path"])
        frames, _, _ = read_video(str(video_path), pts_unit="sec", output_format="THWC")
        if frames.numel() == 0:
            raise ValueError(f"No frames decoded from {video_path}")
        num_frames = frames.shape[0]
        if num_frames < self.temporal_span_frames and not self.pad_short_videos:
            raise ValueError(
                f"Video {item.get('video_id', idx)} has {num_frames} frames, needs {self.temporal_span_frames}"
            )

        start = self._select_start(num_frames, idx)
        end = start + self.temporal_span_frames
        hr_ids = torch.linspace(start, end - 1, self.hr_num_frames).round().long().clamp(max=num_frames - 1)
        hr_frames = frames[hr_ids]
        if self.lr_sample_strategy == "from_hr":
            lr_ids = torch.linspace(0, self.hr_num_frames - 1, self.lr_num_frames).round().long()
            lr_frames = hr_frames[lr_ids]
        else:
            lr_ids = torch.linspace(start, end - 1, self.lr_num_frames).round().long().clamp(max=num_frames - 1)
            lr_frames = frames[lr_ids]
        prompt = self._select_prompt(item, start, min(end, num_frames))

        return {
            "prompts": prompt,
            "hr_frames": self._normalize_and_resize(hr_frames, self.hr_size),
            "lr_frames": self._normalize_and_resize(lr_frames, self.lr_size),
            "idx": idx,
            "video_id": item.get("video_id", video_path.stem),
            "window_start": start,
        }


class TextImagePairDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        eval_first_n=-1,
        pad_to_multiple_of=None
    ):
        """
        Args:
            data_dir (str): Path to the directory containing:
                - target_crop_info_*.json (metadata file)
                - */ (subdirectory containing images with matching aspect ratio)
            transform (callable, optional): Optional transform to be applied on the image
        """
        self.transform = transform
        data_dir = Path(data_dir)

        # Find the metadata JSON file
        metadata_files = list(data_dir.glob('target_crop_info_*.json'))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {data_dir}")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files found in {data_dir}")

        metadata_path = metadata_files[0]
        # Extract aspect ratio from metadata filename (e.g. target_crop_info_26-15.json -> 26-15)
        aspect_ratio = metadata_path.stem.split('_')[-1]

        # Use aspect ratio subfolder for images
        self.image_dir = data_dir / aspect_ratio
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        eval_first_n = eval_first_n if eval_first_n != -1 else len(self.metadata)
        self.metadata = self.metadata[:eval_first_n]

        # Verify all images exist
        for item in self.metadata:
            image_path = self.image_dir / item['file_name']
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        self.dummy_prompt = "DUMMY PROMPT"
        self.pre_pad_len = len(self.metadata)
        if pad_to_multiple_of is not None and len(self.metadata) % pad_to_multiple_of != 0:
            # Duplicate the last entry
            self.metadata += [self.metadata[-1]] * (
                pad_to_multiple_of - len(self.metadata) % pad_to_multiple_of
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary containing:
                - image: PIL Image
                - caption: str
                - target_bbox: list of int [x1, y1, x2, y2]
                - target_ratio: str
                - type: str
                - origin_size: tuple of int (width, height)
        """
        item = self.metadata[idx]

        # Load image
        image_path = self.image_dir / item['file_name']
        image = Image.open(image_path).convert('RGB')

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'prompts': item['caption'],
            'target_bbox': item['target_crop']['target_bbox'],
            'target_ratio': item['target_crop']['target_ratio'],
            'type': item['type'],
            'origin_size': (item['origin_width'], item['origin_height']),
            'idx': idx
        }



def cycle(dl):
    while True:
        for data in dl:
            yield data
