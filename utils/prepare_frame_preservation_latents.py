import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.io import read_video
from tqdm import tqdm

from utils.wan_wrapper import WanVAEWrapper


def load_source_manifest(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def sample_video_frames(video_path: Path, height: int, width: int, target_fps: float = 0.0):
    frames, _, info = read_video(str(video_path), pts_unit="sec", output_format="THWC")
    if frames.numel() == 0:
        raise ValueError(f"No frames decoded from {video_path}")

    source_fps = info.get("video_fps", 0.0) or 0.0
    if target_fps > 0 and source_fps > target_fps:
        step = max(1, round(source_fps / target_fps))
        frames = frames[::step]

    frames = frames.float().permute(0, 3, 1, 2) / 127.5 - 1.0
    frames = F.interpolate(frames, size=(height, width), mode="bilinear", align_corners=False)
    return frames


def encode_video_to_latent(vae: WanVAEWrapper, frames: torch.Tensor, device: torch.device, dtype: torch.dtype):
    # frames: [T, C, H, W] in [-1, 1]
    pixel = frames.permute(1, 0, 2, 3).unsqueeze(0).to(device=device, dtype=dtype)
    with torch.no_grad():
        latent = vae.encode_to_latent(pixel)[0].cpu()
    return latent


def main():
    parser = argparse.ArgumentParser(description="Prepare Wan VAE latents for frame-preservation pretraining.")
    parser.add_argument("--source_manifest", type=Path, required=True, help="JSONL with video_path plus optional prompt/storyboard.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for latent .pt files.")
    parser.add_argument("--output_manifest", type=Path, required=True, help="JSONL manifest consumed by LongHistoryLatentDataset.")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--target_fps", type=float, default=0.0, help="Optional FPS downsampling before VAE encoding; 0 keeps source FPS.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_root = args.source_manifest.parent
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    vae = WanVAEWrapper().to(device=device, dtype=dtype).eval()

    with open(args.output_manifest, "w", encoding="utf-8") as out_f:
        for idx, item in enumerate(tqdm(list(load_source_manifest(args.source_manifest)))):
            video_path = resolve_path(manifest_root, item["video_path"])
            video_id = item.get("video_id", video_path.stem)
            frames = sample_video_frames(video_path, args.height, args.width, args.target_fps)
            latent = encode_video_to_latent(vae, frames, device=device, dtype=dtype)

            latent_name = f"{video_id}.pt"
            latent_path = args.output_dir / latent_name
            torch.save({"clean_latent": latent}, latent_path)

            record = {
                "video_id": video_id,
                "latent_path": str(latent_path.relative_to(args.output_manifest.parent)),
            }
            if "prompt" in item:
                record["prompt"] = item["prompt"]
            if "caption" in item:
                record["caption"] = item["caption"]
            if "storyboard" in item:
                record["storyboard"] = item["storyboard"]
            if "captions" in item:
                record["captions"] = item["captions"]
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
