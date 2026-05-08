# Frame-preservation pretraining data format

This stage trains `FramePreservationDiffusion` with `dataset_type: long_history_manifest`.
The trainer expects `data_path` to point to a JSONL manifest. Each line describes one
source video and points to precomputed Wan VAE latents.

## Required latent files

Each `latent_path` must be either:

- `.pt` saved with a tensor or a dict containing `clean_latent` or `latents`.
- `.npy` saved with a NumPy array.

Accepted tensor shapes:

- `[F, C, H, W]` for clean latents.
- `[S, F, C, H, W]` for ODE/diffusion trajectories; the dataset uses the final
  entry `[-1]` as the clean latent video.

For the default Wan2.1 480×832 setup, each latent frame should be `[16, 60, 104]`.
Wan VAE is temporally compressed, so `F` means **latent frames**, not raw video
frames. In Wan2.1, 81 decoded video frames correspond to 21 latent frames; plan
your raw videos accordingly if you want 120 latent frames. The default config
samples windows of 120 latent frames, so a source video should ideally contain
at least 120 latent frames after any `history_frame_stride`. Shorter videos can
be padded by repeating the last frame when `pad_short_videos: true`.

## Manifest schema

Minimal example:

```jsonl
{"video_id":"clip_0001","latent_path":"latents/clip_0001.pt","prompt":"A woman walks through a sunny kitchen."}
{"video_id":"clip_0002","latent_path":"latents/clip_0002.npy","prompt":"A dog runs across a beach at sunset."}
```

Storyboard example:

```jsonl
{"video_id":"clip_0003","latent_path":"latents/clip_0003.pt","storyboard":[{"start_frame":0,"end_frame":39,"text":"A grandmother enters the kitchen in the morning."},{"start_frame":40,"end_frame":79,"text":"She tidies the bookroom and looks at old photos."},{"start_frame":80,"end_frame":119,"text":"She sits near a cat and starts knitting."}]}
```

Fields:

- `video_id` (optional): stable identifier used for debugging/logging.
- `latent_path` (required): path to the latent file, absolute or relative to the
  manifest directory.
- `prompt` or `caption` (optional): global caption used when no storyboard is present.
- `storyboard` or `captions` (optional): list of timestamped text segments. Each
  segment can use `start_frame`/`end_frame` or `start`/`end`; the dataset returns
  the text nearest to the sampled window center.

## Default sampling behavior

`LongHistoryLatentDataset` loads one source video per item, samples a temporal window,
and returns:

- `clean_latent`: `[history_sequence_length, C, H, W]`
- `prompts`: selected global prompt or nearest storyboard text
- `idx`, `video_id`, `window_start`: metadata for logging/debugging

The model then samples random query frames Ω inside that window, keeps Ω clean,
corrupts all non-query history frames with noise from the configured range, and
trains the DiT LoRA plus history encoder to reconstruct the selected frames.


## Do I need raw videos or annotations?

For this pretraining stage, you do **not** need manually annotated frame labels.
The supervision comes from the video itself: the model randomly chooses query
frames Ω and learns to reconstruct them from the compressed history context.

You can start from a folder of raw videos, but the training code added here reads
precomputed latents for efficiency. The recommended preparation flow is:

1. Collect raw videos and a text prompt per video if available. If you do not
   have captions, use an empty string or generate captions/storyboards with a VLM.
2. Decode each video into frames at your target FPS and resize/crop to 480×832.
3. Encode frames with the Wan VAE into latents of shape `[F, 16, 60, 104]`.
4. Save each latent video as `.pt` or `.npy` and list it in the JSONL manifest.

A helper script is included if you want to start from raw videos:

```bash
python utils/prepare_frame_preservation_latents.py \
  --source_manifest dataset/raw_videos.jsonl \
  --output_dir dataset/frame_preservation_latents \
  --output_manifest dataset/frame_preservation_manifest.jsonl \
  --height 480 --width 832
```

The raw-video source manifest should use `video_path` plus optional `prompt`,
`caption`, `storyboard`, or `captions` fields. The script writes `.pt` files with
`clean_latent` tensors and creates the training manifest consumed by
`LongHistoryLatentDataset`.

Captions are useful because the Wan denoiser still expects text conditioning, but
frame-preservation pretraining does not require object boxes, segmentation masks,
identity labels, or per-frame human annotations.

## LR/HR branch defaults

The default `configs/frame_preservation_pretrain.yaml` enables two history branches:

- LR branch: compression `[4, 4, 2]` for compact global structure.
- HR branch: compression `[2, 2, 1]` for higher-detail/high-frequency memory.

Both branches use the channel ramp `64 -> 128 -> 256 -> 512 -> 512` and then project
into `history_context_dim` tokens. For Wan2.1-T2V-1.3B this is `1536`; for larger
Wan variants, set it to the DiT hidden width used by that checkpoint.
