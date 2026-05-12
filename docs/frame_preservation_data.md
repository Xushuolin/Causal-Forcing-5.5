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


## Choosing Wan2.1-T2V-1.3B vs Wan2.1-T2V-14B

The trainable DiT backbone is selected in the YAML config via
`model_kwargs.model_name`. `WanDiffusionWrapper` resolves this as
`wan_models/{model_name}/`, so the default config uses:

```yaml
model_kwargs:
  model_name: Wan2.1-T2V-1.3B
  timestep_shift: 5.0
```

To train against the 14B DiT, download `wan_models/Wan2.1-T2V-14B` and switch to:

```yaml
model_kwargs:
  model_name: Wan2.1-T2V-14B
  timestep_shift: 5.0
history_context_dim: auto
```

`history_context_dim: auto` reads the DiT hidden width directly from the loaded
backbone, so it maps to 1536 for 1.3B and 5120 for 14B. The text encoder and VAE
wrappers currently load the shared Wan assets from `wan_models/Wan2.1-T2V-1.3B`,
which matches the existing repository convention and is compatible with either
T2V DiT size as long as the Wan model folders have been downloaded.

## Is LoRA attached to the DiT?

Yes. `FramePreservationDiffusion` calls `inject_lora_linear` on
`self.generator.model` when `use_lora: true`. The default target is `blocks.*`,
so Linear layers inside the Wan DiT transformer blocks are wrapped by LoRA while
the base Linear weights are frozen. The optimizer then trains the LoRA parameters
plus the LR/HR history encoder.

## LR/HR branch defaults

The default `configs/frame_preservation_pretrain.yaml` enables two history branches:

- LR branch: compression `[4, 4, 2]` for compact global structure.
- HR branch: compression `[2, 2, 1]` for higher-detail/high-frequency memory.

Both branches use the channel ramp `64 -> 128 -> 256 -> 512 -> 512` and then project
into `history_context_dim` tokens. For Wan2.1-T2V-1.3B this is `1536`; for larger
Wan variants, set it to the DiT hidden width used by that checkpoint.


## Raw-video LR/HR training path (current default)

The default config now uses `dataset_type: long_history_video_manifest`, so you do
not have to precompute a single latent stream before training. The dataset reads
raw videos and returns two resized/fps-sampled tensors:

- HR branch input: `480` video frames resized to `480x832`.
- LR branch input: `240` frames sampled over the **same temporal span** and resized to `120x208`.

During `train_one_step`, `FramePreservationDiffusion.encode_video_batch()` encodes
both tensors with the configured Wan VAE. The LR latent stream is used as the DiT
query/noisy latent stream, while the HR latent stream is fed only to the HR
history-compression branch. This follows the paper figure more closely than the
previous precomputed-latent path.

For long videos, `temporal_span_frames` controls how much source-video time the
window covers. `hr_num_video_frames` and `lr_num_video_frames` are then sampled
uniformly over that same span, so LR and HR cover the same time range but have
different frame rates. For example, `temporal_span_frames: 960`,
`hr_num_video_frames: 480`, and `lr_num_video_frames: 240` covers twice the time
span at half/double effective sampling rates without changing model input sizes.

Minimal raw-video manifest:

```jsonl
{"video_id":"clip_0001","video_path":"videos/clip_0001.mp4","prompt":"A horse runs in a field."}
{"video_id":"clip_0002","video_path":"videos/clip_0002.mp4","prompt":""}
```

Storyboard fields are still optional and use raw video frame indices for this
raw-video dataset. No manual object/mask labels are required.

## Wan2.2 note

Wan2.2-TI2V-5B exposes `Wan2.2_VAE.pth`, T5 weights, config JSON, and sharded
Diffusers-style safetensors on Hugging Face. The current patch makes the VAE and
text encoder folder/checkpoint configurable (`vae_model_name`, `vae_checkpoint`,
`text_encoder_model_name`) and keeps the DiT folder selected by
`model_kwargs.model_name`. A full Wan2.2 DiT loader may still require a dedicated
adapter because this repository's `WanModel.from_pretrained` was originally built
around the Wan2.1 folder/checkpoint layout.
