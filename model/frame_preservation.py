from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from model.base import BaseModel
from utils.lora import inject_lora_linear
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class CompressionBranch(nn.Module):
    """3D-conv history compression branch used by the LR/HR encoder."""

    def __init__(
        self,
        in_channels: int,
        context_dim: int,
        hidden_channels: Tuple[int, ...],
        compression_rate: Tuple[int, int, int],
        num_attention_heads: int,
    ):
        super().__init__()
        if len(compression_rate) != 3:
            raise ValueError("compression_rate must be (height, width, time)")

        h_rate, w_rate, t_rate = compression_rate
        remaining = [max(1, t_rate), max(1, h_rate), max(1, w_rate)]
        layers = []
        current_channels = in_channels

        for next_channels in hidden_channels:
            stride_t = 2 if remaining[0] > 1 else 1
            stride_h = 2 if remaining[1] > 1 else 1
            stride_w = 2 if remaining[2] > 1 else 1
            remaining[0] = max(1, remaining[0] // stride_t)
            remaining[1] = max(1, remaining[1] // stride_h)
            remaining[2] = max(1, remaining[2] // stride_w)
            layers.extend([
                nn.Conv3d(
                    current_channels,
                    next_channels,
                    kernel_size=3,
                    stride=(stride_t, stride_h, stride_w),
                    padding=1,
                ),
                nn.SiLU(),
            ])
            current_channels = next_channels

        self.conv = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(current_channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=current_channels,
            num_heads=num_attention_heads,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(current_channels),
            nn.Linear(current_channels, current_channels * 4),
            nn.SiLU(),
            nn.Linear(current_channels * 4, current_channels),
        )
        self.proj = nn.Linear(current_channels, context_dim)

    def forward(self, history_latents: torch.Tensor) -> torch.Tensor:
        # [B, F, C, H, W] -> [B, C, F, H, W]
        x = history_latents.permute(0, 2, 1, 3, 4)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        x_norm = self.norm(x)
        x = x + self.attn(x_norm, x_norm, x_norm, need_weights=False)[0]
        x = x + self.ffn(x)
        return self.proj(x)


class LightweightHistoryEncoder(nn.Module):
    """LR/HR lightweight video-history encoder for frame-preservation pretraining.

    The paper's practical encoder uses two complementary branches: an LR branch
    for cheap global structure and an HR branch that preserves fine details before
    projecting to the diffusion transformer's inner/context channel width.
    """

    def __init__(
        self,
        in_channels: int = 16,
        context_dim: int = 1536,
        hidden_channels: Tuple[int, ...] = (64, 128, 256, 512, 512),
        compression_rate: Tuple[int, int, int] = (4, 4, 2),
        num_attention_heads: int = 8,
        use_lr_branch: bool = True,
        use_hr_branch: bool = True,
        lr_hidden_channels: Tuple[int, ...] | None = None,
        hr_hidden_channels: Tuple[int, ...] | None = None,
        lr_compression_rate: Tuple[int, int, int] | None = None,
        hr_compression_rate: Tuple[int, int, int] | None = None,
    ):
        super().__init__()
        if not use_lr_branch and not use_hr_branch:
            raise ValueError("At least one of use_lr_branch/use_hr_branch must be enabled")

        lr_hidden_channels = tuple(lr_hidden_channels or hidden_channels)
        hr_hidden_channels = tuple(hr_hidden_channels or hidden_channels)
        lr_compression_rate = tuple(lr_compression_rate or compression_rate)
        hr_compression_rate = tuple(hr_compression_rate or (2, 2, 1))

        self.lr_branch = CompressionBranch(
            in_channels=in_channels,
            context_dim=context_dim,
            hidden_channels=lr_hidden_channels,
            compression_rate=lr_compression_rate,
            num_attention_heads=num_attention_heads,
        ) if use_lr_branch else None
        self.hr_branch = CompressionBranch(
            in_channels=in_channels,
            context_dim=context_dim,
            hidden_channels=hr_hidden_channels,
            compression_rate=hr_compression_rate,
            num_attention_heads=num_attention_heads,
        ) if use_hr_branch else None

    def forward(self, history_latents: torch.Tensor) -> torch.Tensor:
        branches = []
        if self.lr_branch is not None:
            branches.append(self.lr_branch(history_latents))
        if self.hr_branch is not None:
            branches.append(self.hr_branch(history_latents))
        return torch.cat(branches, dim=1)


class FramePreservationDiffusion(BaseModel):
    """Bidirectional frame-query pretraining model for history embeddings."""

    def __init__(self, args, device):
        super().__init__(args, device)
        self.num_train_timestep = args.num_train_timestep
        self.num_query_frames = getattr(args, "num_query_frames", 4)
        self.query_sampling = getattr(args, "query_sampling", "random")
        self.history_noise_min_timestep = getattr(args, "history_noise_min_timestep", 200)
        self.history_noise_max_timestep = getattr(args, "history_noise_max_timestep", 1000)

        context_dim = getattr(args, "history_context_dim", self.generator.model.dim)
        hidden_channels = tuple(getattr(args, "history_hidden_channels", [64, 128, 256, 512, 512]))
        compression_rate = tuple(getattr(args, "history_compression_rate", [4, 4, 2]))
        self.history_encoder = LightweightHistoryEncoder(
            in_channels=getattr(args, "latent_channels", 16),
            context_dim=context_dim,
            hidden_channels=hidden_channels,
            compression_rate=compression_rate,
            num_attention_heads=getattr(args, "history_num_attention_heads", 8),
            use_lr_branch=getattr(args, "use_lr_branch", True),
            use_hr_branch=getattr(args, "use_hr_branch", True),
            lr_hidden_channels=tuple(getattr(args, "lr_hidden_channels", hidden_channels)),
            hr_hidden_channels=tuple(getattr(args, "hr_hidden_channels", hidden_channels)),
            lr_compression_rate=tuple(getattr(args, "lr_compression_rate", compression_rate)),
            hr_compression_rate=tuple(getattr(args, "hr_compression_rate", [2, 2, 1])),
        )
        self.history_encoder.requires_grad_(True)

        self.use_lora = getattr(args, "use_lora", True)
        self.num_lora_layers = 0
        if self.use_lora:
            self.num_lora_layers = inject_lora_linear(
                self.generator.model,
                rank=getattr(args, "lora_rank", 128),
                alpha=getattr(args, "lora_alpha", getattr(args, "lora_rank", 128)),
                dropout=getattr(args, "lora_dropout", 0.0),
                target_modules=tuple(getattr(args, "lora_target_modules", ["blocks.*"])),
            )
        else:
            self.generator.model.requires_grad_(True)

        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

    def _initialize_models(self, args, device):
        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=False)
        self.generator.model.requires_grad_(True)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _sample_query_mask(self, batch_size: int, num_frames: int, device) -> torch.Tensor:
        query_count = min(self.num_query_frames, num_frames)
        mask = torch.zeros(batch_size, num_frames, dtype=torch.bool, device=device)
        if self.query_sampling == "uniform":
            ids = torch.linspace(0, num_frames - 1, query_count, device=device).long()
            mask[:, ids] = True
            return mask

        for batch_index in range(batch_size):
            ids = torch.randperm(num_frames, device=device)[:query_count]
            mask[batch_index, ids] = True
        return mask

    def _build_masked_history(self, clean_latent: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames = clean_latent.shape[:2]
        noise = torch.randn_like(clean_latent)
        index = self._get_timestep(
            self.history_noise_min_timestep,
            self.history_noise_max_timestep,
            batch_size,
            num_frames,
            num_frame_per_block=1,
            uniform_timestep=False,
        )
        timestep = self.scheduler.timesteps[index].to(dtype=self.dtype, device=self.device)
        noised = self.scheduler.add_noise(
            clean_latent.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1),
        ).unflatten(0, (batch_size, num_frames))
        return torch.where(query_mask[:, :, None, None, None], clean_latent, noised)

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        batch_size, num_frames = clean_latent.shape[:2]
        query_mask = self._sample_query_mask(batch_size, num_frames, clean_latent.device)
        masked_history = self._build_masked_history(clean_latent, query_mask)
        history_context = self.history_encoder(masked_history)

        query_latents = clean_latent[query_mask].reshape(
            batch_size, -1, *clean_latent.shape[2:]
        )
        query_frames = query_latents.shape[1]
        noise = torch.randn_like(query_latents)
        index = self._get_timestep(
            0,
            self.scheduler.num_train_timesteps,
            batch_size,
            query_frames,
            num_frame_per_block=1,
            uniform_timestep=True,
        )
        timestep = self.scheduler.timesteps[index].to(dtype=self.dtype, device=self.device)
        noisy_latents = self.scheduler.add_noise(
            query_latents.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1),
        ).unflatten(0, (batch_size, query_frames))
        training_target = self.scheduler.training_target(query_latents, noise, timestep)

        conditional_dict = dict(conditional_dict)
        conditional_dict["history_context"] = history_context
        flow_pred, x0_pred = self.generator(
            noisy_image_or_video=noisy_latents,
            conditional_dict=conditional_dict,
            timestep=timestep,
        )
        loss = F.mse_loss(flow_pred.float(), training_target.float(), reduction="none").mean(dim=(2, 3, 4))
        loss = loss * self.scheduler.training_weight(timestep).unflatten(0, (batch_size, query_frames))
        loss = loss.mean()
        return loss, {
            "x0": query_latents.detach(),
            "x0_pred": x0_pred.detach(),
            "history_context_tokens": torch.tensor(history_context.shape[1], device=clean_latent.device),
            "num_lora_layers": torch.tensor(self.num_lora_layers, device=clean_latent.device),
        }
