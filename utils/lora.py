from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    rank: int = 128
    alpha: float = 128.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("blocks.*",)


class LoRALinear(nn.Module):
    """Low-rank adapter around an existing Linear layer.

    The wrapped base layer stays frozen and the adapter learns
    BAx * alpha / rank, matching the DiT LoRA setup used for the
    history-encoder pretraining/finetuning stages.
    """

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(rank, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, rank))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        lora = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base + lora * self.scaling


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    for pattern in patterns:
        regex = "^" + re.escape(pattern).replace("\\*", ".*") + "$"
        if re.match(regex, name):
            return True
    return False


def mark_only_lora_as_trainable(module: nn.Module):
    for name, param in module.named_parameters():
        param.requires_grad_("lora_" in name)


def inject_lora_linear(
    module: nn.Module,
    rank: int = 128,
    alpha: float = 128.0,
    dropout: float = 0.0,
    target_modules: Iterable[str] = ("blocks.*",),
) -> int:
    """Replace selected Linear layers by LoRALinear wrappers.

    Args:
        module: model to modify in-place.
        rank: LoRA rank.
        alpha: LoRA alpha.
        dropout: dropout applied before the LoRA A projection.
        target_modules: glob-like module-name patterns, e.g. ``blocks.*``.

    Returns:
        Number of Linear layers wrapped.
    """
    patterns = tuple(target_modules)
    replacements: list[tuple[nn.Module, str, nn.Linear]] = []

    for full_name, child in module.named_modules():
        for child_name, grandchild in child.named_children():
            leaf_name = f"{full_name}.{child_name}" if full_name else child_name
            if isinstance(grandchild, nn.Linear) and _matches_any(leaf_name, patterns):
                replacements.append((child, child_name, grandchild))

    for parent, child_name, linear in replacements:
        setattr(parent, child_name, LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout))

    mark_only_lora_as_trainable(module)
    return len(replacements)
