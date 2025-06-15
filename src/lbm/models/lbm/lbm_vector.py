import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from ..base.base_model import BaseModel
from .lbm_config import LBMConfig


@dataclass
class SequenceLBMConfig(LBMConfig):
    """Configuration for ``SequenceLBMModel``.

    This inherits from :class:`LBMConfig` but assumes the source and target
    tensors are embeddings of shape ``(B, T, D)``.
    """

    source_key: str = "source_embedding"
    target_key: str = "target_embedding"


class SequenceDenoiser(nn.Module):
    """Simple transformer-based denoiser for sequence data."""

    def __init__(self, dim: int, nhead: int = 4, hidden_dim: int = 512, num_layers: int = 2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        _ = timestep  # timestep is unused but kept for API consistency
        h = self.transformer(sample)
        return self.to_out(h)


class SequenceLBMModel(BaseModel):
    """LBM variant operating on sequences of embeddings.

    The model injects bridge noise between ``z_source`` and ``z`` and trains a
    denoiser to predict the noise. A custom embedding loss aligns the predictions
    with the ground truth embeddings.
    """

    def __init__(
        self,
        config: SequenceLBMConfig,
        denoiser: Optional[nn.Module],
        training_noise_scheduler: FlowMatchEulerDiscreteScheduler,
        sampling_noise_scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__(config)
        self.denoiser = denoiser
        self.training_noise_scheduler = training_noise_scheduler
        self.sampling_noise_scheduler = sampling_noise_scheduler
        self.latent_loss_type = config.latent_loss_type
        self.latent_loss_weight = config.latent_loss_weight
        self.bridge_noise_sigma = config.bridge_noise_sigma

    def _get_sigmas(self, scheduler, timesteps, n_dim=3, dtype=torch.float32, device="cpu"):
        sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def _timestep_sampling(self, n_samples: int, device: torch.device) -> torch.Tensor:
        idx = torch.randint(
            0,
            self.training_noise_scheduler.config.num_train_timesteps,
            (n_samples,),
            device="cpu",
        )
        return self.training_noise_scheduler.timesteps[idx].to(device)

    def embedding_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.latent_loss_type == "l2":
            return ((prediction - target) ** 2).mean(dim=[1, 2])
        if self.latent_loss_type == "l1":
            return torch.abs(prediction - target).mean(dim=[1, 2])
        raise NotImplementedError(f"Loss type {self.latent_loss_type} not implemented")

    def forward(self, batch: Dict[str, Any], *args, **kwargs) -> Dict[str, torch.Tensor]:
        z = batch[self.config.target_key]
        z_source = batch[self.config.source_key]

        timestep = self._timestep_sampling(n_samples=z.shape[0], device=z.device)
        sigmas = self._get_sigmas(self.training_noise_scheduler, timestep, n_dim=3, device=z.device)

        noisy_sample = (
            sigmas * z_source
            + (1.0 - sigmas) * z
            + self.bridge_noise_sigma
            * (sigmas * (1.0 - sigmas)) ** 0.5
            * torch.randn_like(z)
        )

        prediction = self.denoiser(noisy_sample, timestep)
        target = z_source - z

        loss = self.embedding_loss(prediction, target.detach())
        return {
            "loss": loss.mean(),
            "prediction": prediction,
            "noisy_sample": noisy_sample,
        }
