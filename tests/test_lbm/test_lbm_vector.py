import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from lbm.models.lbm import SequenceLBMConfig, SequenceLBMModel
from lbm.models.lbm.lbm_vector import SequenceDenoiser

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_sequence_lbm_forward():
    config = SequenceLBMConfig()
    scheduler = FlowMatchEulerDiscreteScheduler()
    model = SequenceLBMModel(
        config=config,
        denoiser=SequenceDenoiser(dim=8),
        training_noise_scheduler=scheduler,
        sampling_noise_scheduler=scheduler,
    ).to(DEVICE)

    batch = {
        config.source_key: torch.randn(2, 5, 8).to(DEVICE),
        config.target_key: torch.randn(2, 5, 8).to(DEVICE),
    }

    output = model(batch)
    assert output["prediction"].shape == batch[config.source_key].shape
    assert output["noisy_sample"].shape == batch[config.source_key].shape

