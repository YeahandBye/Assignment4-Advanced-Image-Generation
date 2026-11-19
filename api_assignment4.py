import io
import math

import torch
from fastapi import Query
from fastapi.responses import StreamingResponse
from PIL import Image
from torchvision.utils import make_grid

# Reuse the existing GAN API.
from api import app

from helper_lib.energy_model import EnergyModel
from helper_lib.diffusion_model import SmallUNet

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


def load_weights(model, path):
    state = torch.load(
        path, map_location=device, weights_only=True
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


ebm = load_weights(
    EnergyModel(),
    "checkpoints/ebm_cifar10_5epochs.pth"
)

diffusion = load_weights(
    SmallUNet(),
    "checkpoints/diffusion_cifar10_5epochs.pth"
)


def return_png(images):
    images = (images.detach().cpu().clamp(-1, 1) + 1) / 2
    grid = make_grid(images, nrow=4, padding=2)

    array = (
        grid.permute(1, 2, 0)
        .mul(255).byte().numpy()
    )

    image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(
        buffer, media_type="image/png"
    )


@app.get("/generate/ebm")
def generate_ebm(
    num_samples: int = Query(default=4, ge=1, le=16),
    steps: int = Query(default=50, ge=1, le=200)
):
    x = torch.rand(
        num_samples, 3, 32, 32, device=device
    ) * 2 - 1

    for _ in range(steps):
        x = x.detach().requires_grad_(True)

        energy = ebm(x).sum()
        grad = torch.autograd.grad(energy, x)[0]

        with torch.no_grad():
            x = x - 0.1 * grad
            x = x + 0.01 * torch.randn_like(x)
            x = x.clamp(-1, 1)

    return return_png(x)


@app.get("/generate/diffusion")
def generate_diffusion(
    num_samples: int = Query(default=4, ge=1, le=16),
    steps: int = Query(default=50, ge=2, le=200)
):
    x = torch.randn(
        num_samples, 3, 32, 32, device=device
    )

    with torch.no_grad():
        for step in range(steps):
            t_now = 0.999 * (1 - step / steps)
            t_next = 0.999 * (1 - (step + 1) / steps)

            t = torch.full(
                (num_samples,), t_now, device=device
            )

            predicted_noise = diffusion(x, t)

            signal_now = math.cos(t_now * math.pi / 2)
            noise_now = math.sin(t_now * math.pi / 2)
            signal_next = math.cos(t_next * math.pi / 2)
            noise_next = math.sin(t_next * math.pi / 2)

            predicted_clean = (
                x - noise_now * predicted_noise
            ) / max(signal_now, 1e-4)

            predicted_clean = predicted_clean.clamp(-2, 2)

            x = (
                signal_next * predicted_clean
                + noise_next * predicted_noise
            )

    return return_png(x)
