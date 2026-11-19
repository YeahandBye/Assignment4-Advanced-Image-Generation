import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from helper_lib.diffusion_model import SmallUNet, cosine_schedule

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

torch.manual_seed(42)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=transform
)

model = SmallUNet().to(device)
model.load_state_dict(
    torch.load(
        "checkpoints/diffusion_cifar10_5epochs.pth",
        map_location=device,
        weights_only=True
    )
)
model.eval()

images = torch.stack(
    [dataset[i][0] for i in range(16)]
).to(device)

noise = torch.randn_like(images)

print("Diffusion denoising diagnostic")
print("Device:", device)

with torch.no_grad():
    for time_value in [0.2, 0.5, 0.8]:
        t = torch.full(
            (16,), time_value, device=device
        )

        signal, noise_rate = cosine_schedule(t)
        signal = signal[:, None, None, None]
        noise_rate = noise_rate[:, None, None, None]

        noisy = signal * images + noise_rate * noise

        predicted_noise = model(noisy, t)

        # Estimate the original clean images
        reconstructed = (
            noisy - noise_rate * predicted_noise
        ) / signal

        noise_mse = F.mse_loss(
            predicted_noise, noise
        ).item()

        reconstruction_mse = F.mse_loss(
            reconstructed, images
        ).item()

        baseline_mse = F.mse_loss(
            noisy / signal, images
        ).item()

        print(
            f"t={time_value:.1f} | "
            f"Noise MSE={noise_mse:.4f} | "
            f"Reconstruction MSE={reconstruction_mse:.4f} | "
            f"Baseline MSE={baseline_mse:.4f}"
        )

        # Save original / noisy / reconstructed images
        comparison = torch.cat([
            images[:4],
            noisy[:4],
            reconstructed[:4]
        ])

        save_image(
            (comparison.cpu().clamp(-1, 1) + 1) / 2,
            f"outputs/denoising_t{time_value:.1f}.png",
            nrow=4
        )

print("Diagnostic completed!")
