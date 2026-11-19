import math
import os
import torch
from torchvision.utils import save_image

from helper_lib.diffusion_model import SmallUNet

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

model = SmallUNet().to(device)

state = torch.load(
    "checkpoints/diffusion_cifar10_5epochs.pth",
    map_location=device,
    weights_only=True
)

model.load_state_dict(state)
model.eval()

torch.manual_seed(42)

num_images = 16
num_steps = 100

# Begin with Gaussian noise
x = torch.randn(num_images, 3, 32, 32, device=device)

print("Device:", device)
print("Starting diffusion sampling...")

# Deterministic reverse diffusion using predicted noise
with torch.no_grad():
    for step in range(num_steps):
        t_now = 1.0 - step / num_steps
        t_next = max(0.0, 1.0 - (step + 1) / num_steps)

        t_batch = torch.full(
            (num_images,), t_now, device=device
        )

        predicted_noise = model(x, t_batch)

        signal_now = math.cos(t_now * math.pi / 2)
        noise_now = math.sin(t_now * math.pi / 2)

        signal_next = math.cos(t_next * math.pi / 2)
        noise_next = math.sin(t_next * math.pi / 2)

        # Recover an estimate of the clean image.
        safe_signal = max(signal_now, 1e-4)

        predicted_clean = (
            x - noise_now * predicted_noise
        ) / safe_signal

        predicted_clean = predicted_clean.clamp(-1, 1)

        # Move to the next, less noisy time step.
        x = (
            signal_next * predicted_clean
            + noise_next * predicted_noise
        )

        if (step + 1) % 20 == 0:
            print(
                f"Sampling step {step+1}/{num_steps}",
                flush=True
            )

os.makedirs("outputs", exist_ok=True)

save_image(
    (x.cpu().clamp(-1, 1) + 1) / 2,
    "outputs/diffusion_cifar10_100steps.png",
    nrow=4
)

print("Diffusion sampling completed!")
print("Saved: outputs/diffusion_cifar10_100steps.png")
