import torch
from torchvision.utils import save_image

from helper_lib.energy_model import EnergyModel

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model = EnergyModel().to(device)

state_dict = torch.load(
    "checkpoints/ebm_cifar10_5epochs.pth",
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()

torch.manual_seed(42)

x = torch.rand(16, 3, 32, 32, device=device) * 2 - 1

print("Starting Langevin Sampling...")

for step in range(200):
    x = x.detach().requires_grad_(True)

    energy = model(x).sum()
    grad = torch.autograd.grad(energy, x)[0]

    with torch.no_grad():
        x = x - 0.1 * grad
        x = x + 0.01 * torch.randn_like(x)
        x = x.clamp(-1, 1)

    if (step + 1) % 50 == 0:
        print(
            f"Step {step+1}/200 | "
            f"Mean Energy: {model(x).mean().item():.4f}",
            flush=True
        )

save_image(
    (x.detach().cpu() + 1) / 2,
    "outputs/ebm_generated_200steps.png",
    nrow=4
)

print("Generation complete!")
print("Saved: outputs/ebm_generated_200steps.png")
