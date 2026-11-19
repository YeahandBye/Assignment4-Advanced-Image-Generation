import torch
from helper_lib.energy_model import EnergyModel

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model = EnergyModel().to(device)
model.eval()

# Start with random images
x = torch.randn(4, 3, 32, 32, device=device)
x = x.clamp(-1, 1)

step_size = 0.1
noise_scale = 0.01
num_steps = 10

initial_energy = model(x).mean().item()

# Langevin dynamics: update image pixels, not model weights
for step in range(num_steps):
    x = x.detach().requires_grad_(True)

    energy = model(x).sum()
    grad = torch.autograd.grad(energy, x)[0]

    with torch.no_grad():
        x = x - step_size * grad
        x = x + noise_scale * torch.randn_like(x)
        x = x.clamp(-1, 1)

final_energy = model(x).mean().item()

print("Device:", device)
print("Generated image shape:", x.shape)
print("Initial energy:", initial_energy)
print("Final energy:", final_energy)
print("Langevin sampling test passed!")
