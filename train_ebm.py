import argparse
import random

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from helper_lib.energy_model import EnergyModel


def langevin_sampling(model, images, steps=5,
                      step_size=0.1, noise_std=0.01):
    """Generate negative samples using input gradients."""
    x = images.detach()

    for _ in range(steps):
        x = x.detach().requires_grad_(True)

        energy = model(x).sum()
        grad = torch.autograd.grad(energy, x)[0]

        with torch.no_grad():
            x = x - step_size * grad
            x = x + noise_std * torch.randn_like(x)
            x = x.clamp(-1, 1)

    return x.detach()


def train(args):
    torch.manual_seed(42)
    random.seed(42)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=False,
        transform=transform
    )

    # Small subset for the initial training test.
    if args.max_samples < len(dataset):
        dataset = Subset(
            dataset, range(args.max_samples)
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    model = EnergyModel().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4
    )

    # Replay buffer stores previously generated samples.
    buffer = []

    print("Device:", device)
    print("Training samples:", len(dataset))
    print("Epochs:", args.epochs)
    print("Langevin steps:", args.langevin_steps)

    initial_weights = model.network[0].weight.detach().clone()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(device)
            batch_size = real.size(0)

            # Mix fresh noise with replay-buffer samples.
            fake = torch.rand_like(real) * 2 - 1

            for i in range(batch_size):
                if buffer and random.random() < 0.95:
                    fake[i] = random.choice(buffer).to(device)

            # Generate negative samples.
            fake = langevin_sampling(
                model, fake,
                steps=args.langevin_steps
            )

            # Save sampled images in the replay buffer.
            buffer.extend(
                img.cpu().clone() for img in fake
            )
            buffer = buffer[-256:]

            # Compute energy for real and fake images.
            real_energy = model(real)
            fake_energy = model(fake)

            contrastive_loss = (
                real_energy.mean() - fake_energy.mean()
            )

            regularization = (
                real_energy.square().mean()
                + fake_energy.square().mean()
            )

            loss = contrastive_loss + 0.1 * regularization

            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite training loss")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            print(
                f"Epoch {epoch+1}/{args.epochs} "
                f"Batch {batch_idx+1}/{len(loader)} "
                f"Loss: {loss.item():.4f} "
                f"Real E: {real_energy.mean().item():.4f} "
                f"Fake E: {fake_energy.mean().item():.4f}",
                flush=True
            )

        print(
            f"Average Loss: {total_loss/len(loader):.4f}",
            flush=True
        )

    weight_change = (
        model.network[0].weight.detach() - initial_weights
    ).abs().max().item()

    print("Maximum weight change:", weight_change)

    assert weight_change > 0, "Model weights did not update"

    import os
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    checkpoint_path = "checkpoints/ebm_cifar10_test.pth"

    torch.save(model.state_dict(), checkpoint_path)

    # Save a visualization of negative samples.
    save_image(
        (fake[:16].detach().cpu() + 1) / 2,
        "outputs/ebm_test_samples.png",
        nrow=4
    )

    print("Checkpoint saved:", checkpoint_path)
    print("Sample image saved: outputs/ebm_test_samples.png")
    print("EBM training test completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--langevin-steps", type=int, default=5)

    args = parser.parse_args()
    train(args)
