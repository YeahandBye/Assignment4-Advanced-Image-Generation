import argparse
import os
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from helper_lib.diffusion_model import (
    SmallUNet,
    cosine_schedule
)


def train(args):
    torch.manual_seed(42)

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

    model = SmallUNet().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4
    )

    initial_weights = (
        model.input_conv[0].weight.detach().clone()
    )

    print("Device:", device)
    print("Training samples:", len(dataset))
    print("Epochs:", args.epochs)
    print("Batches per epoch:", len(loader))

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (images, _) in enumerate(loader):
            images = images.to(device)
            batch_size = images.size(0)

            # Random diffusion time for each image
            t = torch.rand(
                batch_size, device=device
            )

            # Add Gaussian noise
            noise = torch.randn_like(images)

            signal_rate, noise_rate = cosine_schedule(t)

            signal_rate = signal_rate[:, None, None, None]
            noise_rate = noise_rate[:, None, None, None]

            noisy_images = (
                signal_rate * images
                + noise_rate * noise
            )

            # Predict the added noise
            predicted_noise = model(noisy_images, t)

            loss = F.mse_loss(
                predicted_noise, noise
            )

            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite diffusion loss")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            print(
                f"Epoch {epoch+1}/{args.epochs} "
                f"Batch {batch_idx+1}/{len(loader)} "
                f"Loss: {loss.item():.4f}",
                flush=True
            )

        average_loss = total_loss / len(loader)

        print(
            f"Epoch {epoch+1} Average Loss: "
            f"{average_loss:.4f}",
            flush=True
        )

    weight_change = (
        model.input_conv[0].weight.detach()
        - initial_weights
    ).abs().max().item()

    print("Maximum weight change:", weight_change)

    assert weight_change > 0, "Weights did not update"

    os.makedirs("checkpoints", exist_ok=True)

    torch.save(
        model.state_dict(),
        "checkpoints/diffusion_cifar10_test.pth"
    )

    print(
        "Checkpoint saved: "
        "checkpoints/diffusion_cifar10_test.pth"
    )
    print("Diffusion training test completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs", type=int, default=1
    )
    parser.add_argument(
        "--max-samples", type=int, default=128
    )
    parser.add_argument(
        "--batch-size", type=int, default=16
    )

    args = parser.parse_args()
    train(args)
