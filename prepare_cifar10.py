import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# CIFAR-10 contains RGB images of size 32x32.
# Normalize pixel values to [-1, 1] for generative models.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

images, labels = next(iter(train_loader))

print("Training samples:", len(train_dataset))
print("Testing samples:", len(test_dataset))
print("Batch image shape:", images.shape)
print("Image value range:", images.min().item(), images.max().item())
print("Available classes:", train_dataset.classes)
print("MPS available:", torch.backends.mps.is_available())
