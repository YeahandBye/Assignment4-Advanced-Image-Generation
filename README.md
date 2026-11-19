# Assignment 4: Advanced Image Generation

Columbia University | Applied Generative AI

## Project Overview

This project extends the image generation API developed in earlier course activities by integrating two generative models:

1. Energy-Based Model (EBM)
2. Diffusion Model

Both models are trained using the CIFAR-10 image dataset. The existing MNIST GAN implementation is preserved for comparison and backward compatibility.

## Dataset

CIFAR-10 contains 60,000 RGB images across 10 classes, including airplanes, automobiles, birds, cats, dogs, and ships.

- Training images: 50,000
- Test images: 10,000
- Image resolution: 32 x 32
- Channels: 3 (RGB)
- Normalization: [-1, 1]

## Energy-Based Model

The EBM uses a convolutional neural network to assign an energy score to each image.

Training is performed using contrastive learning and Langevin sampling to generate negative samples.

Training configuration:
- 2,000 CIFAR-10 training images
- 5 epochs
- Batch size: 32
- Langevin steps during training: 10

The model successfully learned to assign lower energy to real images than to negative samples.

However, generated images remained noisy, indicating limitations in the current training setup.

## Diffusion Model

The diffusion model uses a compact UNet architecture with sinusoidal time embeddings.

Gaussian noise is added to the original images, and the model learns to predict the added noise using Mean Squared Error (MSE).

Training configuration:
- 2,000 CIFAR-10 training images
- 5 epochs
- Batch size: 16
- Cosine noise schedule

The final training epoch achieved an average MSE loss of 0.2654.

A denoising diagnostic demonstrated that the model could partially recover image structures from noisy CIFAR-10 test images.

However, unconditional image generation from random noise remained limited.

## API Integration

The project extends the previous FastAPI implementation and supports three image generation endpoints:

- GET /generate
  - Generate MNIST digits using the original GAN.

- GET /generate/ebm
  - Generate RGB images using EBM and Langevin sampling.

- GET /generate/diffusion
  - Generate RGB images using iterative reverse diffusion.

Both new endpoints were tested successfully and returned HTTP 200 responses with PNG images.

## Installation

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Download and prepare CIFAR-10:

```bash
python3 prepare_cifar10.py
```

## Training

Train the EBM:

```bash
python3 train_ebm.py --epochs 5 --max-samples 2000 --batch-size 32 --langevin-steps 10
```

Train the diffusion model:

```bash
python3 train_diffusion.py --epochs 5 --max-samples 2000 --batch-size 16
```

Note: Training scripts save their weights using test checkpoint filenames. The provided five-epoch checkpoints were copied from the completed training runs.

## Running the API

Start the FastAPI application:

```bash
python3 -m uvicorn api_assignment4:app --host 127.0.0.1 --port 8000
```

Open Swagger documentation:

http://127.0.0.1:8000/docs

Example endpoints:

http://127.0.0.1:8000/generate/ebm?num_samples=4&steps=20

http://127.0.0.1:8000/generate/diffusion?num_samples=4&steps=20

## Theory Questions

The theoretical exercises are included in two Jupyter Notebooks:

- Assignment4_Part2_Diffusion_Theory.ipynb
- Assignment4_Part3_Energy_Theory.ipynb

They cover sinusoidal embeddings, UNet architecture, gradient calculation, autograd, detach(), and gradient accumulation.

## Limitations and Future Improvements

This project demonstrates the implementation, training, sampling, and API integration of EBM and Diffusion models.

The training experiments used only a subset of CIFAR-10, with limited training epochs and compact neural networks.

While the diffusion model showed measurable denoising improvements, both models produced mostly noisy images during unconditional generation.

Potential improvements include longer training, larger networks, improved sampling strategies, and more systematic model evaluation.
