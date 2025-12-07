# Autoencoder Tutorials with PyTorch

A collection of tutorials exploring autoencoders using PyTorch on the MNIST dataset. This repository demonstrates how to build and train both MLP-based and CNN-based autoencoders, with visualization of learned latent representations through PCA.

## Overview

This repository contains implementations and interactive tutorials for understanding autoencoders through hands-on examples. You'll learn how autoencoders compress and reconstruct images, and how different architectures affect the learned representations.

## Repository Structure

```
.
├── autoencoders/
│   ├── mlpmodels.py      # MLP autoencoder implementations
│   └── cnnmodels.py      # CNN autoencoder implementations
├── MNIST.ipynb           # Dataset exploration and baseline PCA
├── MNIST MLP AE.ipynb    # MLP autoencoder training and analysis
├── MNIST CNN AE.ipynb    # CNN autoencoder training and analysis
├── pyproject.toml        # Project dependencies
└── README.md
```

## Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management, which provides fast and reliable Python package installation.
All dependencies are managed through `uv` and specified in `pyproject.toml`.

### Prerequisites
- Python 3.11+
- `uv` package manager

### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/ramanakumars/autoencoders.git
cd autoencoder

# Create virtual environment and install dependencies
uv sync
```

## Usage

Launch Jupyter to run the notebooks:

```bash
# Activate the virtual environment
uv run jupyter notebook
```

Then open the notebooks in the following recommended order:
1. `MNIST.ipynb` - Start here to understand the dataset
2. `MNIST MLP AE.ipynb` - Using a simple autoencoder on the MNIST dataset
3. `MNIST CNN AE.ipynb` - Using a more advanced Convolutional Auto-Encoder on the MNIST dataset

## Tutorials

### 1. MNIST Dataset Exploration (`MNIST.ipynb`)
- Load and visualize the MNIST dataset using torchvision
- Explore dataset statistics and sample images
- Apply PCA on raw flattened pixel data as a baseline
- Understand the dimensionality and structure of the data

### 2. MLP Autoencoder (`MNIST MLP AE.ipynb`)
- Build a multi-layer perceptron (MLP) autoencoder
- Train the model to compress and reconstruct MNIST digits
- Extract latent embeddings from the bottleneck layer
- Apply PCA on learned embeddings and compare with raw data
- Visualize reconstruction quality

### 3. CNN Autoencoder (`MNIST CNN AE.ipynb`)
- Implement a convolutional neural network (CNN) autoencoder
- Leverage spatial structure in images through convolutional layers
- Train and evaluate the CNN model
- Analyze latent space representations with PCA
- Compare CNN vs MLP performance and learned features

## Contributing

Contributions are welcome! Feel free to:
- Report bugs or issues
- Suggest improvements or new tutorials
- Submit pull requests

## License

This project is licensed under the MIT License - see the LICENSE file for details.
