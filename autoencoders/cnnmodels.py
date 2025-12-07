import torch
from einops.layers.torch import Rearrange
from torch import nn


class CNNEncoder(nn.Sequential):
    """
    The encoder is a simple multi-layer perceptron going from
    n_input -> 64 -> n_hidden features with a LeakyReLU activation
    in between
    """

    def __init__(self, n_input_channels: int, n_hidden: int = 8):
        super().__init__(
            nn.Conv2d(n_input_channels, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(64),
            nn.Conv2d(64, 4, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(4),
            Rearrange("b c h w -> b (c h w)"),
            # this following layer is hardcoded for MNIST images (28x28)
            nn.Linear(100, 32),
            nn.ReLU(inplace=True),
            nn.InstanceNorm1d(32),
            nn.Linear(32, n_hidden),
            nn.ReLU(inplace=True),
        )


class CNNDecoder(nn.Sequential):
    """
    The decoder is the opposite of the encoder going from
    n_hidden -> 64 -> n_input features with a LeakyReLU activation
    in between and a sigmoid activation at the end.
    """

    def __init__(self, n_input_channels: int, n_hidden: int = 8):
        super().__init__(
            nn.Linear(n_hidden, 32),
            nn.ReLU(inplace=True),
            nn.InstanceNorm1d(32),
            # this is hardcoded to the MNIST dataset (28x28)
            nn.Linear(32, 100),
            nn.ReLU(inplace=True),
            Rearrange("b (c h w) -> b c h w", c=4, h=5, w=5),
            nn.InstanceNorm2d(4),
            nn.ConvTranspose2d(4, 64, kernel_size=4, stride=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(64),
            nn.ConvTranspose2d(64, 4, kernel_size=4, stride=2, output_padding=1),
            nn.InstanceNorm2d(4),
            nn.Conv2d(4, n_input_channels, (2, 2), stride=1),
            nn.Sigmoid(),
        )


class CNNAutoEncoder(nn.Module):
    def __init__(self, n_input_channels: int, n_hidden: int = 8):
        # initialize the pytorch module
        super().__init__()

        # set up the encoder
        self.encoder = CNNEncoder(n_input_channels, n_hidden)
        self.decoder = CNNDecoder(n_input_channels, n_hidden)

    def forward(self, x: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
        z = self.encoder(x)
        return self.decoder(z), z
