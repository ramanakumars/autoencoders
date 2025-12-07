import torch
from torch import nn


class MLPEncoder(nn.Sequential):
    """
    The encoder is a simple multi-layer perceptron going from
    n_input -> 64 -> n_hidden features with a sigmoid activation
    in between
    """

    def __init__(self, n_inputs: int, n_hidden: int = 16):
        super().__init__(
            nn.Linear(n_inputs, 64),
            nn.ReLU(inplace=True),
            nn.InstanceNorm1d(64),
            nn.Linear(64, n_hidden),
            nn.ReLU(inplace=True),
        )


class MLPDecoder(nn.Sequential):
    """
    The decoder is the opposite of the encoder going from
    n_hidden -> 64 -> n_input features with a sigmoid activation
    in between and a sigmoid activation at the end.
    """

    def __init__(self, n_inputs: int, n_hidden: int = 16):
        super().__init__(
            nn.Linear(n_hidden, 64),
            nn.ReLU(inplace=True),
            nn.InstanceNorm1d(64),
            nn.Linear(64, n_inputs),
            nn.Sigmoid(),
        )


class MLPAutoEncoder(nn.Module):
    def __init__(self, n_inputs: int, n_hidden: int = 16):
        # initialize the pytorch module
        super().__init__()

        # set up the encoder
        self.encoder = MLPEncoder(n_inputs, n_hidden)
        self.decoder = MLPDecoder(n_inputs, n_hidden)

    def forward(self, x: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
        z = self.encoder(x)
        return self.decoder(z), z
