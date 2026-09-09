"""Neural network encoders for time series representation learning."""

import torch.nn as nn

__all__ = ["FCN"]


class FCN(nn.Module):
    """Fully convolutional network encoder with a projection head.

    Architecture follows the original mixup contrastive learning paper:
    three dilated Conv1d blocks with BatchNorm and ReLU, followed by global
    average pooling and a two-layer projection head. The convolution
    paddings are chosen so that the temporal length is preserved.

    Parameters
    ----------
    n_channels : int
        Number of input channels (1 for univariate series).
    feature_dim : int, default=128
        Output dimensionality of the encoder (and of the projection head).
    """

    def __init__(self, n_channels: int, feature_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 128, kernel_size=7, padding=6, dilation=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=8, dilation=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=8, dilation=8),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.proj_head = nn.Sequential(
            nn.Linear(128, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, x):
        """Encode a batch ``x`` of shape ``(batch, n_channels, time)``.

        Returns the projection head output ``z`` and the encoder
        representation ``h``, both of shape ``(batch, feature_dim)``.
        """
        h = self.encoder(x)
        z = self.proj_head(h)
        return z, h
