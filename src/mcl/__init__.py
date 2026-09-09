"""mixupcl: mixup contrastive learning for time series.

Self-supervised representation learning for time series with the MNT-Xent
loss, based on Wickstrøm et al., "Mixing up contrastive learning:
Self-supervised representation learning for time series", Pattern
Recognition Letters 155:54-61, 2022.
"""

from .data import TimeSeriesDataset, load_dataset, to_numpy3d
from .losses import MixupLoss
from .models import FCN
from .trainer import embed, evaluate, mixup_batch, train

__version__ = "0.1.0"

__all__ = [
    "FCN",
    "MixupLoss",
    "TimeSeriesDataset",
    "embed",
    "evaluate",
    "load_dataset",
    "mixup_batch",
    "to_numpy3d",
    "train",
]
