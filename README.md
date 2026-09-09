# mixupcl — Mixup Contrastive Learning for Time Series

[![CI](https://github.com/Wickstrom/MixupContrastiveLearning/actions/workflows/ci.yml/badge.svg)](https://github.com/Wickstrom/MixupContrastiveLearning/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Self-supervised representation learning for time series, based on

> Kristoffer Wickstrøm, Michael Kampffmeyer, Karl Øyvind Mikalsen, Robert Jenssen.
> **Mixing up contrastive learning: Self-supervised representation learning for time series.**
> *Pattern Recognition Letters*, 155:54-61, 2022. ([DOI](https://doi.org/10.1016/j.patrec.2022.02.007) / [arXiv:2203.09270](https://arxiv.org/abs/2203.09270))

Instead of relying on hand-picked noise augmentations, two randomly drawn time
series are mixed with a coefficient `λ ~ Beta(α, α)`. The learning task is to
predict the mixing coefficient from the embeddings of the two original series
and the mixed one, using soft targets `λ` and `1 - λ` in the **MNT-Xent** (mixup
normalized temperature-scaled cross entropy) loss. Representations are
evaluated with a 1-nearest-neighbor classifier on the encoder embeddings.

## Installation

```bash
pip install mixupcl                    # from PyPI
pip install git+https://github.com/Wickstrom/MixupContrastiveLearning.git  # from GitHub
```

Or from a local clone:

```bash
git clone https://github.com/Wickstrom/MixupContrastiveLearning.git
cd MixupContrastiveLearning
pip install .
```

Requires Python ≥ 3.10. Datasets from the [UCR](https://www.cs.ucr.edu/~eamonn/time_series_data/) and
[UEA](http://www.timeseriesclassification.com/) archives are downloaded and
cached automatically via [sktime](https://www.sktime.net).

## Quickstart

```python
import torch
import numpy as np

from mcl.data import TimeSeriesDataset, load_dataset
from mcl.models import FCN
from mcl.trainer import train

torch.manual_seed(0)
np.random.seed(0)

x_tr, y_tr = load_dataset("GunPoint", split="train")
x_te, y_te = load_dataset("GunPoint", split="test")

train_set = TimeSeriesDataset(x_tr, y_tr)
test_set = TimeSeriesDataset(x_te, y_te)

model = FCN(train_set.n_channels)
history = train(model, train_set, test_set, epochs=200, alpha=1.0)

print(f"Final 1-NN test accuracy: {history['accuracies'][-1]:.4f}")
```

More examples: [`examples/train_gunpoint.py`](examples/train_gunpoint.py) and
the [Colab notebook](examples/mixup_gunpoint.ipynb).

## Command line usage

```bash
mcl train --dataset GunPoint --epochs 200 --alpha 1.0
```

Useful options:

| Option | Default | Description |
| --- | --- | --- |
| `--dataset` | `GunPoint` | Any UCR/UEA dataset name (cached via sktime) |
| `--epochs` | `200` | Number of training epochs |
| `--alpha` | `1.0` | Beta(α, α) mixing distribution parameter |
| `--lr` | `1e-3` | Adam learning rate |
| `--tau` | `0.5` | MNT-Xent temperature |
| `--batch-size` | full batch | Minibatch size (paper uses full batch) |
| `--device` | auto | e.g. `cpu`, `cuda`, `mps` |
| `--seed` | none | Random seed |
| `--save-history PATH` | off | Write per-epoch loss/accuracy to CSV |

## API overview

| Object | Description |
| --- | --- |
| `mcl.models.FCN` | Dilated FCN encoder with projection head |
| `mcl.losses.MixupLoss` | The MNT-Xent loss |
| `mcl.data.load_dataset` | Load UCR/UEA datasets as `(n, channels, time)` arrays |
| `mcl.data.TimeSeriesDataset` | Tensor dataset for training/evaluation |
| `mcl.trainer.train` | Training loop, returns loss/accuracy history |
| `mcl.trainer.evaluate` | 1-NN accuracy on encoder embeddings |
| `mcl.trainer.mixup_batch` | Create a mixed batch and its mixing coefficient |

## Results

Results on all datasets in the UCR (univariate) and UEA (multivariate)
archives, for all methods considered in the paper, are provided in
[`results/UCR/`](results/UCR/) and [`results/UEA/`](results/UEA/), including
per-fold accuracies (5 folds) for the learning-based methods.

## Development

```bash
git clone https://github.com/Wickstrom/MixupContrastiveLearning.git
cd MixupContrastiveLearning
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                          # unit tests (synthetic data, no downloads)
pytest --run-integration        # + integration tests on real datasets
ruff check . && ruff format --check .
```

## Citation

If you use this code, please cite:

```bibtex
@article{wickstrom2022mixing,
  title = {Mixing up contrastive learning: Self-supervised representation
           learning for time series},
  author = {Wickstr{\o}m, Kristoffer and Kampffmeyer, Michael and
            Mikalsen, Karl {\O}yvind and Jenssen, Robert},
  journal = {Pattern Recognition Letters},
  volume = {155},
  pages = {54--61},
  year = {2022},
  publisher = {Elsevier},
  doi = {10.1016/j.patrec.2022.02.007}
}
```

## License

[MIT](LICENSE)
