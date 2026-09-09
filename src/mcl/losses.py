"""The MNT-Xent loss for mixup contrastive learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MixupLoss"]


class MixupLoss(nn.Module):
    """Mixup normalized temperature-scaled cross entropy loss (MNT-Xent).

    Given projections ``z_1`` and ``z_2`` of two views of a batch, the
    projection ``z_aug`` of the mixed batch ``lam * x_1 + (1 - lam) * x_2``,
    and the mixing coefficient ``lam``, the task is to predict which of the
    two views each augmented sample was mixed from, with ``lam`` and
    ``1 - lam`` as soft targets.

    Parameters
    ----------
    tau : float, default=0.5
        Temperature used to scale the cosine similarities.
    """

    def __init__(self, tau: float = 0.5) -> None:
        super().__init__()
        self.tau = tau

    def forward(self, z_aug, z_1, z_2, lam: float) -> torch.Tensor:
        z_1 = F.normalize(z_1, dim=1)
        z_2 = F.normalize(z_2, dim=1)
        z_aug = F.normalize(z_aug, dim=1)

        n = z_1.shape[0]
        eye = torch.eye(n, device=z_1.device, dtype=z_1.dtype)
        labels = torch.cat((lam * eye, (1 - lam) * eye), dim=1)

        logits = torch.cat((z_aug @ z_1.T, z_aug @ z_2.T), dim=1) / self.tau

        log_probs = F.log_softmax(logits, dim=1)
        return -(labels * log_probs).sum(dim=1).mean()
