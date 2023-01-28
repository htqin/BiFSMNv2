# ref to: https://github.com/moskomule/mixup.pytorch

from typing import Tuple
import torch
import torch.nn.functional as F


def partial_mixup(input: torch.Tensor, gamma: float,
                  indices: torch.Tensor) -> torch.Tensor:
    if input.size(0) != indices.size(0):
        raise RuntimeError("Size mismatch!")
    perm_input = input[indices]
    return input.mul(gamma).add(perm_input, alpha=1 - gamma)


def mixup(input: torch.Tensor, target: torch.Tensor, gamma: float,
          num_classes) -> Tuple[torch.Tensor, torch.Tensor]:
    target = F.one_hot(target, num_classes)
    indices = torch.randperm(input.size(0),
                             device=input.device,
                             dtype=torch.long)
    return partial_mixup(input, gamma,
                         indices), partial_mixup(target, gamma, indices)


def naive_cross_entropy_loss(input: torch.Tensor,
                             target: torch.Tensor) -> torch.Tensor:
    return -(input.log_softmax(dim=-1) * target).sum(dim=-1).mean()
