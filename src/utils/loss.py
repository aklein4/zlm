import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch.nn import CrossEntropyLoss
from utils.torch_utils import scale_gradient


def fast_lm_loss(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    ignore_index: int = -100,
    shift_logits: bool = True,
    shift_labels: bool = True,
    loss_threshold_lower: float | None = None,
    loss_threshold_upper: float | None = None,
):
    
    # shift if needed
    if shift_logits:
        logits = logits[..., :-1, :].contiguous()
    if shift_labels:
        labels = labels[..., 1:].contiguous()

    # reshape to remove batch dimension
    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)
    
    # calculate the mask
    mask = labels != ignore_index
    mask_sum = mask.float().sum()

    # calculate the logp
    logp = -F.cross_entropy(
        logits, labels,
        reduction='none',
    )
    p = logp.exp()

    # calculate the loss
    if loss_threshold_lower is None:
        assert loss_threshold_upper is None

        loss = -logp.masked_fill(~mask, 0.0).sum() / (mask_sum + 1)
        loss_threshold_perc = 0.0

    else:
        assert loss_threshold_upper is not None
        upper, lower = np.log(loss_threshold_upper), np.log(loss_threshold_lower)

        logp_mask = 1 - torch.clip(
            (logp - lower) / (upper - lower),
            0.0, 1.0
        ).detach()
        logp_scaled = scale_gradient(logp, logp_mask)

        loss = -logp_scaled.masked_fill(~mask, 0.0).sum() / (mask_sum + 1)
        loss_threshold_perc = logp_mask.masked_fill(~mask, 0.0).sum() / (mask_sum + 1)
        
    # calculate the accuracy
    correct = (logits.argmax(dim=-1) == labels).float()
    acc = correct.masked_fill(~mask, 0.0).sum() / (mask_sum + 1)

    # calculate the pcorr
    pcorr = p.masked_fill(~mask, 0.0).sum() / (mask_sum + 1)

    return {
        "loss": loss,
        "acc": acc,
        "pcorr": pcorr,
        "loss_threshold_perc": loss_threshold_perc
    }



def shift_tokens(logits, labels):
    return logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()


def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100, shifted=False) -> torch.Tensor:
    """
    Computes cross entropy loss of `logits` against the ground truth `labels` during
    next token prediction.

    Useful as the loss function of a LLM in pretraining or supervised finetuning.
    """
    # Shift so that tokens < n predict n
    if shifted:
        shift_logits, shift_labels = logits, labels
    else:
        shift_logits, shift_labels = shift_tokens(logits, labels)
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=ignore_index)
    shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
    shift_labels = shift_labels.view(-1)
    shift_labels = shift_labels.to(shift_logits.device)
    return loss_fct(shift_logits, shift_labels)


def accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100, shifted=False) -> float:
    """
    Computes the accuracy of the model's predictions against the ground truth labels.
    
    Args:
        logits (torch.Tensor): The model's output logits.
        labels (torch.Tensor): The ground truth labels.
        ignore_index (int): The index to ignore in the accuracy calculation.
    
    Returns:
        float: The accuracy as a percentage.
    """
    if not shifted:
        logits, labels = shift_tokens(logits, labels)
    mask = labels != ignore_index

    correct = (logits.argmax(dim=-1) == labels).float()
    correct = correct.masked_fill(~mask, 0.0).sum()

    total = mask.float().sum() + 1

    return correct / total    


def pcorr(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100, shifted=False) -> float:
    if not shifted:
        logits, labels = shift_tokens(logits, labels)
    mask = labels != ignore_index

    logp = -F.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        labels.view(-1),
        reduction='none',
    )
    
    p = logp.exp()
    p = p.masked_fill(~mask.view(-1), 0.0).sum()

    total = mask.float().sum() + 1

    return p / total   
