import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss


def shift_tokens(logits, labels):
    return logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()


def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor, vocab_size: int, ignore_index: int = -100, shifted=False) -> torch.Tensor:
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
    shift_logits = shift_logits.view(-1, vocab_size)
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
