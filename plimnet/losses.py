import numpy as np
import torch
from torch.nn import functional as F


def positive_weights(frame, columns):
    weights = []
    for column in columns:
        labels = frame[column].dropna().to_numpy()
        positives = int(np.sum(labels == 1))
        negatives = int(np.sum(labels == 0))
        weights.append(negatives / positives if positives and negatives else 1.0)
    return torch.tensor(weights, dtype=torch.float32)


def masked_bce(logits, targets, mask, pos_weight):
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    denominator = mask.sum(0)
    per_target = (losses * mask).sum(0) / denominator.clamp_min(1.0)
    active = (denominator > 0).float()
    return (per_target * active).sum() / active.sum().clamp_min(1.0)


def joint_loss(lineage_logits, task_logits, batch, task_weights, lineage_weights, lineage_weight=0.4):
    device = task_logits.device
    main = masked_bce(task_logits, batch["task_labels"].to(device), batch["task_mask"].to(device), task_weights)
    auxiliary = masked_bce(lineage_logits, batch["lineage_labels"].to(device),
                           batch["lineage_mask"].to(device), lineage_weights)
    return main + lineage_weight * auxiliary
