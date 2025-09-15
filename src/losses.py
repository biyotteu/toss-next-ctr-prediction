import torch
import torch.nn.functional as F

# BCE with optional per-example weights (for downsampled batches)
def weighted_bce_with_logits(logits, targets, weight=None):
    return F.binary_cross_entropy_with_logits(logits, targets.float(), weight=weight)

# Focal BCE (optional)
def focal_bce_with_logits(logits, targets, gamma=2.0, alpha=0.25, reduction='mean'):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
    p_t = p*targets + (1-p)*(1-targets)
    alpha_t = alpha*targets + (1-alpha)*(1-targets)
    loss = alpha_t * (1 - p_t).pow(gamma) * ce
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss