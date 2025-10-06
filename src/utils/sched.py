import math

def cosine_warmup_lr(epoch, step, steps_per_epoch, base_lr, warmup_epochs=1, total_epochs=10):
    gstep = epoch * steps_per_epoch + step
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    if gstep < warmup_steps:
        return base_lr * (gstep + 1) / max(1, warmup_steps)
    # cosine decay
    progress = (gstep - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))
