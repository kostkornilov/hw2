"""PyTorch backend for the paper's compact kernels on GPUs without AdaSplash support."""

import torch
from entmax import sparsemax, entmax15, entmax_bisect


def compact_attention(q, k, v, mask, alpha):
    # Match the author's dot-product scale=1.0. FP32 protects support selection
    # from FP16 overflow; cast only the final weights/value product back.
    with torch.autocast(device_type=q.device.type, enabled=False):
        scores = q.float() @ k.float().transpose(-1, -2)
        has_keys = mask.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~mask, float("-inf"))
        scores = torch.where(has_keys, scores, torch.zeros_like(scores))
        if alpha == 2.0:
            weights = sparsemax(scores, dim=-1)
        elif alpha == 1.5:
            weights = entmax15(scores, dim=-1)
        else:
            weights = entmax_bisect(scores, alpha=alpha, dim=-1, n_iter=50)
        weights = weights.masked_fill(~mask, 0)
        return (weights @ v.float()).to(v.dtype)
