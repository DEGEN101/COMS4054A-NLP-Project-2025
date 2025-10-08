import torch as T
import torch.nn as nn
import numpy as np


def get_attentions(Q: T.Tensor, K: T.Tensor, V: T.Tensor, d_k: int, mask: T.Tensor | None = None) -> tuple[T.Tensor, T.Tensor]:
    scores: T.Tensor = Q @ K.transpose(-2, -1) / np.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(~mask, -np.inf)
    
    probs = T.softmax(scores, dim=-1)

    if mask is not None:
        probs = T.where(T.isnan(probs), T.zeros_like(probs), probs)

    return probs @ V, probs


def split_heads(matrix: T.Tensor, n_heads: int, d_k: int):
    batch_size, sequence_length, _ = matrix.size()
    return matrix.view(batch_size, sequence_length, n_heads, d_k)


def combine_heads(matrix: T.Tensor):
    batch_size, sequence_length, n_heads, d_k = matrix.size()
    return matrix.contiguous().view(batch_size, sequence_length, n_heads * d_k)


def generate_mask(batch_tokens: T.Tensor, pad_token_id: int = 0) -> T.Tensor:
    _, seq_len = batch_tokens.size()

    pad_mask = (batch_tokens != pad_token_id).unsqueeze(1).unsqueeze(2)
    causal_mask = T.tril(T.ones((seq_len, seq_len), dtype=T.bool)).to(batch_tokens.device)
    combined_mask = pad_mask & causal_mask.unsqueeze(0)

    return combined_mask
