import math
import torch
import torch.nn as nn

# this function is also in pt3_linear.py
def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    '''
    transformer language model: distribution p_theta(x_i+1 | x1:i) for each sequence x of length m+1 and i=1..m
    given training set D of sequences of length m, use standard cross-entropy (negative log likelihood) loss:
    L(theta) = -1/N * sum_{x in D} sum_{i=1}^{m} log p_theta(x_i+1 | x1:i)
    '''
    # subtract largest element for numerical stability 
    # cancel out log and exp when possible
    # handle additional batch dim and return average across batch 
    # batch always comes first before vocabulary size
    
    # for each row, find largest logit and subtract it from the row
    logits = logits - logits.max(dim=-1, keepdim=True).values
    logsumexp = torch.log(torch.exp(logits).sum(dim=-1, keepdim=True))
    # each entry is log of sum of exponents of all logits for that position

    # get correct token logit 
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    # shape: (batch_size, sequence_length)

    # compute cross-entropy loss
    loss = logsumexp - target_logits
    # shape: (batch_size, sequence_length)

    # return average across batch
    return loss.mean()