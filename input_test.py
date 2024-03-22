import torch.nn as nn
import torch
import math
import numpy as np

k = 1

def global_pattern(bs, seq_len):
    gp = torch.zeros(seq_len, seq_len)
    gp[0,:] = gp[:,0] = 1
    gp = gp.bool().expand(bs, seq_len, seq_len)
    return ~ gp

def local_pattern(bs, seq_len):
    lp = torch.zeros(seq_len, seq_len)
    diff = torch.abs(torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1))
    k_clamped = k if k < seq_len else seq_len - 1
    lp[diff > k_clamped] = 1
    lp = lp.bool().expand(bs, seq_len, seq_len)
    return lp

gp = global_pattern(2,8)
lp = local_pattern(2,8)
fp = ~(~gp + ~lp)
print(fp)