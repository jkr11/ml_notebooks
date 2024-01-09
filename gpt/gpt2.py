import torch
import torch.nn as nn

from typing import Optional, Union
from tqdm import trange
from torch import Tensor
import numpy as np


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.to_qkv = nn.Linear(dim, 3*dim, bias=True) # attn = dim -> q|k|v
        self.proj = nn.Linear(dim, dim, bias=True)
        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads
    
    def forward(self, x, startpos, mask:Optional[Tensor]) -> Tensor:
        if mask is not None or startpos.val == 0:
            startpos = startpos.val
        
        xqkv = self.to_qkv(x)
        
