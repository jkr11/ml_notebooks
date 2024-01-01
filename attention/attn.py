from torch import Tensor
import torch.nn as nn
import torch
import torch.nn.functional as f
import math
from typing import Tuple, List, Sequence, Optional

def sdpa(query:Tensor, key:Tensor, value:Tensor) -> Tensor:
  score = query @ key.transpose(-2,-1)
  d_k = math.sqrt(query.size(-1))
  return f.softmax(score / d_k, dim = 1) @ value

class AttnHead(nn.Module):
  def __init__(self, dim:int, dim_q:int, dim_k:int, dim_v:int):
    super().__init__(self, AttnHead)
    dim_v = dim_k
    self.q = nn.Linear(dim, dim_q)
    self.k = nn.Linear(dim, dim_k)
    self.v = nn.Linear(dim, dim_v)
  
  def forward(self, query:Tensor, key:Tensor, value:Tensor) -> Tensor:
    return sdpa(
      self.q(query), self.k(key), self.v(value)
    )

class MHA(nn.Module):
  def __init__(self, dim:int, dim_q:int, dim_k:int, dim_v:int, num_heads:int = 8):
    super().__init__(self, MHA)
    self.heads = nn.ModuleList(
      [AttnHead(dim, dim_q, dim_k, dim_v) for _ in range(num_heads)]
    )
    self.linear = nn.Linear(num_heads * dim_k, dim)

  def forward(self, query:Tensor, key:Tensor, value:Tensor) -> Tensor:
    return self.linear(
      torch.cat([head(query, key, value) for head in self.heads], dim=-1)
    )
