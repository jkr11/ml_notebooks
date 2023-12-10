import torch
import torch.nn as nn
import torch.nn.functional as F 

class PreNorn():
  def __init__(self, fn, dim):
    super(self).__init__()
    self.norm = nn.LayerNorm(dim)
    self.fn = fn

  def forward(self, x, **kwargs):
    return self.fn(self.norm(x), **kwargs)

    
