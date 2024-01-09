import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class Attention(nn.Module):
    def __init__(self, dim=512, heads=8):
        super().__init__()
        self.wq, self.wl, self.wv, self.wo = [nn.Linear(dim, heads * dim) for _ in range(4)]
    
    def forward(self, x):
        pass 

        
class FF(nn.Module):
    def __init__(self, dim, hdim, multiple_of):
        super().__init__()
        hidden_dim = int(2 * hdim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.w3 = nn.Linear(dim, hidden_dim)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    

class TransBlock(nn.Module):
    def __init__(self,dim, heads, multiple_of, normeps):
        self.heads = 8
        self.dim = 512
        self.head_dim = 64
        self.attention = Attention(dim, heads)
        self.ff = FF(dim, 4*dim, multiple_of)
        self.a_norm = RMSNorm(dim, normeps)
        self.f_norm = RMSNorm(dim, normeps)
    
    def forward(self, x, mask, freq):
        pass

