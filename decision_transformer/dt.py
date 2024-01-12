import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
  def __init__(self, embedding_dim,  seq_len, num_heads,r_dropout, dropout):
    super().__init__()
    self.n1 = nn.LayerNorm(embedding_dim)
    self.n2 = nn.LayerNorm(embedding_dim)
    self.dropout = nn.Dropout(r_dropout)

    self.attn = nn.MultiHeadAttention(
      embedding_dim, num_heads, dropout
    )
    self.ff = nn.Sequential(
      nn.Linear(embedding_dim, 4 * embedding_dim),
      nn.GELU(),
      nn.Linear(4 * embedding_dim, embedding_dim),
      nn.Dropout(r_dropout),
    )
    self.causal_mask = torch.tril(torch.ones(seq_len, seq_len)).to(bool)
    self.seq_len = seq_len


  def forward(self, x, mask=None):
    c_mask = self.causal_mask(: x.shape[1], : x.shape[1])
    norm_x = self.n1(x)
    att_out = self.attn(
      q = norm_x,
      key = norm_x,
      value = norm_x,
      attn_mask = x_mask,
      key_padding_mask = mask,
      need_weights = False,
    )[0]
    x = x + self.drop(att_out)
    x = x + self.ff(self.norm2(x))
    return x

class Transformer(nn.Module):
  def __init__(self, statedim, adim, seqlen, eplen, emdim, nlayer, nheads, attndrop, resdrop, enbdrop, maxact):    super().__init__()
    
