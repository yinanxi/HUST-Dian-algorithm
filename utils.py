import torch
import torch.nn as nn

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        return self.linear(x)

class RowParallelLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        return self.linear(x)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return self.weight * (x / (rms + self.eps))

def apply_rotary_emb(x, freqs_cis):
    # x shape: [batch, seq_len, heads, dim]
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis[:, :x_.shape[1]]
    freqs_cis = freqs_cis.view(1, freqs_cis.shape[1], 1, freqs_cis.shape[-1], 2)

    x_out2 = torch.stack([
        x_[..., 0] * freqs_cis[..., 0] - x_[..., 1] * freqs_cis[..., 1],
        x_[..., 0] * freqs_cis[..., 1] + x_[..., 1] * freqs_cis[..., 0],
    ], -1)

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)

def precompute_freqs_cis(dim, seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len, device=freqs.device).float()
    freqs = torch.einsum("i,j->ij", t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis
