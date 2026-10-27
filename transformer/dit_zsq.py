import torch
from transformer_zsq import MultiHeadAttention
from vit_zsq import ViTMLP


def timestep_embedding(
    t: torch.Tensor,
    dim: int,
    max_period: int = 10000,
) -> torch.Tensor:
    half = dim // 2
    i = torch.arange(half, device=t.device, dtype=torch.float32)
    freqs = torch.pow(max_period, -i / half)
    freqs = freqs[None, :]
    t = t[:, None]
    angles = t.float() * freqs
    embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim = -1)
    if dim % 2 !=0:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:,-1:])], dim=-1)
    return embedding

class TimestepEmbedder(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        frequency_embedding_dim: int = 256,
    ):
        super().__init__()
        self.frequency_embedding_dim = frequency_embedding_dim
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(frequency_embedding_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
    def forward(self, t):
        t = timestep_embedding(t, self.frequency_embedding_dim)
        t = self.mlp1(t) # [B, dim]
        return t

def modulate(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return (1 + scale[:, None, :]) * x + shift[:, None, :]

class AdaLNZeroConditioner(torch.nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 6 * hidden_dim)
        )
        torch.nn.init.zeros_(self.proj[-1].weight)
        torch.nn.init.zeros_(self.proj[-1].bias)

    def forward(self, condition):
        chunk_conditions = self.proj(condition)
        return chunk_conditions.chunk(6, dim=-1)

class DiTBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        self.norm1 = torch.nn.LayerNorm(hidden_dim, elementwise_affine=False, eps = 1e-6)
        self.attn = MultiHeadAttention(hidden_dim, num_heads)
        self.norm2 = torch.nn.LayerNorm(hidden_dim, elementwise_affine=False, eps = 1e-6)
        self.mlp = ViTMLP(hidden_dim, mlp_dim, dropout)
        self.conditioner = AdaLNZeroConditioner(hidden_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        g_msa, s_msa, a_msa, g_mlp, s_mlp, a_mlp = self.conditioner(condition)
        latent = modulate(self.norm1(x), s_msa, a_msa)
        x = x + g_msa[:, None, :] * self.attn(latent, latent, latent, mask=None)
        latent = modulate(self.norm2(x), s_mlp, a_mlp)
        output = x + g_mlp[:, None, :] * self.mlp(latent)
        return output