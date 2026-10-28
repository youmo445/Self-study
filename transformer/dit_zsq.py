import torch
from transformer_zsq import MultiHeadAttention
from vit_zsq import ViTMLP, PatchEmbedding


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
        super().__init__()
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
        attn_output, _ = self.attn(latent, latent, latent, mask=None)
        x = x + g_msa[:, None, :] * attn_output
        latent = modulate(self.norm2(x), s_mlp, a_mlp)
        output = x + g_mlp[:, None, :] * self.mlp(latent)
        return output

class FinalLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        patch_size: int,
        out_channels: int,
    ):
        super().__init__()
        self.conditioner = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 2 * hidden_dim)
        )
        torch.nn.init.zeros_(self.conditioner[-1].bias)
        torch.nn.init.zeros_(self.conditioner[-1].weight)
        self.norm_final = torch.nn.LayerNorm(hidden_dim, eps = 1e-6, elementwise_affine=False)
        self.linear = torch.nn.Linear(hidden_dim, patch_size * patch_size * out_channels)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        norm_x = self.norm_final(x)
        scale, shift = self.conditioner(condition).chunk(2, dim=-1)
        x = modulate(norm_x, shift, scale)
        x = self.linear(x)
        return x

def unpatchify(
    x: torch.Tensor,
    patch_size: int,
    out_channels: int,
    grid_height: int,
    grid_width: int,
) -> torch.Tensor:
    B, N, P2C = x.shape
    x = x.reshape(B, N, patch_size, patch_size, out_channels)
    x = x.reshape(B, grid_height, grid_width, patch_size, patch_size, out_channels)
    x = x.permute(0, 5, 1, 3, 2, 4)
    x = x.reshape(B, out_channels, grid_height * patch_size, grid_width * patch_size)
    return x



class LabelEmbedder(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int,
        class_dropout_prob: float=0.1
    ):
        super().__init__()
        self.null_class_id = num_classes
        self.class_dropout_prob = class_dropout_prob
        self.embedding = torch.nn.Embedding(num_classes + 1, hidden_dim)

    def forward(self, label):
        if self.training and self.class_dropout_prob > 0:
            drop_mask = torch.rand(label.shape, device=label.device) < self.class_dropout_prob
            null_label = torch.full_like(label, self.null_class_id)
            label = torch.where(drop_mask, null_label, label)
        return self.embedding(label)


class DiT(torch.nn.Module):
    def __init__(
        self,
        image_height: int,
        image_width: int,
        patch_size: int,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        num_heads: int,
        mlp_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.0,
        class_dropout_prob: float=0.2
    ):
        super().__init__()
        self.time_emb = TimestepEmbedder(hidden_dim)
        self.dit_block = torch.nn.ModuleList(
            [
                DiTBlock(hidden_dim, num_heads, mlp_dim, dropout) 
                for _ in range(num_layers)
            ]
        )
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.grid_height = image_height // patch_size
        self.grid_weight = image_width // patch_size
        self.num_patches = self.grid_weight * self.grid_height
        self.PatchEmbedding = PatchEmbedding(in_channels, patch_size, hidden_dim)
        self.final_layer = FinalLayer(hidden_dim, patch_size, out_channels)
        self.pos_emb = torch.nn.Parameter(torch.randn(1, self.num_patches, hidden_dim) * 0.02)
        self.label_embedding = LabelEmbedder(num_classes, hidden_dim, class_dropout_prob)


    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        x = self.PatchEmbedding(x) # [B, C, H, W]->[B, L, D]
        x = x + self.pos_emb
        time_condition = self.time_emb(t) # [B, D]
        label_condition = self.label_embedding(label)
        condition = time_condition + label_condition
        for block in self.dit_block:
            x = block(x, condition)
        x = self.final_layer(x, condition)
        x = unpatchify(x, self.patch_size, self.out_channels, self.grid_height, self.grid_weight)
        return x