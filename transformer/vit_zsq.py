import torch
from transformer_zsq import MultiHeadAttention
def patchify(image, patch_size):
    B, C, H, W = image.shape
    P = patch_size
    Hp = H // P
    Wp = W // P
    image = image.reshape(B, C, Hp, P, Wp, P)
    image = image.permute(0, 2, 4, 1, 3, 5)
    image = image.reshape(B, Hp * Wp, C * P * P)
    return image

class PatchEmbedding(torch.nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.patch_dim = in_channels * patch_size * patch_size
        self.fc1 = torch.nn.Linear(self.patch_dim, embed_dim)
    def forward(self, image):
        image = patchify(image, self.patch_size) # [B,C,H,W] -> [B, Hp*Wp, CPP]
        image_emb = self.fc1(image)
        return image_emb

class ViTEmbedding(torch.nn.Module):
    def __init__(self, H, W, in_channels, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.patch_dim = in_channels * patch_size * patch_size
        self.num_patched = (H // patch_size) * (W // patch_size)
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_dim)
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embedding = torch.nn.Parameter(torch.randn(1, self.num_patched + 1, embed_dim) * 0.02)
    def forward(self, image):
        image = self.patch_embedding(image) # [B, T, d]
        B = image.shape[0]
        cls_token = self.cls_token.expand(B, -1, -1)
        image = torch.cat([cls_token, image], dim = 1)
        image = image + self.pos_embedding
        return image

class ViTMLP(torch.nn.Module):
    def __init__(self, embed_dim, mlp_dim, dropout=0.0):
        super().__init__()
        self.fc1 = torch.nn.Linear(embed_dim, mlp_dim)
        self.gelu = torch.nn.GELU()
        self.dropout1 = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(mlp_dim, embed_dim)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(self, x):
        # [B,N+1,D] -> [B,N+1,Dff]
        x = self.fc1(x)
        # GELU、Dropout
        x = self.gelu(x)
        x = self.dropout1(x)
        # [B,N+1,Dff] -> [B,N+1,D]
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class ViTEncoderBlock(torch.nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_dim,
        dropout=0.0,
    ):
        super().__init__()

        self.ln1 = torch.nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.ln2 = torch.nn.LayerNorm(embed_dim)
        self.mlp = ViTMLP(embed_dim, mlp_dim, dropout)

    def forward(self, x):
        # 第一部分：Pre-LN Self-Attention
        # x: [B,N+1,D]
        normalized_x = self.ln1(x)
        attention_output, attention_weights = self.attn(normalized_x, normalized_x, normalized_x, mask = None)
        x = x + self.dropout1(attention_output)
        # 第二部分：Pre-LN MLP
        normalized_x = self.ln2(x)
        mlp_output = self.mlp(normalized_x)
        x = x + mlp_output
        return x, attention_weights

class ViTEncoder(torch.nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_dim,
        num_layers,
        dropout=0.0,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            ViTEncoderBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout)
            for _ in range(num_layers)
        )
        self.fn_ln = torch.nn.LayerNorm(embed_dim)
    def forward(self, x):
        attention_weights_list = []
        for layer in self.layers:
            x, attention_weights = layer(x)
            attention_weights_list.append(attention_weights)
        x = self.fn_ln(x)
        return x

class VisionTransformer(torch.nn.Module):
    def __init__(
        self,
        H,
        W,
        in_channels,
        patch_size,
        embed_dim,
        num_heads,
        mlp_dim,
        num_layers,
        num_classes,
        dropout=0.0,
    ):
        super().__init__()

        self.embedding = ViTEmbedding(H, W, in_channels, patch_size, embed_dim)
        self.embedding_dropout = torch.nn.Dropout(dropout)
        self.encoder = ViTEncoder(embed_dim, num_heads, mlp_dim, num_layers, dropout)
        self.classifier = torch.nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        x = self.encoder(x)
        cls_output = x[:,0,:]
        logits = self.classifier(cls_output)
        return logits