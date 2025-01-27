import torch
import torch.nn as nn


class PatchEmbeddingLayer(nn.Module):
    '''
    Returns a tensor of a flattened patchified image
    with a cls token and positional embeddings
    with shape (batch_size, num_patches + 1, embed_dim)
    '''
    def __init__(self, embed_dim, image_size, image_channels, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.unfold = torch.nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        self.linear = nn.Linear(self.patch_size * self.patch_size * image_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + (image_size**2 // patch_size**2), embed_dim), requires_grad=True)

    def forward(self, x):
        batch_size, c, h, w = x.shape
        assert h % self.patch_size == 0 and w % self.patch_size == 0, 'Image dimensions must be divisible by patch size'
        assert h == w, 'Image must be square'

        x = self.unfold(x).transpose(1, 2)
        x = self.linear(x)
        x = torch.cat((self.cls_token.expand(batch_size, -1, -1), x), dim=1)
        x += self.pos_embedding
        return x
    

class MultiHeadAttention(nn.Module):
    '''
    MultiHeadAttention module
    '''
    def __init__(self, embed_dim, num_heads=12, bias=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.wqkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.wo = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, num_patches, embed_dim = x.shape

        #shape->(3, batch, heads, num_tokens, head_dim)
        qkv = self.wqkv(x).reshape(batch_size, num_patches, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)

        #(batch, heads, num_tokens, head_dim)->(batch, num_tokens, heads, head_dim)->(batch, num_tokens, embed_dim)
        out = (attention @ v).transpose(1, 2).reshape(batch_size, num_patches, embed_dim)
        out = self.wo(out)
        return out
    

class MLPBlock(nn.Module):
    '''
    Simple MLP block
    '''
    def __init__(self, embed_dim, hidden_features, activation=nn.GELU(), dropout=0.):
        super().__init__()

        self.fc1 = nn.Linear(embed_dim, hidden_features)
        self.act = activation
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    

class TransformerEncoder(nn.Module):
    '''
    Transformer Encoder block
    '''
    def __init__(self, embed_dim, num_heads=12, dropout=0., bias=False):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(embed_dim, num_heads, bias=bias)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        out1 = self.norm1(x)
        out2 = self.mha(out1)
        x = x + out2
        out1 = self.norm2(x)
        out2 = self.mlp(out1)
        x = x + out2
        return x
    

class VIT(nn.Module):
    '''
    Vision Transformer model
    '''
    def __init__(self, image_size, image_channels, 
                 patch_size, embed_dim=768, num_heads=12, num_layers=4, num_classes=2, mlp_dropout=0., bias=False):
        super().__init__()
        self.patch_embedding = PatchEmbeddingLayer(embed_dim, image_size, image_channels, patch_size)
        self.transformer_encoder = nn.Sequential(*[TransformerEncoder(embed_dim, num_heads, mlp_dropout, bias) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        x = self.norm(x)
        x = x[:, 0]
        x = self.classifier(x)
        return x