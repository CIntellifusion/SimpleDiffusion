import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import Optional
from timm.models.vision_transformer import PatchEmbed

def modulate(x, shift, scale):
    return x * (1+scale) + shift


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, scale_factor=1.0, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim) * scale_factor)

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
class Attention(nn.Module):
    def __init__(self,dim,num_heads,
                 qkv_bias=False,qk_norm=False,
                 attn_drop=0.0,proj_drop=0.0,
                 rope=None,fused_attn=False):
        super().__init__()
        assert dim % num_heads == 0, "Dimension must be divisible by number of heads."
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, scale_factor=1.0, eps=1e-6)
            self.k_norm = RMSNorm(self.head_dim, scale_factor=1.0, eps=1e-6)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        self.rope = rope
        if rope is not None:
            assert rope.dim == self.head_dim, "RoPE dimension must match head dimension"
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.fused_attn = fused_attn
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each has shape (B, num_heads, N, head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
    
class AdaLayerNormZero(nn.Module):
    """
    An adaptive layer norm with zero initialization. 
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size*3,bias = True),
        )
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False,eps=1e-6)
        self.initialize_weights()
    def initialize_weights(self):
        # zero out 
        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)
    def forward(self, x, c):
        shift,scale,gate = self.modulation(c).chunk(3,dim=-1)
        return modulate(self.norm(x),shift,scale),gate
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class DiTBlock(nn.Module):
    """
    A DiT transformer block with adaptive lary norm zero conditioning. 
    """
    def __init__(self, hidden_size,num_heads, mlp_ratio=None, rope=None,qk_norm=False,
                 **block_kwargs
                 ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_head = num_heads
        self.mlp_ratio = mlp_ratio
        self.rope = rope
        self.norm1 = AdaLayerNormZero(hidden_size)
        self.attn = Attention(dim=hidden_size, num_heads=num_heads,qk_norm=qk_norm)
        self.use_mlp = mlp_ratio is not None 
        self.ff = FeedForward(dim=hidden_size, hidden_dim=int(hidden_size*mlp_ratio))

    def forward(self, x, c):
        x, gate1 = self.norm1(x,c)
        x = self.attn(x) * gate1 + x
        if self.use_mlp:
            x, gate2 = self.norm1(x,c)
            x = self.ff(x) * gate2 + x
        return x
class DiTFinalLayer(nn.Module):
    def __init__(self,hidden_size,out_channels):
        super().__init__()
        self.norm_final = AdaLayerNormZero(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels,bias= True)
        self.initialize_weights()
    def initialize_weights(self):
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    def forward(self,x,c):
        x, _ = self.norm_final(x,c)
        x = self.linear(x)
        return x
    
import numpy as np 
from timm.models.vision_transformer import PatchEmbed, Mlp

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

import math 

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Same as DiT.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        Args:
            t: A 1-D Tensor of N indices, one per batch element. These may be fractional.
            dim: The dimension of the output.
            max_period: Controls the minimum frequency of the embeddings.
        Returns:
            An (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
    
        args = t[:, None].float() * freqs[None]
        
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return embedding
    
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class scm_AdaLoss(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.logvar = nn.Sequential(
            TimestepEmbedder(in_channels),
            nn.Linear(in_channels, 1)
        )
    
        self.init_weight()
        
    def forward(self, t, loss):
        b = t.shape[0]
        logvar = self.logvar(t.flatten()).view(b, -1, 1, 1)
        print(f"factor {1/torch.exp(logvar)} logvar {logvar}")
        weighted = (1/torch.exp(logvar)) * loss + logvar
        return weighted
    
class DiT(nn.Module):
    def __init__(self, 
                 in_channels: int=3,
                 imsize: int=64,
                 patch_size: int=2,
                 pos_emb_type = "learned1d",
                 hidden_size: int=512,
                 depth : int=12,
                 num_heads: int=8,
                 mlp_ratio: float=4.0,
                 learn_sigma : bool = True,
                 qk_norm: bool = False,
                 use_gradient_checkpointing : bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = 3
        self.pos_emb_type = pos_emb_type
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.learn_sigma = learn_sigma
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        self.x_embedder = PatchEmbed(imsize, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size=hidden_size,num_heads=num_heads,mlp_ratio=mlp_ratio,rope=None,qk_norm=qk_norm)
            for _ in range(depth)
        ])
        self.final_layer = DiTFinalLayer(hidden_size=hidden_size,out_channels=self.out_channels*patch_size*patch_size)
    
    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, c):
        # x: nchw 
        x = self.x_embedder(x)+self.pos_embed  # (B, num_patches, hidden_size)
        t = self.t_embedder(c)
        B, N, C = x.shape
        assert N == self.num_patches, f"Input sequence length {N} doesn't match num_patches {self.num_patches}"
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, t)
            else:
                x = block(x, t)
        x = self.final_layer(x, t)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    Same as DiT.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob, disable_label_dropout=False):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        if disable_label_dropout:
            self.dropout_prob = 0

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    
    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class ConditionalDiT(DiT):
    def __init__(self, 
                 in_channels: int=3,
                 pos_emb_type = "learned1d",
                 hidden_size: int=512,
                 depth : int=12,
                 num_heads: int=8,
                 mlp_ratio: float=4.0,
                 learn_sigma : bool = True,
                 use_gradient_checkpointing : bool = False,
                 disable_label_dropout=False,
                 num_classes: int = 1000,
                 class_dropout_prob: float = 0.1):
        super().__init__(in_channels,pos_emb_type,hidden_size,depth,num_heads,mlp_ratio,learn_sigma,use_gradient_checkpointing)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob, disable_label_dropout)
    def forward(self, x, t, cond):
        # x: nchw 
        x = self.x_embedder(x)+self.pos_embed  # (B, num_patches, hidden_size)
        t = self.t_embedder(t)
        cond_emb = self.y_embedder(cond,self.training)
        t = t + cond_emb
        B, N, C = x.shape
        assert N == self.num_patches, f"Input sequence length {N} doesn't match num_patches {self.num_patches}"
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, t)
            else:
                x = block(x, t)
        x = self.final_layer(x, t)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x
if __name__ == "__main__":
    DiT_model = DiT(
        out_channels=3,
        pos_emb_type = "learned1d",
        hidden_size=512,
        depth = 12,
        num_heads=8,
        mlp_ratio=4.0,
        learn_sigma = True,
        use_gradient_checkpointing = False
    )
    x = torch.randn(2, 3, 64, 64)
    c = None
    output = DiT_model(x, c)
    print(output.shape)
