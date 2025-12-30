import torch 
from torch import nn
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

naive_attention_module = Attention(dim=64, num_heads=8, qkv_bias=True, qk_norm=False, rope=None, fused_attn=False)
x = torch.randn(2, 16, 64)  # Example input tensor
out = naive_attention_module(x)
def model_wrapper(x_input):
    return naive_attention_module(x_input)
delta_x = torch.randn_like(x) * 1e-3  # Small perturbation
F_avg, F_avg_grad = torch.func.jvp(model_wrapper, (x,), (delta_x,))
print("Output shape:", out.shape)
print("JVP output shape:", F_avg.shape)
print("JVP gradient shape:", F_avg_grad.shape)

