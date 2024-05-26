### Unet 2D denoise network 
### borrowed from https://github.com/SingleZombie/DL-Demos/blob/master/dldemos/ddim/network.py

import torch 
from torch import nn
from torch.nn import functional as F 

class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()

        # Assume d_model is an even number for convenience
        assert d_model % 2 == 0

        pe = torch.zeros(max_seq_len, d_model)
        i_seq = torch.linspace(0, max_seq_len - 1, max_seq_len)
        j_seq = torch.linspace(0, d_model - 2, d_model // 2)
        pos, two_i = torch.meshgrid(i_seq, j_seq)
        pe_2i = torch.sin(pos / 10000**(two_i / d_model))
        pe_2i_1 = torch.cos(pos / 10000**(two_i / d_model))
        pe = torch.stack((pe_2i, pe_2i_1), 2).reshape(max_seq_len, d_model)

        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.embedding.weight.data = pe
        self.embedding.requires_grad_(False)

    def forward(self, t):
        return self.embedding(t)


def create_norm(type, shape):
    if type == 'ln':
        return nn.LayerNorm(shape)
    elif type == 'gn':
        return nn.GroupNorm(16, shape[0])


def create_activation(type):
    if type == 'relu':
        return nn.ReLU()
    elif type == 'silu':
        return nn.SiLU()


class ResBlock(nn.Module):
    def __init__(self,
                 shape,
                 in_c,
                 out_c,
                 time_c,
                 norm_type='ln',
                 activation_type='silu'):
        super().__init__()
        self.norm1 = create_norm(norm_type, shape)
        self.norm2 = create_norm(norm_type, (out_c, *shape[1:]))
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.time_layer = nn.Linear(time_c, out_c)
        self.activation = create_activation(activation_type)
        if in_c == out_c:
            self.residual_conv = nn.Identity()
        else:
            self.residual_conv = nn.Conv2d(in_c, out_c, 1)

    def forward(self, x, t):
        n = t.shape[0]
        out = self.activation(self.norm1(x))
        out = self.conv1(out)

        t = self.activation(t)
        t = self.time_layer(t).reshape(n, -1, 1, 1)
        out = out + t

        out = self.activation(self.norm2(out))
        out = self.conv2(out)
        out += self.residual_conv(x)
        return out


class SelfAttentionBlock(nn.Module):

    def __init__(self, shape, dim, norm_type='ln'):
        super().__init__()

        self.norm = create_norm(norm_type, shape)
        self.q = nn.Conv2d(dim, dim, 1)
        self.k = nn.Conv2d(dim, dim, 1)
        self.v = nn.Conv2d(dim, dim, 1)
        self.out = nn.Conv2d(dim, dim, 1)

    def forward(self, x):

        n, c, h, w = x.shape

        norm_x = self.norm(x)
        q = self.q(norm_x)
        k = self.k(norm_x)
        v = self.v(norm_x)

        # n c h w -> n h*w c
        q = q.reshape(n, c, h * w)
        q = q.permute(0, 2, 1)
        # n c h w -> n c h*w
        k = k.reshape(n, c, h * w)

        qk = torch.bmm(q, k) / c**0.5
        qk = torch.softmax(qk, -1)
        # Now qk: [n, h*w, h*w]

        qk = qk.permute(0, 2, 1)
        v = v.reshape(n, c, h * w)
        res = torch.bmm(v, qk)
        res = res.reshape(n, c, h, w)
        res = self.out(res)

        return x + res

# multihead_attention = nn.MultiheadAttention(embed_dim, num_heads)

# # 使用 MultiheadAttention 层处理输入张量
# output, attention_weights = multihead_attention(x, x, x)

class ResAttnBlock(nn.Module):

    def __init__(self,
                 shape,
                 in_c,
                 out_c,
                 time_c,
                 with_attn,
                 norm_type='ln',
                 activation_type='silu'):
        super().__init__()
        self.res_block = ResBlock(shape, in_c, out_c, time_c, norm_type,
                                  activation_type)
        if with_attn:
            self.attn_block = SelfAttentionBlock((out_c, shape[1], shape[2]),
                                                 out_c, norm_type)
        else:
            self.attn_block = nn.Identity()

    def forward(self, x, t):
        x = self.res_block(x, t)
        x = self.attn_block(x)
        return x


class ResAttnBlockMid(nn.Module):

    def __init__(self,
                 shape,
                 in_c,
                 out_c,
                 time_c,
                 with_attn,
                 norm_type='ln',
                 activation_type='silu'):
        super().__init__()
        self.res_block1 = ResBlock(shape, in_c, out_c, time_c, norm_type,
                                   activation_type)
        self.res_block2 = ResBlock((out_c, shape[1], shape[2]), out_c, out_c,
                                   time_c, norm_type, activation_type)
        if with_attn:
            self.attn_block = SelfAttentionBlock((out_c, shape[1], shape[2]),
                                                 out_c, norm_type)
        else:
            self.attn_block = nn.Identity()

    def forward(self, x, t):
        x = self.res_block1(x, t)
        x = self.attn_block(x)
        x = self.res_block2(x, t)
        return x


class UNet(nn.Module):

    def __init__(self,
                 n_steps,
                 latent_shape,
                 channels=[10, 20, 40, 80],
                 pe_dim=10,
                 with_attns=False,
                 norm_type='ln',
                 activation_type='silu',
                 num_res_block=2):
        super().__init__()
        C, H, W = latent_shape
        layers = len(channels)
        Hs = [H]
        Ws = [W]
        cH = H
        cW = W
        self.NUM_RES_BLOCK = num_res_block
        for _ in range(layers - 1):
            cH //= 2
            cW //= 2
            if cH==0 or cW ==0:
                raise ValueError(f"invalid channel config, to many down sample layers {channels}, and feature will be {cH}*{cW}")
            Hs.append(cH)
            Ws.append(cW)
        if isinstance(with_attns, bool):
            with_attns = [with_attns] * layers

        self.pe = PositionalEncoding(n_steps, pe_dim)
        time_c = 4 * channels[0]
        self.pe_linears = nn.Sequential(nn.Linear(pe_dim, time_c),
                                        create_activation(activation_type),
                                        nn.Linear(time_c, time_c))
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        prev_channel = channels[0]
        for channel, cH, cW, with_attn in zip(channels[0:-1], Hs[0:-1],
                                              Ws[0:-1], with_attns[0:-1]):
            encoder_layer = nn.ModuleList()
            for index in range(self.NUM_RES_BLOCK):
                if index == 0:
                    modules = ResAttnBlock(
                        (prev_channel, cH, cW), prev_channel, channel, time_c,
                        with_attn, norm_type, activation_type)
                else:
                    modules = ResAttnBlock((channel, cH, cW), channel, channel,
                                           time_c, with_attn, norm_type,
                                           activation_type)
                encoder_layer.append(modules)
            self.encoders.append(encoder_layer)
            self.downs.append(nn.Conv2d(channel, channel, 2, 2))
            prev_channel = channel

        cH = Hs[-1]
        cW = Ws[-1]
        channel = channels[-1]
        self.mid = ResAttnBlockMid((prev_channel, cH, cW), prev_channel,
                                   channel, time_c, with_attns[-1], norm_type,
                                   activation_type)

        prev_channel = channel
        for channel, cH, cW, with_attn in zip(channels[-2::-1], Hs[-2::-1],
                                              Ws[-2::-1], with_attns[-2::-1]):
            self.ups.append(nn.ConvTranspose2d(prev_channel, channel, 2, 2))

            decoder_layer = nn.ModuleList()
            for _ in range(self.NUM_RES_BLOCK):
                modules = ResAttnBlock((2 * channel, cH, cW), 2 * channel,
                                       channel, time_c, with_attn, norm_type,
                                       activation_type)

                decoder_layer.append(modules)

            self.decoders.append(decoder_layer)

            prev_channel = channel

        self.conv_in = nn.Conv2d(C, channels[0], kernel_size = 3, stride = 1, padding = 1)
        self.conv_out = nn.Conv2d(prev_channel, C, kernel_size = 3, stride = 1, padding = 1)
        self.activation = create_activation(activation_type)

    def forward(self, x, t):
        t = self.pe(t)
        pe = self.pe_linears(t)
        # print("in unet x",x.shape)
        x = self.conv_in(x)
        # print("in unet conv_in",x.shape)
        encoder_outs = []
        for encoder, down, encidx in zip(self.encoders, self.downs,range(len(self.encoders))):
            tmp_outs = []
            for index in range(self.NUM_RES_BLOCK):
                # print(f"encoder {encidx} {index}",x.shape)
                x = encoder[index](x, pe)
                tmp_outs.append(x)
            tmp_outs = list(reversed(tmp_outs))
            encoder_outs.append(tmp_outs)
            x = down(x)
        x = self.mid(x, pe)
        for decoder, up, encoder_out in zip(self.decoders, self.ups,
                                            encoder_outs[::-1]):
            x = up(x)

            # If input H/W is not even
            pad_x = encoder_out[0].shape[2] - x.shape[2]
            pad_y = encoder_out[0].shape[3] - x.shape[3]
            x = F.pad(x, (pad_x // 2, pad_x - pad_x // 2, pad_y // 2,
                          pad_y - pad_y // 2))

            for index in range(self.NUM_RES_BLOCK):
                c_encoder_out = encoder_out[index]
                x = torch.cat((c_encoder_out, x), dim=1)
                x = decoder[index](x, pe)
        x = self.conv_out(self.activation(x))
        return x

if __name__=="__main__":
    ### unit tests : python unet.py 
    ### for large unet 
    latent_shape = [32,8,8]
    bs = 2
    latents = torch.randn(bs,*latent_shape)
    unet_config = {"channels":[64,128,64,32],
                   "pe_dim":100,
                   "with_attns":True,
                   "norm_type":"ln"
                   }
    print("[unit test 3]:  channel test:")
    model = UNet(1000,latent_shape,**unet_config)
    ### for latent test 
    latent_shape = [32,8,8]
    bs = 2
    latents = torch.randn(bs,*latent_shape)
    t = torch.randint(0,1000,(bs,),device = latents.device)
    eps = torch.randn_like(latents,device=latents.device)
    output = model(latents,t)
    print("unet test",output.shape)
    exit()
    print("[unit test 2]: latent shape test:")
    model = UNet(1000,latent_shape)
    t = torch.randint(0,1000,(bs,),device = latents.device)
    eps = torch.randn_like(latents,device=latents.device)
    ### unit tests : python unet.py 
    image_shape = [3,64,64]
    out_dim = 3
    print("[unit test 1]: attention test:")
    attnblock = SelfAttentionBlock(
        shape=image_shape,
        dim=out_dim
    )
    
    x = torch.randn([1,*image_shape])
    output = attnblock(x)
    print("input:",x.shape,x.mean(),x.std(),x.max())
    print("output:",output.shape,output.mean(),output.std(),output.max())
    