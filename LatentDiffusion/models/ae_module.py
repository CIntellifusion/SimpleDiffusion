import torch 
from torch import nn 
import math 
import numpy as np 
"""
    A simple implementation of Gaussian MLP Encoder and Decoder
"""

class SimpleEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(SimpleEncoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear(hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        
    def forward(self, x):
        h_       = self.LeakyReLU(self.FC_input(x))
        h_       = self.LeakyReLU(self.FC_input2(h_))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     # encoder produces mean and log of variance 
                                                       #             (i.e., parateters of simple tractable normal distribution "q"
        
        return mean, log_var
    
    
class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(SimpleDecoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h     = self.LeakyReLU(self.FC_hidden(x))
        h     = self.LeakyReLU(self.FC_hidden2(h))
        
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat
    

"""
    A more complex implementation of Resnet Encoder with Attention
    reference: lvdm/moddules/ae_modules.py
"""
def Normalize(in_channels,num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-5)

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

class AttnBlock(nn.Module):
    # attention based on conv net 
    def __init__(self,in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        # convlutional layers of kernelsize 1 ,stride 1 , padding 0 means 
        # a linear layer in the shape of [H,W]
        self.q = nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0) # also called FFN 
        
    def forward(self,x):
        h_ = x
        h_ = self.norm(h_)# would this be different to h_=self.norm(x)?
        
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        
        # compute scale dot product attention 
        b,c,h,w = q.shape
        q = q.reshape(b,c,-1).permute(0,2,1) # b,h*w,c
        k = k.reshape(b,c,-1) # b,c,h*w 
        w_ = torch.bmm(q,k)# b,h*w,c @ b,c,h*w -> b,h*w,h*w
        w_ = w_ * (int(c)**(-0.5)) # why? 
        w_ = torch.nn.functional.softmax(w_, dim=2)#b,h*w,h*w 
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        
        v = v.reshape(b,c,-1)
        h_ = torch.bmm(v,w_) 
        h_ = h_.reshape(b,c,h,w)
        
        h_ = self.proj_out(h_)
        return h_ + x 
        

def make_attn(in_channels,attn_type="vanilla"):
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type=="none":
        return nn.Identity(in_channels)
    else:
        raise NotImplementedError(f"Attention type {attn_type} is not implemented")

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        self.in_channels = in_channels
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)
    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x
    
class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        self.in_channels = in_channels
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x
def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class Encoder(nn.Module):
    # downsample blocks : resblock+attention
    # mid : resblock+attention
    def __init__(self, *, ch,  ch_mult=(1,2,4,8), num_res_blocks,
                attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                **ignore_kwargs):
        super().__init__()
        self.ch = ch 
        self.temb_ch = 0 
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution 
        self.in_channels = in_channels
        
        # down sampling 
        self.conv_in = nn.Conv2d(in_channels,
                                 self.ch,
                                 kernel_size = 3,
                                 stride = 1,
                                 padding=1)
        
        cur_res = resolution 
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            attn = nn.ModuleList()
            block = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                          out_channels=block_out,
                                          temb_channels=self.temb_ch,
                                          dropout=dropout))
                block_in = block_out 
                if cur_res in attn_resolutions:
                    attn.append(make_attn(block_in,attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in,resamp_with_conv)
                cur_res = cur_res // 2
            self.down.append(down)
        # middle 
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in,attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.norm_out = Normalize(block_in)
        self.conv_mean = nn.Conv2d(block_in,
                                  2*z_channels if double_z else z_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)
        self.conv_var = nn.Conv2d(block_in,
                                  2*z_channels if double_z else z_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)
    def forward(self,x):
        temb = None 
        
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            h = hs[-1]
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h,temb)
                if len(self.down[i_level].attn)>0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions-1:
                h = self.down[i_level].downsample(h)
            hs.append(h)
        # mid
        h = hs[-1]
        h = self.mid.block_1(h,temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h,temb)
        
        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        mean = self.conv_mean(h)
        log_var = self.conv_var(h)
        return mean,log_var 
    
class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        
        
        # compute 
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("AE working on z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # print(f'decoder-input={z.shape}')
        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)
        # print(f'decoder-conv in feat={h.shape}')

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        # print(f'decoder-mid feat={h.shape}')

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
                # print(f'decoder-up feat={h.shape}')
            if i_level != 0:
                h = self.up[i_level].upsample(h)
                # print(f'decoder-upsample feat={h.shape}')

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        # print(f'decoder-conv_out feat={h.shape}')
        if self.tanh_out:
            h = torch.tanh(h)
        else:
            h = torch.sigmoid(h)
        return h
    def sample(self,n_sample):
        z = torch.randn(n_sample,*self.z_shape[1:])
        return self.forward(z)


if __name__ == '__main__':
    resolution = 64
    in_channels = 3
    encoder = Encoder(ch=256,
                      resolution=resolution,
                      in_channels=in_channels,
                      ch_mult=(1,2,4,8),
                      num_res_blocks=2,
                      attn_resolutions=(16,),
                      dropout=0.0,
                      resamp_with_conv=True,
                      z_channels=128,
                      double_z=True,
                      use_linear_attn=False,
                      use_checkpoint=False).to("cuda")
    decoder = Decoder(ch=256,
                      out_ch=3,
                      resolution=resolution,
                      in_channels=in_channels,
                      ch_mult=(1,2,4,8),
                      num_res_blocks=2,
                      attn_resolutions=(16,),
                      dropout=0.0,
                      resamp_with_conv=True,
                      z_channels=256,
                      give_pre_end=False,   
                      tanh_out=False,
                      use_linear_attn=False,
                      use_checkpoint=False).to("cuda")
    x = torch.randn(1,3,64,64).to("cuda")
    z = torch.randn(1,256,8,8).to("cuda")
    print(encoder(x)[0].shape)
    print(encoder(x)[1].shape)
    print(decoder(z).shape)