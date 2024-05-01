"""
autor: haoyu
date: 20240501
an simplified unconditional diffusion for image generation
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
import os , cv2 
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping, Callback
import numpy as np
""" 
an simple overview of the code structure
for a diffusion generator , we need to define: 
1. denoise network 
2. data
3. DDIM scheduler
4. trainer

during the training process:
1. the unet receive timestep t and image x_t, and predict x_{t+1}
2. the diffusion scheduler receive x_t and x_{t+1} and predict q(t,x)

during the inference stage:
1. start from a noise or an input image as x_t 
2. predict x_t-1
3. update x_t = x_t-1
4. repeat until reach the target timestep
"""


### Unet 2D denoise network 
### borrowed from https://github.com/SingleZombie/DL-Demos/blob/master/dldemos/ddim/network.py

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        return nn.GroupNorm(32, shape[0])


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
                 image_shape,
                 channels=[10, 20, 40, 80],
                 pe_dim=10,
                 with_attns=False,
                 norm_type='ln',
                 activation_type='silu'):
        super().__init__()
        C, H, W = image_shape
        layers = len(channels)
        Hs = [H]
        Ws = [W]
        cH = H
        cW = W
        self.NUM_RES_BLOCK = 2
        for _ in range(layers - 1):
            cH //= 2
            cW //= 2
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

        self.conv_in = nn.Conv2d(C, channels[0], 3, 1, 1)
        self.conv_out = nn.Conv2d(prev_channel, C, 3, 1, 1)
        self.activation = create_activation(activation_type)

    def forward(self, x, t):
        t = self.pe(t)
        pe = self.pe_linears(t)

        x = self.conv_in(x)

        encoder_outs = []
        for encoder, down in zip(self.encoders, self.downs):
            tmp_outs = []
            for index in range(self.NUM_RES_BLOCK):
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

# Example usage:
# Create a UNet model with 3 levels of downsampling, 1 middle block, and 3 levels of upsampling
# Embed time_step with a maximum of 10 time steps and 64-dimensional embedding

unet = UNet(n_steps=1000, image_shape=[3,128,128])
time_step = torch.tensor([5])  # Example time_step value (5th time step)
input_tensor = torch.randn(1, 3, 128, 128)  # Example input tensor
output = unet(input_tensor, time_step)
print(output.shape)  # Check output shape

### data
import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./", batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        # This method is intended for dataset downloading and preparation
        # We will download the MNIST dataset here (only called on 1 GPU in distributed training)
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None,transform=None):
        # This method is called on every GPU in the distributed setup and should split the data
        if transform is None : 
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        
        if stage == 'fit' or stage is None:
            self.train_dataset = MNIST(self.data_dir, train=True, transform=transform)
            self.val_dataset = MNIST(self.data_dir, train=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

# Example usage:
# data_module = MNISTDataModule(data_dir="./data", batch_size=64)


### DDIM scheduler
class DDPM(nn.Module):
    def __init__(self, min_beta, max_beta, N):
        super(DDPM, self).__init__()
        # linearly interpolate between min_beta and max_beta for N steps
        betas = torch.linspace(min_beta, max_beta, N)
        alphas = 1 - betas
        alpha_bars = alphas.cumprod(dim=0)# cumulative product of alphas in reverse order
        alpha_bars_prev = torch.cat(
            (torch.tensor([1]), alpha_bars[:-1]))# add 1 at the beginning
        self.register_buffer("alphas",alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)
        self.register_buffer("betas", betas)
        self.N = N
        
    def sampling_forward(self,x,t,eps=None):
        alpha_bar = self.alpha_bars[t].reshape(-1,1,1,1)
        if eps is None:
            eps = torch.randn_like(x)
        result = eps * torch.sqrt(1-alpha_bar) + torch.sqrt(alpha_bar)*x
        return result
        
        
    def sampling_backward(self, image_or_shape,net,device="cuda",simple_var=True):
        if isinstance(image_or_shape,torch.Tensor):
            x = image_or_shape
        else:
            x = torch.randn(image_or_shape,device=device)
        for t in range(self.N-1,-1,-1):
            x = self.sampling_step(net, x, t, simple_var)
        return x
        
    def sampling_step(self,net,x_t, t,simple_var):
        bs = x_t.shape[0]
        t_tensor = t*torch.ones(bs,dtype=torch.long,device=x_t.device).reshape(-1,1)
        if t== 0:
            noise = 0 
        if simple_var:
            var = self.betas[t]
        else:
            var = self.betas[t] * self.alpha_bars[t] * (1-self.alpha_bars_prev[t])/(1-self.alpha_bars[t-1])
        
        noise = torch.rand_like(x_t) * torch.sqrt(var)
        eps = net(x_t,t_tensor)
        mean = (x_t -(1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) *eps) / torch.sqrt(self.alphas[t])
        x_t_prev = mean + noise
        return x_t_prev

### trainer 

class LightningImageDenoiser(pl.LightningModule):
    def __init__(self, batch_size=64, lr=0.001,N=1000):
        super(LightningImageDenoiser, self).__init__()
        self.save_hyperparameters()  # Save hyperparameters for logging
        self.model = UNet(n_steps=1000, image_shape=[1,32,32])
        self.ddpm = DDPM(min_beta=0.0001,max_beta=0.02,N=N)
        self.criterion = nn.MSELoss()
        self.N = N 
        self.batch_size = batch_size
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, _ = batch
        # print(images.shape)
        bs = images.shape[0]
        t = torch.randint(0,self.N,(bs,),device = images.device)
        eps = torch.rand_like(images,device=images.device)
        x_t = self.ddpm.sampling_forward(images, t, eps)
        eps_theta = self.model(x_t, t.reshape(bs, 1))
        loss = self.criterion(eps,eps_theta)
        self.log('train_loss', loss)
        return loss

    def train_dataloader(self):
        transform = transforms.Compose([
            # transforms.RandomCrop(size=(28, 28)),
            transforms.Resize((32, 32)),  # Resize images to (128, 128)
            transforms.ToTensor(),           # Convert images to tensors
            transforms.Normalize((0.5), (0.5))  # Normalize images
        ])
        
        data_module = MNISTDataModule(data_dir="./data", batch_size=self.batch_size)
        data_module.prepare_data()
        data_module.setup(transform=transform)
        return data_module.train_dataloader()
        dataset = ImageFolder(root='./fake_image_folder', transform=transform)
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True)

def sample_image(ddpm,net,output_dir,image_shape,n_sample,device="cuda",simple_var=True):
    max_batch_size = 32
    net.to(device)
    net.eval()
    with torch.no_grad():
        for i in range(0,n_sample,max_batch_size):
            shape = (max_batch_size, *image_shape)
            imgs = ddpm.sampling_backward(shape,
                                        net,
                                        device=device,
                                        simple_var=simple_var,
                                        ).detach().cpu()
            imgs = (imgs + 1) / 2 * 255
            print(imgs.shape)
            imgs = imgs.clamp(0, 255).to(torch.uint8).permute(0,2,3,1).numpy()
            # output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            for j, img in enumerate(imgs):
                cv2.imwrite(f'{output_dir}/{i*max_batch_size+j}.jpg', img)
### args and main function 

if __name__ == "__main__":
    is_train = True
    if is_train:
        model = LightningImageDenoiser( batch_size = 512)

        # 设置保存 checkpoint 的回调函数
        checkpoint_callback = ModelCheckpoint(
            dirpath="./checkpoints",  # 保存 checkpoint 的目录
            filename="model-{epoch:02d}-{val_loss:.2f}",  # checkpoint 文件名格式
            monitor="train_loss",  # 监控的指标，这里使用验证集损失
            mode="min",  # 指定监控模式为最小化验证集损失
            save_top_k=3,  # 保存最好的 3 个 checkpoint
            verbose=True
        )
        
        # 创建 PyTorch Lightning Trainer
        pretrain_path = "./checkpoints/model-epoch=185-val_loss=0.00.ckpt"
        trainer = pl.Trainer(
            devices="3,",                    # 使用一块 GPU 进行训练
            max_epochs=400,             # 最大训练 epoch 数
            logger=pl.loggers.TensorBoardLogger("logs/", name="mnist_example"),
            # progress_bar_refresh_rate=20,  # 进度条刷新频率
            callbacks=[checkpoint_callback],  # 注册 checkpoint 回调函数
        )

        # 启动模型训练
        trainer.fit(model,ckpt_path = pretrain_path)
    else:
        paths = ["/data2/wuhaoyu/SimpleDiffusion/UnconditionalDiffusion/checkpoints/model-epoch=35-val_loss=0.00.ckpt",
        "/data2/wuhaoyu/SimpleDiffusion/UnconditionalDiffusion/checkpoints/model-epoch=57-val_loss=0.00.ckpt",
        "/data2/wuhaoyu/SimpleDiffusion/UnconditionalDiffusion/checkpoints/model-epoch=15-val_loss=0.00.ckpt"]
        paths = os.listdir("./checkpoints")
        paths = [os.path.join("./checkpoints",i) for i in paths]
        for path in paths:
            ckpt = os.path.basename(path).replace(".ckpt","")
            image_shape=[1,32,32]
            net = UNet(n_steps=1000,image_shape=image_shape )
            net.load_state_dict(torch.load(path),strict=False)
            ddpm = DDPM(min_beta=0.0001,max_beta=0.02,N=1000)
            sample_image(ddpm,
                    net,
                    f'./sample/{ckpt}',
                    image_shape=image_shape,
                    n_sample=32,
                    device="cuda:3",
                    )