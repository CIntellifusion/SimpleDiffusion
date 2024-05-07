"""
autor: haoyu
date: 20240501-0506
an simplified unconditional diffusion for image generation
"""
import os , cv2 ,argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from datasets import load_dataset
from torchvision.datasets import MNIST
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping, Callback
import numpy as np

from torch.optim.lr_scheduler import ReduceLROnPlateau


""" 
an simple overview of the code structure
for a diffusion generator , we need to define: 
1. denoise network 
2. mnist data and 
3. DDIM scheduler
4. pytorch lightning trainer

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

# unet = UNet(n_steps=1000, image_shape=[3,128,128])
# time_step = torch.tensor([5])  # Example time_step value (5th time step)
# input_tensor = torch.randn(1, 3, 128, 128)  # Example input tensor
# output = unet(input_tensor, time_step)
# print(output.shape)  # Check output shape

### data
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

    def train_dataloader(self,num_workers):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=num_workers)

    def val_dataloader(self,num_workers):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,num_workers=num_workers)

class CelebDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64,num_workers=63):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def split_dataset(self, dataset, split_ratio=0.2):
        """
        Divides the dataset into training and validation sets.

        Args:
        - dataset (datasets.Dataset): The dataset to be divided
        - split_ratio (float): The proportion of the validation set, default is 0.2

        Returns:
        - train_dataset (datasets.Dataset): The divided training set
        - val_dataset (datasets.Dataset): The divided validation set
        """
        num_val_samples = int(len(dataset) * split_ratio)

        val_dataset = dataset.shuffle(seed=42).select(range(num_val_samples))
        train_dataset = dataset.shuffle(seed=42).select(range(num_val_samples, len(dataset)))

        return train_dataset, val_dataset

    def prepare_data(self):
        self.dataset = load_dataset('nielsr/CelebA-faces')

    def setup(self, stage=None, transform=None):
        if stage == 'fit' or stage is None:
            self.train_dataset, self.val_dataset = self.split_dataset(self.dataset['train'], split_ratio=0.2)

    @staticmethod
    def collate_fn(batch):
        # for example in batch:
        #     image = example['image']
        #     image.save("/home/haoyu/research/simplemodels/cache/test.jpg")
        transform = transforms.Compose([
            transforms.Resize((64, 64)),  # Resize images to (128, 128)
            transforms.ToTensor(),           # Convert images to tensors
            # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))  # Normalize images
            transforms.Lambda(lambda x: (x - 0.5) * 2)
        ])
        transformed_batch = torch.stack([transform(example['image']) for example in batch])
        # print("transformerd",transformed_batch.mean(),transformed_batch.min(),transformed_batch.max())
        return transformed_batch,None

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers)

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
        
    def sample_forward(self,x,t,eps=None):
        alpha_bar = self.alpha_bars[t].reshape(-1,1,1,1)
        if eps is None:
            eps = torch.randn_like(x)
        result = eps * torch.sqrt(1-alpha_bar) + torch.sqrt(alpha_bar)*x
        return result
        
    @torch.no_grad()
    def sample_backward(self, image_or_shape,net,device="cuda",simple_var=True):
        if isinstance(image_or_shape,torch.Tensor):
            x = image_or_shape
        else:
            x = torch.randn(image_or_shape,device=device)
        # debug 
        # print(x.max(),x.min(),x.mean())
        # for t in range(self.N-1,-1,-1):
        #     self.sample_backward_step(net, x, t, simple_var)
        # exit()
        for t in range(self.N-1,-1,-1):
            x = self.sample_backward_step(net, x, t, simple_var)
        return x
    @torch.no_grad()
    def sample_backward_step(self,net,x_t, t,simple_var,use_noise=True,clip_denoised=False):
        bs = x_t.shape[0]
        t_tensor = t*torch.ones(bs,dtype=torch.long,device=x_t.device).reshape(-1,1)
        if t == 0:
            noise = 0 
        else:
            if simple_var:
                var = self.betas[t]
            else:
                var = (1-self.alpha_bars_prev[t])/(1-self.alpha_bars[t]) * self.betas[t] 
            #这个地方还真写错了 randn_like和rand_like不一样wor
            noise = torch.randn_like(x_t) * torch.sqrt(var)
        eps = net(x_t,t_tensor)
        # with open("./cache.txt",'a') as f:
        #     f.write(f"{eps.mean().item()},{eps.max().item()},{eps.min().item()}\n")
        eps = ((1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t])) *eps 
        mean = (x_t - eps) / torch.sqrt(self.alphas[t])
        # eps = torch.sqrt(1-self.alpha_bars[t]) * eps 
        # print(1-self.alpha_bars[t])
        # mean = (x_t-eps)/torch.sqrt(self.alpha_bars[t])
        # print(f"{eps.mean().item()},{eps.max().item()},{eps.min().item()}")
        if use_noise:
            x_t_prev = mean + noise
        else:
            x_t_prev = mean
        if clip_denoised:
            x_t_prev.clamp_(-1., 1.)
        # print("noise",self.betas[t],noise.mean(),noise.max(),noise.min())
        # print("t",t_tensor[0],"eps:",eps.max(),eps.min(),eps.mean())
        # print(t_tensor)
        return x_t_prev

class DDIM(DDPM):
    def __init__(self,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02,
                 ddim_step: int = 20, # sample interval of ddim 
                  N: int=1000):
        super().__init__(min_beta, max_beta,N)
        self.ddim_step = ddim_step
        
    def sample_backward(self, image_or_shape, net, device="cuda", simple_var=True):
        if isinstance(image_or_shape,torch.Tensor):
            x = image_or_shape
        else:
            x = torch.randn(image_or_shape,device=device)
        
        sample_timestep = torch.linspace(0,1,self.ddim_step+1,device=device)
        for i in range(self.ddim_step-1,0,-1):
            bs = x.shape[0]
            t_cur = sample_timestep[i]
            t_prev = sample_timestep[i-1]
            
            ab_p = self.arlpha_bars[t_prev]
            ab_c = self.alpha_bars[t_cur]
            t_tensor = (bs * torch.ones(x.shape[0],device=device,dtype=torch.long)).reshpae(-1,1)
            eps = net(x, t_tensor)
            
            if simple_var:
                var = torch.sqrt(1-ab_p/ab_c) 
            else:
                eta = 1 # for ddim eta=1
                var = eta * (1 - ab_p) / (1 - ab_c) * (1 - ab_c / ab_p)
            noise = torch.randn_like(x)
            
            x = torch.sqrt(ab_p/ab_c) * x +\
                eps * var  + \
                torch.sqrt(var) * noise
        return x 
### trainer 
class LightningImageDenoiser(pl.LightningModule):
    def __init__(self, 
                 batch_size=512, 
                 lr=0.001,
                 min_beta=0.0001,
                 max_beta=0.02,
                 N=1000,
                 imsize=32,
                 num_workers=63,
                 channels = 1,
                 scheduler = "CosineAnnealingLR",
                 sample_output_dir = "./samples",
                 ):
        super(LightningImageDenoiser, self).__init__()
        self.save_hyperparameters()  # Save hyperparameters for logging
        image_shape = [channels,imsize,imsize]
        self.model = UNet(n_steps=N, image_shape=image_shape)
        self.ddpm = DDPM(min_beta=min_beta,max_beta=max_beta,N=N)
        self.criterion = nn.MSELoss()
        self.N = N 
        self.lr = lr 
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.scheduler = scheduler
        self.image_shape = image_shape
        self.sample_output_dir = sample_output_dir
    def forward(self, batch):
        # print(batch)
        images,_= batch
        # print(images.shape)
        bs = images.shape[0]
        t = torch.randint(0,self.N,(bs,),device = images.device)
        eps = torch.randn_like(images,device=images.device)
        x_t = self.ddpm.sample_forward(images, t, eps)
        # print(images.max(),images.min(),x_t.max(),x_t.min())
        eps_theta = self.model(x_t, t.reshape(bs, 1))
        # print(t)
        # print("training ",eps.max(),x_t.max(),eps_theta.max())
        loss = self.criterion(eps,eps_theta)
        return loss 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.scheduler == "ReduceLROnPlateau":
            
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',  # 监控验证集上的损失
                    'mode': 'min'           # 当监控指标不再降低时，减少学习率
                }
            }
        elif self.scheduler == "CosineAnnealingLR":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(optimizer, T_max=10)  # 定义CosineAnnealingLR调度器
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': self.scheduler
            }
        else:
            return optimizer
    def lr_scheduler_step(self, epoch, batch_idx, optimizer,**kwargs):
        # Manually update the learning rate based on the scheduler
        if self.scheduler is not None:
            self.scheduler.step()  # Update the scheduler
            
    def validation_step(self, batch, batch_idx):
        val_loss = self(batch)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return val_loss 
    
    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('train_loss', loss)
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False)
        return loss


    def sample_images(self, output_dir, n_sample=10, device="cuda", simple_var=True):
        max_batch_size = 32
        self.to(device)
        self.model.eval()
        with torch.no_grad():
            for i in range(0, n_sample, max_batch_size):
                shape = (min(max_batch_size, n_sample - i),*self.image_shape)
                imgs = self.ddpm.sample_backward(shape, self.model, device=device, simple_var=simple_var).detach().cpu()
                print("in sample images: ",imgs.max(),imgs.min())
                imgs = (imgs + 1) / 2 * 255
                imgs = imgs.clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
                os.makedirs(output_dir, exist_ok=True)
                for j, img in enumerate(imgs):
                    cv2.imwrite(f'{output_dir}/{i + j}.jpg', img)
    
    def on_train_epoch_end(self):
        output_dir = os.path.join(self.sample_output_dir, f'{self.current_epoch}')
        self.sample_images(output_dir=output_dir,n_sample=10,device="cuda",simple_var=True)    


    
###  parse args 
def parse_args():
    """
    解析命令行参数，并返回解析结果。
    """
    parser = argparse.ArgumentParser(description='Training script')
    
    parser.add_argument('--expname', type=str, default=None ,help='expname of this experiment')
    parser.add_argument('--train', action='store_true', help='Whether to run in training mode')
    parser.add_argument('--devices', type=str, default='0,', help='Specify the device(s) for training (e.g., "cuda" or "cuda:0")')
    parser.add_argument('--max_epochs', type=int, default=1200, help='Maximum number of epochs for training')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum number of epochs for training')
    parser.add_argument('--min_beta', type=float, default=0.0001, help='Minimum value of beta for DDPM')
    parser.add_argument('--max_beta', type=float, default=0.02, help='Maximum value of beta for DDPM')
    parser.add_argument('--max_step', type=int, default=1000, help='Number of steps (N) for DDPM')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=63, help='num_workers training data loader')
    parser.add_argument('--channels', type=int, default=3, help='channels of image ')
    parser.add_argument('--imsize', type=int, default=64, help='image size ')
    parser.add_argument('--scheduler', type=str, default="None", help='lr policy')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    expname = args.expname
    if args.train:
        
        # dataset = "mnist"
        dataset = "celeba"
        if dataset =="celeba":
            data_module = CelebDataModule(batch_size=args.batch_size,num_workers=args.num_workers)
        else:
            data_module = MNISTDataModule(data_dir="./data", batch_size=args.batch_size)
            
        data_module.prepare_data()
        data_module.setup()

        model = LightningImageDenoiser(
            min_beta=args.min_beta,
            max_beta=args.max_beta,
            N = args.max_steps,
            batch_size = args.batch_size,
            num_workers=args.num_workers,
            channels=args.channels,
            imsize= args.imsize,
            lr = args.lr,
            scheduler=args.scheduler,
            sample_output_dir=f"./sample/{expname}"
            )

        # 设置保存 checkpoint 的回调函数
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"./checkpoints/{expname}",  # 保存 checkpoint 的目录
            filename="model-{epoch:02d}-{val_loss:.5f}",  # checkpoint 文件名格式
            monitor="val_loss",  # 监控的指标，这里使用验证集损失
            mode="min",  # 指定监控模式为最小化验证集损失
            save_top_k=3,  # 保存最好的 3 个 checkpoint
            verbose=True
        )
        
        # pretrain_path = "/data2/wuhaoyu/SimpleDiffusion/UnconditionalDiffusion/checkpoints/model-epoch=443-train_loss=0.00147.ckpt"
        # pretrain_path = "/home/haoyu/research/simplemodels/SimpleDiffusion/UnconditionalDiffusion/checkpoints/model-epoch=159-val_loss=0.00454.ckpt"
        pretrain_path = None 
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=args.devices,                    # 使用一块 GPU 进行训练
            max_epochs=args.max_epochs,             # 最大训练 epoch 数
            logger=pl.loggers.TensorBoardLogger("logs/", name=expname),
            # progress_bar_refresh_rate=20,  # 进度条刷新频率
            callbacks=[checkpoint_callback],  # 注册 checkpoint 回调函数
        )

        trainer.fit(model,data_module,ckpt_path = pretrain_path)
    else:
        ckpt_folder = f"./checkpoints/{expname}"
        paths = os.listdir(ckpt_folder)
        paths = [os.path.join(ckpt_folder,i) for i in paths]
        paths = ["/home/haoyu/research/simplemodels/SimpleDiffusion/UnconditionalDiffusion/checkpoints/linear_normal/model-epoch=1184-val_loss=0.00332.ckpt"]
        for path in paths:
            ckpt = os.path.basename(path).replace(".ckpt","")
            model = LightningImageDenoiser(
                min_beta=args.min_beta,
                max_beta=args.max_beta,
                N = args.max_steps,
                batch_size = args.batch_size,
                num_workers=args.num_workers,
                channels=args.channels,
                imsize= args.imsize,
                lr = args.lr,
                scheduler=args.scheduler,
                )
            model.load_state_dict(torch.load(path)['state_dict'],strict=True)
            
            # net.load_state_dict(torch.load(path)['state_dict'],strict=True)
            # ddpm = DDPM(min_beta=0.0001,max_beta=0.02,N=1000)
            model.sample_images(f'./sample/{ckpt}',n_sample=32,device="cuda:0")
            # sample_image(model.ddpm,
            #         model.model,
            #         f'./sample/{ckpt}',
            #         image_shape=image_shape,
            #         n_sample=32,
            #         device="cuda:3",
            #         )