"""
author: haoyu 
base on: https://github.com/Jackson-Kang/Pytorch-VAE-tutorial
to implement a VAE training framework using pytorch lightning 
"""

import torch
import torch.nn as nn

import numpy as np
import os , cv2 , sys
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import pytorch_lightning as pl 
        
class VAE(nn.Module):
    def __init__(self, 
                encoder_config,
                decoder_config,
                 device="cuda"):
        super(VAE, self).__init__()
        self.encoder = instantiate_from_config(encoder_config)
        self.decoder = instantiate_from_config(decoder_config)
        self.device = device
    
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var,device = self.device)#.to(DEVICE)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
    
    def encode(self,x):
        mean, log_var = self.encoder(x)
        # print("reparameter: ",mean.shape,log_var.shape,x.shape);
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        # print("z latent:", z.shape);exit()
        return z
    def decode(self,z):
        return self.decoder(z)
    
    def sample(self,n_sample):
        return self.decoder.sample(n_sample)
    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat  = self.decoder(z)
        
        return x_hat, mean, log_var

    def loss_fn(self,x, x_hat, mean, log_var):
        # print("loss fn",x.shape,x_hat.shape,log_var.shape,mean.shape)
        x = x.view(x.shape[0],-1)
        x_hat = x_hat.view(x_hat.shape[0],-1)
        log_var = log_var.view(log_var.shape[0],-1)
        mean = mean.view(mean.shape[0],-1)
        # print("loss fn",x.shape,x_hat.shape,log_var.shape,mean.shape);#exit()
        # print("x max min",x.max(),x.min(),x_hat.max(),x_hat.min());exit()
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss + KLD
        # return KLD 


class VAETrainer(pl.LightningModule):
    def __init__(self, 
                 batch_size=512, 
                 lr=0.001,
                 imsize=32,
                 num_workers=63,
                 channels = 1,
                 scheduler = "CosineAnnealingLR",
                 sample_output_dir = "./samples",
                 sample_epoch_interval = 20,
                 vae_config={},
                 device = "cuda",
                 ):
        super(VAETrainer, self).__init__()
        # self.model = VAE(resolution=imsize,in_channels=channels,device=device)
        self.model = instantiate_from_config(vae_config)
        self.save_hyperparameters()  # Save hyperparameters for logging
        image_shape = [channels,imsize,imsize]
        self.lr = lr 
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.scheduler = scheduler
        self.image_shape = image_shape
        self.sample_output_dir = sample_output_dir
        self.sample_epoch_interval = sample_epoch_interval


    def forward(self, batch):
        x, _ = batch
        # x = x.view(self.batch_size, x_dim)
        x_hat, mean, log_var = self.model(x) 
        loss = self.model.loss_fn(x, x_hat, mean, log_var)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
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
        output_file =  os.path.join(output_dir , "generated_images.png")
        with torch.no_grad():
            # noise = torch.randn(n_sample, latent_dim)#.to(DEVICE)
            generated_images = self.model.sample(n_sample)
            save_image(generated_images.view(n_sample,*self.image_shape),output_file, nrow=5, normalize=True)
    def on_fit_start(self):
        output_dir = os.path.join(self.sample_output_dir, f'init_ckpt')
        os.makedirs(output_dir,exist_ok=True)
        self.sample_images(output_dir=output_dir,n_sample=25,device="cuda",simple_var=True)    

    def on_train_epoch_end(self):
        if (self.current_epoch+1) % self.sample_epoch_interval==0:
            print(f"sampling {self.current_epoch}/{self.sample_epoch_interval},")
            output_dir = os.path.join(self.sample_output_dir, f'{self.current_epoch}')
            os.makedirs(output_dir,exist_ok=True)
            self.sample_images(output_dir=output_dir,n_sample=25,device="cuda",simple_var=True)    

    
import argparse 
from omegaconf import OmegaConf
from util import instantiate_from_config

def parse_args():
    parser = argparse.ArgumentParser(description='Training script')
    ## epoch 200 with loss 0.02 is enough to generate on mnist 
    parser.add_argument('--expname', type=str, default=None ,help='expname of this experiment')
    parser.add_argument('--train', action='store_true', help='Whether to run in training mode')
    parser.add_argument('--auto_resume', action='store_true', help='whether resume from trained checkpoint ')
    parser.add_argument("-b", "--base", nargs="*", metavar="configs/train.yaml", help="paths to base configs. Loaded from left-to-right. Parameters can be overwritten or added with command-line options of the form `--key value`.", default=list())
    args = parser.parse_args()

    return args
# 尽量不要用可变的变量作为参数，这样更改了之后又bug不知道
# 重要参数不默认， 默认参数不重要
# 硬编码的地方要写注释 反之后来不知道为什么 
# 多写脚本处理问题 

# 0608 modification 
# the VAE trainer should only contain training logics 
# and the model config should be decoupled 


if __name__ == "__main__":
    args= parse_args()
    configs = [OmegaConf.load(cfg) for cfg in args.base]
    config = OmegaConf.merge(*configs)
    expname = config.expname
    # data_module = MNISTDataModule(data_dir=dataset_path, batch_size=100, num_workers=63)
    data_module = instantiate_from_config(config.data)
    from pytorch_lightning.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
    dirpath=f"./checkpoints/{expname}",  # 保存 checkpoint 的目录
    filename="model-{epoch:02d}-{val_loss:.5f}",  # checkpoint 文件名格式
    monitor="val_loss",  # 监控的指标，这里使用验证集损失
    mode="min",  # 指定监控模式为最小化验证集损失
    save_top_k=3,  # 保存最好的 3 个 checkpoint
    verbose=True
    )

    model = instantiate_from_config(config.model)
    trainer = instantiate_from_config(config.trainer)
    # config checkpoint 
    if config.pretrain_path != "None":
        pretrain_path = config.pretrain_path
    else:
        pretrain_path = None 
    trainer.fit(model,data_module,ckpt_path =pretrain_path)
    