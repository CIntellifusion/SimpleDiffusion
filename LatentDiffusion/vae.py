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
import torch.optim as optim
from data.data_wrapper import MNISTDataModule,CelebDataModule
from models.ae_module import SimpleEncoder,SimpleDecoder
from models.ae_module import Encoder,Decoder

    
        
class VAE(nn.Module):
    def __init__(self, resolution,in_channels,
                 z_channels =16 ,
                 device="cuda"):
        super(VAE, self).__init__()
        print("resolution:",resolution,"in_channels:",in_channels)
        self.encoder = Encoder(
                      ch=16,
                      resolution=resolution,
                      in_channels=in_channels,
                      ch_mult=(1,2,4,8),
                      num_res_blocks=2,
                      attn_resolutions=(16,),
                      dropout=0.0,
                      resamp_with_conv=True,
                      z_channels=z_channels,
                      double_z=False,
                      use_linear_attn=False,
                      use_checkpoint=False)
        self.decoder = Decoder(ch=16,
                      out_ch=in_channels,
                      resolution=resolution,
                      in_channels=in_channels,
                      ch_mult=(1,2,4,8),
                      num_res_blocks=2,
                      attn_resolutions=(16,),
                      dropout=0.0,
                      resamp_with_conv=True,
                      z_channels=z_channels,
                      give_pre_end=False,   
                      tanh_out=False,
                      use_linear_attn=False,
                      use_checkpoint=False)
        
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
                 device = "cuda"
                 ):
        super(VAETrainer, self).__init__()
        self.model = VAE(resolution=imsize,in_channels=channels,device=device)
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

    


if __name__=="__main__":
    dataset_path = '../data'
    imsize = 64
    batch_size = 64
    # data_module = MNISTDataModule(data_dir=dataset_path, batch_size=100, num_workers=63)
    data_module = CelebDataModule(batch_size=batch_size,
                        num_workers=63,imsize=imsize)
    cuda = True
    DEVICE = torch.device("cuda" if cuda else "cpu")
    lr = 1e-3
    epochs = 40
    sample_epoch_interval = 1
    expname = "vae_celeb64_ch16_outz16"
    sample_output_dir = f"./sample/{expname}"
    from pytorch_lightning.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
    dirpath=f"./checkpoints/{expname}",  # 保存 checkpoint 的目录
    filename="model-{epoch:02d}-{val_loss:.5f}",  # checkpoint 文件名格式
    monitor="val_loss",  # 监控的指标，这里使用验证集损失
    mode="min",  # 指定监控模式为最小化验证集损失
    save_top_k=3,  # 保存最好的 3 个 checkpoint
    verbose=True
    )
    pretrain_path = "/home/haoyu/research/simplemodels/LatentDiffusion/checkpoints/vae/model-epoch=38-val_loss=10231.90039.ckpt"
    pretrain_folder = "./checkpoints/vae_aemodule_celeb64/" 
    pretrain_path = os.path.join(pretrain_folder,"model-epoch=01-val_loss=406142.43750.ckpt") 
    pretrain_path=None 
    model = VAETrainer(batch_size=batch_size,
                        channels=3,
                        lr=lr,imsize=imsize,
                        num_workers=63,
                        sample_output_dir=sample_output_dir,
                        sample_epoch_interval=sample_epoch_interval)
    
    trainer = pl.Trainer(
                        accelerator = "gpu",
                        devices=1,
                        max_epochs=epochs,
                        logger=pl.loggers.TensorBoardLogger("logs/", name=expname),
                        callbacks=[checkpoint_callback],
                        )
    
    trainer.fit(model=model,datamodule=data_module,ckpt_path=pretrain_path)


    
    