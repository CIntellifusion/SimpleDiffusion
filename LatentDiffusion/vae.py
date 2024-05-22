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



"""
    A simple implementation of Gaussian MLP Encoder and Decoder
"""

class SimpleEncoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(SimpleEncoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        
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
        
    
        
class VAE(nn.Module):
    def __init__(self, x_dim, hidden_dim, latent_dim,device):
        super(VAE, self).__init__()
        self.encoder = SimpleEncoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = SimpleDecoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)
        self.device = device
    
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var,device = self.device)#.to(DEVICE)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
    
    def encode(self,x):
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        return z
    def decode(self,z):
        return self.decoder(z)
                
    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat  = self.decoder(z)
        
        return x_hat, mean, log_var

    def loss_fn(self,x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss + KLD


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
                 ):
        super(VAETrainer, self).__init__()
        self.model = VAE(x_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.lr = lr
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
        x = x.view(self.batch_size, x_dim)
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
            noise = torch.randn(n_sample, latent_dim)#.to(DEVICE)
            generated_images = self.model.decoder(noise)
            save_image(generated_images.view(n_sample,*self.image_shape),output_file, nrow=5, normalize=True)
    
    def on_train_epoch_end(self):
        if (self.current_epoch+1) % self.sample_epoch_interval==0:
            print(f"sampling {self.current_epoch}/{self.sample_epoch_interval},")
            output_dir = os.path.join(self.sample_output_dir, f'{self.current_epoch}')
            os.makedirs(output_dir,exist_ok=True)
            self.sample_images(output_dir=output_dir,n_sample=25,device="cuda",simple_var=True)    

    


if __name__=="__main__":
    dataset_path = '../data'
    data_module = MNISTDataModule(data_dir=dataset_path, batch_size=100, num_workers=63)
    cuda = True
    DEVICE = torch.device("cuda" if cuda else "cpu")
    x_dim  = 784
    hidden_dim = 400
    latent_dim = 784
    lr = 1e-3
    epochs = 100
    sample_epoch_interval = 10
    sample_output_dir = "./samples"
    expname = "vae"
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
    model = VAETrainer(batch_size=100,
                                 lr=lr,imsize=28,
                                 num_workers=63,
                                 sample_output_dir=sample_output_dir,
                                 sample_epoch_interval=sample_epoch_interval)
    
    trainer = pl.Trainer(gpus=1 if cuda else 0,
                         max_epochs=epochs,
                         logger=pl.loggers.TensorBoardLogger("logs/", name=expname),
                         callbacks=[checkpoint_callback],
                         )
    
    trainer.fit(model=model,datamodule=data_module,ckpt_path=pretrain_path)


    
    