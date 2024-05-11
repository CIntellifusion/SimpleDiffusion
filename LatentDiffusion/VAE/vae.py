"""
author: haoyu 
base on: https://github.com/Jackson-Kang/Pytorch-VAE-tutorial
to implement a VAE training framework using pytorch lightning 
"""

import torch
import torch.nn as nn

import numpy as np
import os , cv2 
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import pytorch_lightning as pl 
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
# Model Hyperparameters

"""
    MNSIT dataset 
"""
from torchvision.datasets import MNIST
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./", batch_size=64,num_workers=63):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

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
                # transforms.Normalize((0.1307,), (0.3081,)) # 
            ])
        
        if stage == 'fit' or stage is None:
            self.train_dataset = MNIST(self.data_dir, train=True, transform=transform)
            self.val_dataset = MNIST(self.data_dir, train=False, transform=transform)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=self.num_workers,pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,num_workers=self.num_workers,pin_memory=True)

"""
    celeba dataset
"""
from datasets import load_dataset 
class CelebDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64,num_workers=63,imsize=32):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.imsize = args.imsize 
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
            transforms.Resize((imsize,imsize)),  
            transforms.ToTensor(),           
            # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))  # Normalize images
            # transforms.Lambda(lambda x: (x - 0.5) * 2) # unconment 
        ])
        transformed_batch = torch.stack([transform(example['image']) for example in batch])
        # print("transformerd",transformed_batch.mean(),transformed_batch.min(),transformed_batch.max())
        return transformed_batch,None

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size, 
                          collate_fn=self.collate_fn, 
                          shuffle=True, num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          collate_fn=self.collate_fn, num_workers=self.num_workers,
                          pin_memory=True)


"""
    A simple implementation of Gaussian MLP Encoder and Decoder
"""

class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        h_       = self.LeakyReLU(self.FC_input(x))
        h_       = self.LeakyReLU(self.FC_input2(h_))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     # encoder produces mean and log of variance 
                                                       #             (i.e., parateters of simple tractable normal distribution "q"
        
        return mean, log_var
    
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
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
    def __init__(self, x_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)
    
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(DEVICE)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat            = self.decoder(z)
        
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
            noise = torch.randn(n_sample, latent_dim).to(DEVICE)
            generated_images = self.model.decoder(noise)
            save_image(generated_images.view(n_sample,*self.image_shape),output_file, nrow=5, normalize=True)
    
    def on_train_epoch_end(self):
        if self.current_epoch+1 % self.sample_epoch_interval==0:
            output_dir = os.path.join(self.sample_output_dir, f'{self.current_epoch}')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            self.sample_images(output_dir=output_dir,n_sample=25,device="cuda",simple_var=True)    

    


if __name__=="__main__":
    
    dataset_path = '../data'
    data_module = MNISTDataModule(data_dir=dataset_path, batch_size=100, num_workers=63)
    cuda = True
    DEVICE = torch.device("cuda" if cuda else "cpu")
    x_dim  = 784
    hidden_dim = 400
    latent_dim = 200
    lr = 1e-3
    epochs = 30
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
    
    model = VAETrainer(batch_size=100,
                                 lr=lr,imsize=28,
                                 num_workers=63,
                                 sample_output_dir=sample_output_dir,
                                 sample_epoch_interval=sample_epoch_interval)
    
    trainer = pl.Trainer(gpus=1 if cuda else 0, max_epochs=epochs)
    trainer.fit(model=model,
                datamodule=data_module)


    
    