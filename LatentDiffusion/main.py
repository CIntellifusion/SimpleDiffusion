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
from torchvision.utils import save_image, make_grid

### local files 
from data.data_wrapper import MNISTDataModule,CelebDataModule
from models.unet import UNet
from schedulers.ddpm import DDPM
from vae import VAE
from util import images2gif
## sorry to use global value 
imsize = 32 
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


### trainer 
class LatentDiffusion(pl.LightningModule):
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
                 sample_epoch_interval = 20,
                 unet_config = {"channels":[10, 20, 40 ,80],"pe_dim":100,"with_attns":True},
                 vae_config = {"resolution":64,'in_channels':3,"device":"cuda:0"},
                 vae_pretrained_path = None
                 ):
        super(LatentDiffusion, self).__init__()
        self.save_hyperparameters()  # Save hyperparameters for logging
        image_shape = [channels,imsize,imsize]
        #TODO: latent shape 
        self.ddpm = DDPM(min_beta=min_beta,max_beta=max_beta,N=N)
        self.vae = VAE(**vae_config)
        self.latent_shape = self.vae.decoder.z_shape[1:]
        self.denoiser = UNet(n_steps=N, 
                             latent_shape=self.latent_shape,
                             **unet_config)
        self.vae_config = vae_config
        self.config_vae(vae_pretrained_path)
        self.criterion = nn.MSELoss()
        self.N = N 
        self.lr = lr 
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.scheduler = scheduler
        self.image_shape = image_shape

        self.sample_output_dir = sample_output_dir
        self.sample_epoch_interval = sample_epoch_interval
        
    def config_vae(self,pretrained_path):
        ckpt = torch.load(pretrained_path)
        if "state_dict" in ckpt.keys():
            ckpt = ckpt["state_dict"]
        new_state_dict = {}
        for k,v in ckpt.items():
            new_state_dict[k.replace("model.","")] = v
        self.vae.load_state_dict(new_state_dict)
        
        # freeze parameters
        for param in self.vae.parameters():
            param.requires_grad = False
        # eval mode
        self.vae.eval()
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.denoiser.parameters(), lr=self.lr)
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
        elif self.scheduler =="LineaerLR":
            from torch.optim.lr_scheduler import StepLR
            self.scheduler = StepLR(optimizer, step_size=2000, gamma=0.9)  # 定义CosineAnnealingLR调度器
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
    
    def AE_encode(self,x):
        # x = torch.concat(x)
        # print("AE_encode",x.shape)# [128, 1, 28, 28]
        # bs = x.shape[0]
        # x = x.view(bs, *self.latent_shape)
        return self.vae.encode(x) # encoder posterior ; tensor 
    
    def AE_decode(self,x):
        return self.vae.decode(x) 
            
    def get_input(self,batch):
        images,_= batch
        return self.AE_encode(images)
    
    def forward(self, batch):
        # print(batch)
        latents = batch 
        bs = latents.shape[0]
        # print("forward ae encode shape",latents.shape)
        # latents = latents.reshape(bs,*self.latent_shape)
        # print("forward ae reshaped encode shape",latents.shape)

        t = torch.randint(0,self.N,(bs,),device = latents.device)
        eps = torch.randn_like(latents,device=latents.device)
        x_t = self.ddpm.sample_forward(latents, t, eps)
        # print(latents.max(),latents.min(),x_t.max(),x_t.min())
        eps_theta = self.denoiser(x_t, t.reshape(bs, 1))
        # print(t)
        # print("training ",eps.max(),x_t.max(),eps_theta.max())
        loss = self.criterion(eps,eps_theta)
        return loss 
    def validation_step(self, batch, batch_idx):
        batch = self.get_input(batch)
        val_loss = self(batch)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return val_loss 
    
    def training_step(self, batch, batch_idx):
        batch = self.get_input(batch)
        loss = self(batch)
        self.log('train_loss', loss)
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False)
        return loss


    def sample_images(self, output_dir, n_sample=9, device="cuda", simple_var=True):
        max_batch_size = 32
        self.to(device)
        self.denoiser.eval()
        name = "generated_images.png"
        os.makedirs(output_dir, exist_ok=True)
        with torch.no_grad():
            for i in range(0, n_sample, max_batch_size):
                # shape = (min(max_batch_size, n_sample - i),*self.image_shape)
                bs = min(max_batch_size, n_sample - i)
                shape = (bs,*self.latent_shape)
                latents = self.ddpm.sample_backward(shape, self.denoiser, device=device, simple_var=simple_var)
                imgs = self.AE_decode(latents.view((bs,*self.latent_shape))).detach().cpu()
                
                print("in sample images: ",imgs.max(),imgs.min())
                # imgs = (imgs + 1) / 2 * 255
                # imgs = imgs.clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)#.numpy()
                output_file = os.path.join(output_dir,name )
                channels,h,w = self.image_shape
                save_image(imgs.view(n_sample,channels,h,w),output_file, nrow=3, normalize=True)
    
    def on_train_epoch_end(self):
        if (self.current_epoch + 1)  % self.sample_epoch_interval==0:
            output_dir = os.path.join(self.sample_output_dir, f'{self.current_epoch+1:05}')
            self.sample_images(output_dir=output_dir,n_sample=9,device="cuda",simple_var=True)    

    # after training , call imagetogif
    def on_fit_end(self):
        folder = self.sample_output_dir
        savepath = os.path.join(folder, "generated_video.gif")
        subfolders = sorted(os.listdir(self.sample_output_dir))
        name = "generated_images.png"
        image_files = sorted([os.path.join(folder,sf,name) for sf in subfolders])
        images2gif(image_files,savepath)
    
###  parse args 
def parse_args():
    """
    解析命令行参数，并返回解析结果。
    """
    parser = argparse.ArgumentParser(description='Training script')
    ## epoch 200 with loss 0.02 is enough to generate on mnist 
    parser.add_argument('--expname', type=str, default=None ,help='expname of this experiment')
    parser.add_argument('--train', action='store_true', help='Whether to run in training mode')
    parser.add_argument('--devices', type=str, default='0,', help='Specify the device(s) for training (e.g., "cuda" or "cuda:0")')
    parser.add_argument('--max_epochs', type=int, default=600, help='Maximum number of epochs for training')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum number of epochs for training')
    parser.add_argument('--min_beta', type=float, default=0.0001, help='Minimum value of beta for DDPM')
    parser.add_argument('--max_beta', type=float, default=0.02, help='Maximum value of beta for DDPM')
    parser.add_argument('--max_step', type=int, default=1000, help='Number of steps (N) for DDPM')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=63, help='num_workers training data loader')
    parser.add_argument('--channels', type=int, default=3, help='channels of image ')
    parser.add_argument('--imsize', type=int, default=64, help='image size ')
    parser.add_argument('--scheduler', type=str, default="None", help='lr policy')
    parser.add_argument('--dataset', type=str, default="mnist", help='dataset')
    parser.add_argument('--sample_epoch_interval', type=int, default=1, help='sample interval')
    parser.add_argument('--vae_ckpt', type=str, default="/home/haoyu/research/simplemodels/LatentDiffusion/checkpoints/vae/model-epoch=81-val_loss=10050.03320.ckpt", help='pretrained vae checkpoint path')
    args = parser.parse_args()

    return args

#python main.py  --train --expname ldm_celeb64 --max_epochs 100 --batch_size 64 --dataset celeba --vae_ckpt /home/haoyu/research/simplemodels/LatentDiffusion/checkpoints/vae_aemodule_celeb64_halftiny/model-epoch=38-val_loss=397428.12500.ckpt
#python main.py  --train --expname ldm_celeb64 --max_epochs 100 --batch_size 64 --dataset celeba --vae_ckpt /home/haoyu/research/simplemodels/LatentDiffusion/checkpoints/vae_celeb64_ch16_outz16/model-epoch=38-val_loss=398023.75000.ckpt
#python main.py  --train --expname ldm_celeb64_unetmid --max_epochs 100 --batch_size 64 --dataset celeba --vae_ckpt /home/haoyu/research/simplemodels/LatentDiffusion/checkpoints/vae_celeb64_ch16_outz16/model-epoch=38-val_loss=398023.75000.ckpt
#python main.py  --train --expname ldm_celeb64_unetlarget --max_epochs 100 --batch_size 64 --dataset celeba --vae_ckpt /home/haoyu/research/simplemodels/LatentDiffusion/checkpoints/vae_celeb64_ch16_outz16/model-epoch=38-val_loss=398023.75000.ckpt
if __name__ == "__main__":
    args = parse_args()
    expname = args.expname
    imsize = args.imsize
    if args.train:
        dataset = args.dataset
        # dataset = "celeba"
        if dataset =="celeba":
            data_module = CelebDataModule(batch_size=args.batch_size,
                                          num_workers=args.num_workers,imsize=args.imsize)
        elif dataset=="mnist":
            data_module = MNISTDataModule(data_dir="/home/haoyu/research/simplemodels/data", 
                                          batch_size=args.batch_size,num_workers=args.num_workers)
        else:
            raise NotImplementedError("Not supported dataset")
        data_module.prepare_data()
        data_module.setup()
        default_unet_config = {"channels":[64,128,128,64],
                   "pe_dim":100,
                   "with_attns":True,
                   "norm_type":"ln"
                   }
        model = LatentDiffusion(
            min_beta=args.min_beta,
            max_beta=args.max_beta,
            N = args.max_steps,
            batch_size = args.batch_size,
            num_workers=args.num_workers,
            channels=args.channels,
            imsize= args.imsize,
            lr = args.lr,
            scheduler=args.scheduler,
            sample_output_dir=f"./sample/{expname}",
            sample_epoch_interval=args.sample_epoch_interval,
            vae_pretrained_path = args.vae_ckpt,
            unet_config=default_unet_config
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
        
        pretrain_path = None 
        # pretrain_path = "/home/haoyu/research/simplemodels/LatentDiffusion/checkpoints/ldm_celeb64/model-epoch=07-val_loss=0.77585.ckpt" 
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
            model = LatentDiffusion(
                min_beta=args.min_beta,
                max_beta=args.max_beta,
                N = args.max_steps,
                batch_size = args.batch_size,
                num_workers=args.num_workers,
                channels=args.channels,
                imsize= args.imsize,
                lr = args.lr,
                scheduler=args.scheduler,
                sample_epoch_interval=args.sample_epoch_interval
                )
            model.load_state_dict(torch.load(path)['state_dict'],strict=True)
            model.sample_images(f'./sample/{ckpt}',n_sample=32,device="cuda:0")