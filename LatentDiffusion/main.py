"""
autor: haoyu
date: 20240501-0506
an simplified unconditional diffusion for image generation
"""
import os
import argparse
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.optim.lr_scheduler import ReduceLROnPlateau
### local files 
## sorry to use global value 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from schedulers.ddpm import DDPM
from util import instantiate_from_config
from util import images2gif
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
                 unet_config = {},
                 vae_config = {},
                 vae_pretrained_path = None
                 ):
        super(LatentDiffusion, self).__init__()
        self.save_hyperparameters()  # Save hyperparameters for logging
        image_shape = [channels,imsize,imsize]
        self.ddpm = DDPM(min_beta=min_beta,max_beta=max_beta,N=N)
        print("===================")
        print(vae_config)
        self.vae = instantiate_from_config(vae_config)
        self.latent_shape = self.vae.decoder.z_shape[1:]
        unet_config["params"]["n_steps"]=N
        unet_config["params"]["latent_shape"]=self.latent_shape
        self.denoiser = instantiate_from_config(unet_config)
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
    parser = argparse.ArgumentParser(description='Training script')
    ## epoch 200 with loss 0.02 is enough to generate on mnist 
    parser.add_argument('--expname', type=str, default=None ,help='expname of this experiment')
    parser.add_argument('--train', action='store_true', help='Whether to run in training mode')
    parser.add_argument('--auto_resume', action='store_true', help='whether resume from trained checkpoint ')
    parser.add_argument("-b", "--base", nargs="*", metavar="configs/train.yaml", help="paths to base configs. Loaded from left-to-right. Parameters can be overwritten or added with command-line options of the form `--key value`.", default=list())
    args = parser.parse_args()

    return args
if __name__ == "__main__":
    args= parse_args()
    # args, unknown = parser.parse_known_args()
    # parser = Trainer.add_argparse_args(parser)
    configs = [OmegaConf.load(cfg) for cfg in args.base]
    # cli = OmegaConf.from_dotlist(unknown)
    # config = OmegaConf.merge(*configs, cli)
    config = OmegaConf.merge(*configs)
    expname = config.expname
    imsize = config.imsize
    if args.train:
        data_module = instantiate_from_config(config.data)
        data_module.prepare_data()
        data_module.setup()
        model = instantiate_from_config(config.model)
        
        # 设置保存 checkpoint 的回调函数
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"./checkpoints/{expname}",  # 保存 checkpoint 的目录
            filename="model-{epoch:02d}-{val_loss:.5f}",  # checkpoint 文件名格式
            monitor="val_loss",  # 监控的指标，这里使用验证集损失
            mode="min",  # 指定监控模式为最小化验证集损失
            save_top_k=3,  # 保存最好的 3 个 checkpoint
            verbose=True
        )
        trainer_config = config.trainer.params
        trainer = pl.Trainer(
            **trainer_config,
            logger=pl.loggers.TensorBoardLogger("logs/", name=expname),
            callbacks=[checkpoint_callback],  # 注册 checkpoint 回调函数
        )
        if config.pretrain_path != "None":
            pretrain_path = config.pretrain_path
        else:
            pretrain_path = None 
        trainer.fit(model,data_module,ckpt_path =pretrain_path)
    else:
        ckpt_folder = f"./checkpoints/{expname}"
        paths = os.listdir(ckpt_folder)
        paths = [os.path.join(ckpt_folder,i) for i in paths]
        paths = ["/home/haoyu/research/simplemodels/SimpleDiffusion/UnconditionalDiffusion/checkpoints/linear_normal/model-epoch=1184-val_loss=0.00332.ckpt"]
        for path in paths:
            ckpt = os.path.basename(path).replace(".ckpt","")
            model = instantiate_from_config(config.model)
            model.load_state_dict(torch.load(path)['state_dict'],strict=True)
            model.sample_images(f'./sample/{ckpt}',n_sample=32,device="cuda:0")