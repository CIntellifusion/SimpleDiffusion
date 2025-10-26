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
import numpy as np 
from torchvision.utils import save_image
from torch.optim.lr_scheduler import ReduceLROnPlateau
### local files 
## sorry to use global value 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from util import instantiate_from_config
from util import images2gif
# imsize = 32 
torch.set_float32_matmul_precision('high')
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
                 lr=0.001,
                 N=1000,
                 imsize=32,
                 channels = 1,
                 scheduler = "CosineAnnealingLR",
                 sample_output_dir = "./samples",
                 sample_epoch_interval = 20,
                 noise_scheduler_config = {},
                 model_config = {},
                 vae_config = {},
                 vae_pretrained_path = '',
                 model_pretrained_path = '',
                 loss_type="flow_matching"
                 ):
        super(LatentDiffusion, self).__init__()
        self.save_hyperparameters()  # Save hyperparameters for logging
        image_shape = [channels,imsize,imsize]
        self.noise_scheduler = instantiate_from_config(noise_scheduler_config)
        print("===================")
        print(vae_config)
        self.vae = instantiate_from_config(vae_config)
        self.latent_shape = [3,64,64]
        # model_config["params"]["n_steps"]=N
        # model_config["params"]["latent_shape"]=[3,64,64]
        self.denoiser = instantiate_from_config(model_config)
        self.vae_config = vae_config
        self.config_vae(vae_pretrained_path)
        self.criterion = nn.MSELoss()
        self.N = N 
        self.lr = lr 
        self.scheduler = scheduler
        self.image_shape = image_shape

        self.sample_output_dir = sample_output_dir
        self.sample_epoch_interval = sample_epoch_interval
        
        self.loss_type = loss_type
        if model_pretrained_path != '':
            self.load_state_dict(torch.load(model_pretrained_path,weights_only=False)['state_dict'],strict=True)
        
        if self.loss_type == "simple_consistency_distillation":
            print(f"creating reference model for consistency distillation")
            self.ref_model = instantiate_from_config(model_config)
            denoiser_state_dict = self.denoiser.state_dict()
            self.ref_model.load_state_dict(denoiser_state_dict,strict=True)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
                
        
    def config_vae(self,pretrained_path):
        if os.path.exists(pretrained_path) is False:
            print("no pretrained vae found, use identity mapping")
            return 
        else:
            ckpt = torch.load(pretrained_path,weights_only=False)
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
            self.scheduler = CosineAnnealingLR(optimizer, T_max=50)  # 定义CosineAnnealingLR调度器
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
    
    
    def AE_encode(self,x):
        # x = torch.concat(x)
        # print("AE_encode",x.shape)# [128, 1, 28, 28]
        # bs = x.shape[0]
        # x = x.view(bs, *self.latent_shape)
        return self.vae.encode(x) # encoder posterior ; tensor 
    
    def AE_decode(self,x):
        return self.vae.decode(x) 
            
    def get_input(self,batch):
        images,_ = batch
        return self.AE_encode(images)
    
    def forward(self, batch):
        # unconditional generation , no class label
        latents = batch 
        # latents range : [-1,1]
        # latents shape : B,C,H,W 
        # t range : [0,1]
        loss_type = self.loss_type
        # print("loss type: ",loss_type)
        if loss_type == "flow_matching":
            bs = latents.shape[0]
            t = torch.rand((bs,1),device = latents.device)
            sigma = t.reshape(bs,1,1,1)
            x_t,target = self.noise_scheduler.sample_forward(latents, sigma)
            model_output = self.denoiser(x_t, t)
            loss = self.criterion(target,model_output)
        elif loss_type == "simple_consistency_distillation":
            bs = latents.shape[0]
            t = torch.rand((bs,1),device = latents.device)
            # re-name
            data = latents
            sigma = t.reshape(bs,1,1,1)
            # sample forward 
            noise = torch.randn_like(data).to(data.device)
            # import pdb; pdb.set_trace()
            # print(f"noise size : {noise.shape}, data size: {data.shape} sigma size: {sigma.shape}")
            x_t = sigma * data + (1-sigma)  * noise
            target = data - noise
            # get velocity 
            # import pdb ; pdb.set_trace()
            assert hasattr(self,"ref_model")
            with torch.no_grad():
                cuda_state = torch.cuda.get_rng_state()
                ref_velocity = self.ref_model(x_t, t)
            
            def model_wrapper(x_t, t):
                torch.cuda.set_rng_state(cuda_state)
                output = self.denoiser(x_t, t)
                return output
            
            # Compute average velocity via JVP
            v_t = torch.ones_like(t)
            sigma_end = torch.ones_like(sigma)
            F_avg, F_avg_grad = torch.func.jvp(model_wrapper, (x_t, t), (ref_velocity, v_t))
            F_avg_grad = F_avg_grad.detach()
            F_avg_sg = F_avg.detach()
            # Compute average velocity target
            v_bar = ref_velocity + (sigma_end - sigma) * F_avg_grad
            g = F_avg_sg - v_bar
            # Compute interpolated target with relaxation
            alpha = 1 - sigma  ** 0.5 
            target = F_avg_sg - alpha * g.clamp(min=-1, max=1)

            # Weight CM loss by time
            beta = torch.cos(sigma * np.pi / 2).flatten()
            cm_loss = self.norm_l2_loss(F_avg, target) * beta.flatten()
            loss = cm_loss.mean()
            
        return loss 

    def norm_l2_loss(self, pred, target, p=0.5, c=1e-3):
        """Norm L2 loss with outlier resistance"""
        e = torch.mean((pred - target) ** 2, dim=(1, 2, 3), keepdim=False)
        loss = e / (e + c).pow(p).detach()
        return loss

    def validation_step(self, batch, batch_idx):
        batch = self.get_input(batch)
        val_loss = self(batch)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return val_loss 
    
    def training_step(self, batch, batch_idx):
        batch = self.get_input(batch)
        # print("training step input shape",batch.shape,batch.max(),batch.min())
        loss = self(batch)
        self.log('train_loss', loss,on_step=True, on_epoch=True, prog_bar=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False,prog_bar=True)
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
                latents = self.noise_scheduler.sample_backward(shape,50, self.denoiser, device=device, simple_var=simple_var)
                imgs = self.AE_decode(latents.view((bs,*self.latent_shape))).detach().cpu()
                # imgs expected range [-1,1]
                output_file = os.path.join(output_dir,name)
                channels,h,w = self.image_shape
                save_image(imgs.view(n_sample,channels,h,w),output_file, nrow=3, normalize=True)
    
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.global_step % 100 ==0:
            print(f"train batch {batch_idx} finished")
            output_dir = os.path.join(self.sample_output_dir, f'global_step={self.global_step:05}')
            self.denoiser.eval()
            os.makedirs(output_dir, exist_ok=True)
            with torch.no_grad():
                bs = 9
                shape = (bs,*self.latent_shape)
                if self.loss_type == "simple_consistency_distillation":
                    for sample_steps in [1,2,4,8]:
                        latents = self.noise_scheduler.consistency_sample(shape,sample_steps, self.denoiser, device="cuda")
                        imgs = self.AE_decode(latents.view((bs,*self.latent_shape))).detach().cpu()
                        # imgs expected range [-1,1]
                        name = f"generated_images_samples={sample_steps}steps.png"
                        output_file = os.path.join(output_dir,name)
                        channels,h,w = self.image_shape
                        save_image(imgs.view(bs,channels,h,w),output_file, nrow=3, normalize=True)
                elif self.loss_type == "flow_matching":
                    self.sample_images(output_dir=output_dir,n_sample=9,device="cuda",simple_var=True)
    def on_train_start(self):
        # create sample output dir 
        output_dir = os.path.join(self.sample_output_dir, f'global_step=0')
        os.makedirs(output_dir, exist_ok=True)
        if self.loss_type == "simple_consistency_distillation":
            bs = 9
            shape = (bs,*self.latent_shape)
            for sample_steps in [1,2,4,8]:
                latents = self.noise_scheduler.consistency_sample(shape,sample_steps, self.denoiser, device="cuda")
                imgs = self.AE_decode(latents.view((bs,*self.latent_shape))).detach().cpu()
                # imgs expected range [-1,1]
                name = f"generated_images_samples={sample_steps}steps.png"
                output_file = os.path.join(output_dir,name)
                channels,h,w = self.image_shape
                save_image(imgs.view(bs,channels,h,w),output_file, nrow=3, normalize=True)
        elif self.loss_type == "flow_matching":
            self.sample_images(output_dir=output_dir,n_sample=9,device="cuda",simple_var=True)
        # import pdb; pdb.set_trace()
        
    def on_train_epoch_end(self):
        if (self.current_epoch + 1)  % self.sample_epoch_interval==0:
            output_dir = os.path.join(self.sample_output_dir, f'epoch={self.current_epoch+1:05}')
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
        logger = pl.loggers.TensorBoardLogger("logs/", name=expname)
        log_dir_path = logger.log_dir
        print(f"Log directory: {log_dir_path}")
        sample_output_dir = os.path.join(log_dir_path, "samples")
        model.sample_output_dir = sample_output_dir
        print("model sample output_dir is replaced to ", sample_output_dir)
        # 设置保存 checkpoint 的回调函数
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(log_dir_path,"checkpoints"),  # 保存 checkpoint 的目录
            filename="model-{epoch:02d}-{val_loss:.5f}",  # checkpoint 文件名格式
            # monitor="val_loss",  # 监控的指标，这里使用验证集损失
            # mode="min",  # 指定监控模式为最小化验证集损失
            # save_top_k=3,  # 保存最好的 3 个 checkpoint
            verbose=True
        )
        trainer_config = config.trainer.params
        trainer = pl.Trainer(
            **trainer_config,
            logger=logger,
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