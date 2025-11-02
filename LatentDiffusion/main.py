"""
autor: haoyu
date: 20240501-0506
an simplified unconditional diffusion for image generation
"""
import os
import argparse
from tqdm import trange
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
from util import get_obj_from_str
import torch_fidelity
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
                 imsize=32,
                 channels = 1,
                 lr_scheduler_config = {},
                 optimizer_params={},
                 sample_output_dir = "./samples",
                 sample_epoch_interval = 20,
                 sample_step_interval = 1000,
                 noise_scheduler_config = {},
                 model_config = {},
                 vae_config = {},
                 vae_pretrained_path = '',
                 model_pretrained_path = '',
                 loss_type="flow_matching",
                 fix_val_noise=True,
                 timestep_inverse=False,
                 r_config = None,
                 scheduler_eps= 0.0 
                 ):
        super(LatentDiffusion, self).__init__()
        self.save_hyperparameters()  # Save hyperparameters for logging
        image_shape = [channels,imsize,imsize]
        self.noise_scheduler = instantiate_from_config(noise_scheduler_config)
        self.vae = instantiate_from_config(vae_config)
        self.latent_shape = image_shape

        self.denoiser = instantiate_from_config(model_config)
        self.vae_config = vae_config
        self.config_vae(vae_pretrained_path)
        
        self.criterion = nn.MSELoss()
        self.optimizer_params = optimizer_params 
        self.lr_scheduler_config = lr_scheduler_config
        self.image_shape = image_shape

        self.sample_output_dir = sample_output_dir
        self.sample_epoch_interval = sample_epoch_interval
        self.sample_step_interval = sample_step_interval
        
        self.loss_type = loss_type
        if model_pretrained_path != '':
            self.load_state_dict(torch.load(model_pretrained_path,weights_only=False)['state_dict'],strict=True)
        
        if "simple_consistency" in self.loss_type:
            if self.loss_type == "simple_consistency_distillation":
                print(f"creating reference model for consistency distillation")
                self.ref_model = instantiate_from_config(model_config)
                denoiser_state_dict = self.denoiser.state_dict()
                self.ref_model.load_state_dict(denoiser_state_dict,strict=True)
                self.ref_model.eval()
                for param in self.ref_model.parameters():
                    param.requires_grad = False
            elif self.loss_type =="simple_consistency_training":
                print('simple consistency training')
            else:
                raise NotImplementedError(f"unknown simple consistency loss type {self.loss_type}")
            self.r_config = r_config
            assert self.r_config is not None
            self.scheduler_eps = scheduler_eps 
            
            # self.create_scm_adaptive_weight_on_sigma()
        self.fix_val_noise = fix_val_noise
        self.val_noise = None
        
        self.inverse=timestep_inverse # from 0 - 1 or from 1 - 0 
        
    def create_scm_adaptive_weight_on_sigma(self):
        from models.dit import scm_AdaLoss
        self.adaptive_t_weight=scm_AdaLoss(
            in_channels=self.denoiser.hidden_size,
        )
        
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
        scheduler_class = self.lr_scheduler_config.get("target", None)
        optimizer = torch.optim.AdamW(self.denoiser.parameters(), lr=self.optimizer_params.lr, weight_decay=self.optimizer_params.weight_decay, betas=(0.9, self.optimizer_params.beta), eps=self.optimizer_params.eps)
        if scheduler_class is None: 
            return optimizer
        
        scheduler_class = get_obj_from_str(scheduler_class)
        scheduler_params = self.lr_scheduler_config.get("params", {})
        scheduler = scheduler_class(optimizer,**scheduler_params)
        return {
                'optimizer': optimizer,
                'lr_scheduler': {
                'scheduler':   scheduler,
                    'interval': 'step',  
                    'frequency': 1,     
                }
        }

    def AE_encode(self,x):
        # x = torch.concat(x)
        # print("AE_encode",x.shape)# [128, 1, 28, 28]
        # bs = x.shape[0]
        # x = x.view(bs, *self.latent_shape)
        return self.vae.encode(x) # encoder posterior ; tensor 
    
    def AE_decode(self,x):
        return self.vae.decode(x) 
            
    def get_input(self,batch):
        images,class_label = batch
        # print(f"class label : {class_label}")
        return self.AE_encode(images)
    
    def denoiser_wrapper(self,x_t,t):
        if self.inverse:
            return -self.denoiser(x_t,1-t)
        else:
            return self.denoiser(x_t,t)
        
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
            t ,sigma = self.noise_scheduler.sample_t(bs, latents.device)
            x_t,target = self.noise_scheduler.sample_forward(latents, t.reshape(bs,1,1,1))
            # print(f"t shape: {t.shape}, sigma shape: {sigma.shape} ")
            model_output = self.denoiser(x_t, sigma.reshape(bs,1))
            loss = self.criterion(target,model_output)

        elif "simple_consistency" in loss_type:
            bs = latents.shape[0]
            def model_wrapper(x_t, t):
                torch.cuda.set_rng_state(cuda_state)
                output = self.denoiser_wrapper(x_t, t)
                return output
            # get velocity: need ref_velocity and cuda_state 
            if self.loss_type == "simple_consistency_training":
                cuda_state = torch.cuda.get_rng_state()
                t ,sigma = self.noise_scheduler.sample_t(bs, latents.device)
                # t(o-pi/2) sigma(0,1)
                # sigma = t.reshape(bs,1,1,1)
                t = t.reshape(bs,1,1,1)
                x_t,target = self.noise_scheduler.sample_forward(latents, t.reshape(bs,1,1,1))
                ref_velocity = target 
                
                # Compute average velocity via JVP
                v_sigma = torch.ones_like(sigma)
                sigma_end = torch.ones_like(sigma)
                F_avg, F_avg_grad = torch.func.jvp(model_wrapper, (x_t, sigma), (ref_velocity, v_sigma))
                F_avg_grad = F_avg_grad.detach()
                F_avg_sg = F_avg.detach()
                # Compute average velocity target
                r_factor = min(self.r_config.r_factor_max, self.global_step / self.r_config.r_factor_warmup_steps) 
                g = -torch.cos(t) * torch.cos(t) * (F_avg_sg - ref_velocity) - \
                    (torch.cos(t) * torch.sin(t)) * (F_avg_grad + x_t) * r_factor
                # Compute interpolated target with relaxation
                target = F_avg_sg + g.clamp(min=-1, max=1)
                # Weight CM loss by time
                cm_loss = self.norm_l2_loss(F_avg, target)
                # cm_loss = self.adaptive_t_weight(sigma,cm_loss)
                loss = cm_loss.mean()
                
            elif self.loss_type == "simple_consistency_distillation":
                t = torch.rand((bs,1),device = latents.device)
                t = torch.clamp(t, self.scheduler_eps ,1.0 - self.scheduler_eps)
                # re-name
                data = latents
                sigma = t.reshape(bs,1,1,1)
                # sample forward 
                noise = torch.randn_like(data).to(data.device)
                # print(f"noise size : {noise.shape}, data size: {data.shape} sigma size: {sigma.shape}")
                x_t = sigma * data + (1-sigma)  * noise
                target = data - noise
                assert hasattr(self,"ref_model")
                
                with torch.no_grad():
                    cuda_state = torch.cuda.get_rng_state()
                    if self.inverse:
                        ref_velocity = -self.ref_model(x_t,1 - t)
                    else:
                        ref_velocity = self.ref_model(x_t, t)
                        
                # Compute average velocity via JVP
                v_t = torch.ones_like(t)
                sigma_end = torch.ones_like(sigma)
                F_avg, F_avg_grad = torch.func.jvp(model_wrapper, (x_t, t), (ref_velocity, v_t))
                F_avg_grad = F_avg_grad.detach()
                F_avg_sg = F_avg.detach()
                # Compute average velocity target
                r_factor = min(self.r_config.r_factor_max, self.global_step / self.r_config.r_factor_warmup_steps) 
                v_bar = ref_velocity + (sigma_end - sigma) * F_avg_grad * r_factor
                g = F_avg_sg - v_bar 
                # Compute interpolated target with relaxation
                alpha = torch.sin(sigma * np.pi / 2).reshape(bs,1,1,1) # 1-cosa**2 = sin**2 < sina < 1
                target = F_avg_sg - alpha * g.clamp(min=-1, max=1)
                # Weight CM loss by time
                beta = torch.cos(sigma * np.pi / 2).flatten()
                cm_loss = self.norm_l2_loss(F_avg, target) * beta
                # cm_loss = self.adaptive_t_weight(sigma,cm_loss)
                loss = cm_loss.mean()
            else:
                raise NotImplementedError(f"unknown simple consistency loss type {self.loss_type}")
           
            

        return loss 

    def norm_l2_loss(self, pred, target, p=0.5, c=1e-2):
        """Norm L2 loss with outlier resistance
        changelog: training before 2025-11-2 using c = 1e-3, after 1e-2
        1e-2 provided in sCM paper,
        before following facm: loss = e / (e + c).pow(p).detach()
        after following algo1 in scm paper: g = (||g||+c)
        """
        e = torch.mean((pred - target) ** 2, dim=(1, 2, 3), keepdim=False)
        # loss = e / (e + c).pow(p).detach()
        loss = e / (e.pow(p).detach() + c)
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

    def sample_images(self, output_dir, n_sample=9, device="cuda", simple_var=True,max_batch_size = 32,num_inference_step = 50,save_mode='batch'):
        self.to(device)
        os.makedirs(output_dir, exist_ok=True)
        self.denoiser.eval()
        global_rank = self.global_rank

        for i in trange(0, n_sample, max_batch_size, leave=False):
            # shape = (min(max_batch_size, n_sample - i),*self.image_shape)
            bs = min(max_batch_size, n_sample - i)
            # print(f"bs {bs}")
            shape = (bs,*self.latent_shape)
            if self.fix_val_noise:
                if self.val_noise is None:
                    shape = (bs,*self.latent_shape)
                    print(f'generating noise in rank={global_rank} with shape {shape}')
                    noise = torch.randn(shape,device=device)
                    self.val_noise = noise.cpu()
                else:
                    noise = self.val_noise.to(device)
                    assert noise.shape == (bs, *self.latent_shape)
                image_or_shape = noise
            else:
                image_or_shape = shape
                
            if "simple_consistency" in self.loss_type:
                sampling_method="consistency"
                latents = self.noise_scheduler.consistency_sample(image_or_shape=image_or_shape,
                                                                    num_inference_step=num_inference_step,
                                                                    net = self.denoiser_wrapper, 
                                                                    device="cuda")
            elif self.loss_type == "flow_matching":
                sampling_method="euler"
                latents = self.noise_scheduler.euler_sample(image_or_shape=image_or_shape,
                                                        num_inference_step=num_inference_step, 
                                                        net=self.denoiser, 
                                                        device=device, 
                                                        simple_var=simple_var)
            
            # imgs expected range [-1,1]
            imgs = self.AE_decode(latents.view((bs,*self.latent_shape))).detach().cpu()
            
            # save file 
            if save_mode == 'batch':
                name = f"method={sampling_method}_steps={num_inference_step}_samples={i}-{i+bs}_rank={global_rank}.png"
                output_file = os.path.join(output_dir,name)
                channels,h,w = self.image_shape
                save_image(imgs.view(bs,channels,h,w),output_file, nrow=3, normalize=True)
            elif save_mode == 'single':
                channels,h,w = self.image_shape
                for j in range(bs):
                    name = f"method={sampling_method}_steps={num_inference_step}_samples={i+j}_rank={global_rank}.png"
                    output_file = os.path.join(output_dir,name)
                    save_image(imgs[j].view(channels,h,w),output_file, normalize=True)
        
        self.denoiser.train()
        
    def fid_evaluation(self,n_sample=5000):
        old_fix_val_noise = self.fix_val_noise
        self.fix_val_noise = False # for val fid
        fid_eval_folder = os.path.join(self.sample_output_dir,"fid-eval",f'epoch={self.current_epoch}')
        inference_steps = [20,30,50] if self.loss_type == "flow_matching" else [1,2,4,8]
        
        for num_inference_step in inference_steps:
            output_dir = os.path.join(fid_eval_folder, f'inference_step={num_inference_step}')
            os.makedirs(output_dir, exist_ok=True)
            # TODO: sample on each device parallel to save time. 
            n_sample_per_device = n_sample // max(1, self.trainer.num_devices)
            self.sample_images(output_dir=output_dir,n_sample=n_sample_per_device,simple_var=True,max_batch_size=10,num_inference_step=num_inference_step,save_mode='single')
            if self.global_rank ==0:
                metrics_dict = torch_fidelity.calculate_metrics(
                    input1=output_dir,
                    input2="ground-truth-celeba",
                    fid=True
                )
                try:
                    self.log(f"fid/num_inference_step={num_inference_step}",metrics_dict['frechet_inception_distance'],sync_dist=True)
                except:
                    print("logging in tensorboard FID failed")
                output_fid_file = os.path.join(fid_eval_folder, f'fid_metric.txt')
                with open(output_fid_file,'a') as f:
                    f.write(f"epoch: {self.current_epoch} num_inference_step: {num_inference_step} FID: {metrics_dict['frechet_inception_distance']}\n")
                # self.log(f"num_inference_step={num_inference_step} fid",metrics_dict['frechet_inception_distance'])
                print(f"FID after training(num_inference_step={num_inference_step}): ",metrics_dict['frechet_inception_distance'])
        self.fix_val_noise = old_fix_val_noise
       
    def on_train_batch_end(self, outputs, batch, batch_idx):
        inference_steps = [20,30,50] if self.loss_type == "flow_matching" else [1,2,4,8]
        if self.global_step % self.sample_step_interval == 0:
            latents = self.get_input(batch) 
            # print(f"latents shape and max min : {latents.shape} {latents.max()} {latents.min()}")
            save_image(self.AE_decode(latents).detach().cpu(),os.path.join(self.sample_output_dir, f"ground_truth_step={self.global_step}.png"),nrow=4,normalize=True)
            print(f"train epoch {self.current_epoch} batch {batch_idx} finished , global step {self.global_step}")
            visualize_training_data_dir = os.path.join(self.sample_output_dir, "training_data_visualization")
            os.makedirs(visualize_training_data_dir, exist_ok=True)
            output_dir = os.path.join(visualize_training_data_dir, f'global_step={self.global_step:05}')
            os.makedirs(output_dir, exist_ok=True)
            for num_inference_step in inference_steps:
                self.sample_images(output_dir=output_dir,n_sample=9,device="cuda",simple_var=True,max_batch_size=9,num_inference_step=num_inference_step)
            
    def on_train_start(self):
        # create sample output dir 
        inference_steps = [20,30,50] if self.loss_type == "flow_matching" else [1,2,4,8]
        output_dir = os.path.join(self.sample_output_dir, f'global_step=00000')
        os.makedirs(output_dir, exist_ok=True)
        for num_inference_step in inference_steps:
            self.sample_images(output_dir=output_dir,n_sample=9,device="cuda",simple_var=True,max_batch_size=9,num_inference_step=num_inference_step)
            
    def on_train_epoch_end(self):
        inference_steps = [20,30,50] if self.loss_type == "flow_matching" else [1,2,4,8]
        if (self.current_epoch + 1)  % self.sample_epoch_interval==0 :
            self.fid_evaluation()
        else:
            output_dir = os.path.join(self.sample_output_dir, f'epoch={self.current_epoch+1:05}')
            for num_inference_step in inference_steps:
                self.sample_images(output_dir=output_dir,n_sample=9,device="cuda",simple_var=True,max_batch_size=9,num_inference_step=num_inference_step)
        
    
    def on_after_backward(self):
        # 在反向传播之后计算梯度范数
        if self.global_step % 100 != 0:
            return
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        self.log('grad_norm', total_norm, on_step=True, on_epoch=True, prog_bar=True)
 
    def on_fit_end(self):
        self.fid_evaluation()
        
###  parse args 
def parse_args():
    parser = argparse.ArgumentParser(description='Training script')
    ## epoch 200 with loss 0.02 is enough to generate on mnist 
    parser.add_argument('--expname', type=str, default=None ,help='expname of this experiment')
    parser.add_argument('--ckpt', type=str, default=None ,help='ckpt_path')
    parser.add_argument('--train', action='store_true', help='Whether to run in training mode')
    parser.add_argument('--auto-resume', action='store_true', help='whether resume from trained checkpoint ')
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
    # update config name 
    if args.expname is not None:
        config.expname = args.expname
    
    expname = config.expname
    imsize = config.imsize
    
    if args.train:
        data_module = instantiate_from_config(config.data)
        data_module.prepare_data()
        data_module.setup()
        
        model = instantiate_from_config(config.model)
        
        # 从配置中获取训练器参数
        trainer_config_dict = config.trainer.params
        
        # 自定义版本号，例如使用时间戳
        import time
        version = time.strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join("logs", expname, version)
        
        # 创建 logger，指定版本
        logger = pl.loggers.TensorBoardLogger("logs", name=expname, version=version)
        # 此时 logger.log_dir 应该等于 log_dir
        assert logger.log_dir == log_dir
        # 使用环境变量判断当前 rank
        global_rank = int(os.environ.get("RANK", 0))
        
        # 只在 rank 0 上创建目录和打印信息
        if global_rank == 0:
            print(f"Log directory: {log_dir}")
            sample_output_dir = os.path.join(log_dir, "samples")
            os.makedirs(sample_output_dir, exist_ok=True)
            print("model sample output_dir is replaced to ", sample_output_dir)
        else:
            sample_output_dir = os.path.join(log_dir, "samples")
        
        model.sample_output_dir = sample_output_dir
        
        # 设置保存 checkpoint 的回调函数
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(log_dir, "checkpoints"),
            filename="model-{epoch:02d}-{val_loss:.5f}",
            monitor="val_loss",
            mode="min",
            save_top_k=30,
            verbose=True
        )
        
        trainer = pl.Trainer(
            **trainer_config_dict,
            logger=logger,
            callbacks=[checkpoint_callback],
        )
        
        if config.pretrain_path != "None":
            pretrain_path = config.pretrain_path
        else:
            pretrain_path = None 
        
        trainer.fit(model, data_module, ckpt_path=pretrain_path)
    else:
        if args.ckpt is not None: 
            model = instantiate_from_config(config.model)
            sample_dir = f'./samples/{args.expname}'
            os.makedirs(sample_dir,exist_ok=True)
            model.fix_val_noise = False 
            model.load_state_dict(torch.load(args.ckpt,weights_only=False)['state_dict'],strict=True)
            model.sample_images(sample_dir,n_sample=81,max_batch_size=9,device="cuda")
        else:
            ckpt_folder = f"./logs/{expname}"
            paths = os.listdir(ckpt_folder)
            paths = [os.path.join(ckpt_folder,i) for i in paths]
            # paths = ["/home/haoyu/research/simplemodels/SimpleDiffusion/UnconditionalDiffusion/checkpoints/linear_normal/model-epoch=1184-val_loss=0.00332.ckpt"]
            for path in paths:
                ckpt = os.path.basename(path).replace(".ckpt","")
                model = instantiate_from_config(config.model)
                model.load_state_dict(torch.load(path,weights_only=False)['state_dict'],strict=True)
                model.sample_images(f'./sample/{ckpt}',n_sample=32,device="cuda:0")