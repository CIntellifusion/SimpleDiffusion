import torch 
from torch import nn 

class FlowMatching(nn.Module):
    def __init__(self):
        super().__init__()
    
    def sample_t(self, batch_size, device):
        timestep = torch.rand((batch_size,1),device = device)
        sigma = timestep
        return  timestep, sigma
    
    def sample_forward(self,data , sigma):
        # x0 = noise ; x1 = data 
        # t \in [0,1)
        # from sigma=0 to sigma=1, the noise level decreases
        # linear interpolation 
        noise = torch.randn_like(data).to(data.device)
        xt = sigma * data + (1-sigma) * noise
        # xt' = sigma' * x1 + (1-sigma') * x0
        # xt' - xt = (sigma'-sigma) * x1 + (sigma-sigma') * x0
        #          = (sigma'-sigma) * (x1 - x0)
        # xt' = xt + (sigma'-sigma) * (x1 - x0)
        return xt ,  data-noise  
    
    @torch.no_grad()
    def euler_sample(self, image_or_shape,num_inference_step ,net,device="cuda",simple_var=True):
        # simple_var for compatibility
        if isinstance(image_or_shape,torch.Tensor):
            x = image_or_shape.to(device)
        else:
            x = torch.randn(image_or_shape,device=device)
        sigmas = torch.linspace(0,1,num_inference_step+1,device=device)
        for sigma in sigmas[:-1]:
            pred_v = net(x, sigma.repeat(x.shape[0],1))
            x_next = x + pred_v * 1 / num_inference_step
            x = x_next
        return x
    
    @torch.no_grad()
    def consistency_sample(self, image_or_shape,num_inference_step ,net,device="cuda"):
        # simple_var for compatibility
        if isinstance(image_or_shape,torch.Tensor):
            x = image_or_shape.to(device)
        else:
            x = torch.randn(image_or_shape,device=device)
        sigmas = torch.linspace(0,1,num_inference_step+1,device=device).reshape(-1,1,1,1,1).repeat(1,x.shape[0],1,1,1)
        sigma_end = sigmas[-1]
        for i, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            pred_v = net(x, sigma.reshape(x.shape[0],1))
            pred_end = x + (sigma_end-sigma) * pred_v
            noise = torch.randn_like(x).to(x.device)
            x = sigma_next * pred_end + (1 - sigma_next) * noise 
        return pred_end

class TrigFlow(nn.Module):
    def __init__(self):
        super().__init__()
    
    def sample_t(self, batch_size, device):
        # sigma \in [0,1)
        # timestep \in [0,pi/2)
        sigma = torch.rand((batch_size,1),device = device) 
        timestep = sigma * torch.pi / 2 
        return  timestep, sigma
    
    def sample_forward(self,data , t):
        # t \in [0,pi/2)
        noise = torch.randn_like(data).to(data.device)
        # Interpolation Process
        xt = torch.cos(t) * data + torch.sin(t) * noise
        # x0 = data 
        # x_{pi/2} = noise 
        # xt' = cos(t')*data + sin(t')* noise
        # xt' - xt = (cos(t')-cos(t)) * data + (sin(t')-sin(t))*noise
        # lim t' -> t [(cos(t')-cos(t)) / (t'-t) ] = -sin(t)
        # lim t' -> t [(sin(t')-sin(t))] / (t'-t) ] = cos(t)
        # (xt' - xt ) / (t'-t) =cos(t) * noise  - sin(t) * data  = target
        # xt'= xt+ target * delta t
        # eq.4 in scm paper 
        # Construct Target Velocity
        target = torch.cos(t) * noise - torch.sin(t) * data 
        return xt , target 
    
    @torch.no_grad()
    def euler_sample(self, image_or_shape,num_inference_step ,net,device="cuda",simple_var=True):
        # simple_var for compatibility
        if isinstance(image_or_shape,torch.Tensor):
            x = image_or_shape.to(device)
        else:
            x = torch.randn(image_or_shape,device=device)
        # sigma \in [0,pi/2)
        t = torch.linspace(torch.pi / 2 ,0 ,num_inference_step+1,device=device) 
        for t_cur,t_next in zip(t[:-1],t[1:]):
            sigma_cur = t_cur.repeat(x.shape[0],1) * 2 / torch.pi 
            pred_v = net(x, sigma_cur)
            # print(f"x {x.shape} pred_v {pred_v.shape} sigma_cur {sigma_cur.shape} sigma_next {sigma_next.shape}")
            delta_t = (t_next-t_cur).repeat(x.shape[0],1,1,1)
            x_next = x + pred_v * delta_t
            x = x_next
        return x
    
    @torch.no_grad()
    def consistency_sample(self, image_or_shape,num_inference_step ,net,device="cuda"):
        # simple_var for compatibility
        if isinstance(image_or_shape,torch.Tensor):
            x = image_or_shape.to(device)
        else:
            x = torch.randn(image_or_shape,device=device)
        sigmas = torch.linspace(0,1,num_inference_step+1,device=device).reshape(-1,1,1,1,1).repeat(1,x.shape[0],1,1,1)
        sigma_end = sigmas[-1]
        for i, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            pred_v = net(x, sigma.reshape(x.shape[0],1))
            pred_end = x + (sigma_end-sigma) * pred_v
            noise = torch.randn_like(x).to(x.device)
            x = sigma_next * pred_end + (1 - sigma_next) * noise 
        return pred_end
    
class DiscreteFlowMatching(nn.Module):
    def __init__(self,num_train_steps=1000):
        super().__init__()
        self.num_train_steps = num_train_steps
        self.timesteps = torch.arange(0,num_train_steps+1)
        self.sigmas = self.timesteps / num_train_steps
        print(f"DiscreteFlowMatching with {num_train_steps} steps initialized.")
        print(self.sigmas)
        print(self.timesteps)
    
    def sample_t(self, batch_size, device):
        self.sigmas = self.sigmas.to(device)
        timestep = torch.randint(0, self.num_train_steps, (batch_size,), device=device)
        sigma = self.sigmas[timestep].to(device)
        return timestep,sigma
    
    def sample_forward(self,data , timestep):
        # timestep \in [0,1000)
        # linear interpolation 
        noise = torch.randn_like(data).to(data.device)
        self.sigmas = self.sigmas.to(data.device)
        sigma = self.sigmas[timestep].reshape(-1,1,1,1)
        xt = (1-sigma) * data + sigma * noise
        return xt ,  noise-data
    
    @torch.no_grad()
    def euler_sample(self, image_or_shape,num_inference_step ,net,device="cuda",simple_var=True):
        print("in euler sample",self.num_train_steps,num_inference_step)
        timesteps = torch.linspace(self.num_train_steps,0,num_inference_step+1,device=device).long()
        
        self.sigmas = self.sigmas.to(device)
        if isinstance(image_or_shape,torch.Tensor):
            x = image_or_shape.to(device)
        else:
            x = torch.randn(image_or_shape,device=device)
            
        # 1000 900 800 
        
        for idx, (t_cur, t_next) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            sigma = self.sigmas[t_cur].reshape(-1,1,1,1).repeat(x.shape[0],1,1,1)
            next_sigma = self.sigmas[t_next].reshape(-1,1,1,1).repeat(x.shape[0],1,1,1)
            d_sigma = next_sigma - sigma
            # print(f"Step {idx+1}/{num_inference_step}: t_cur={t_cur.item()}, t_next={t_next.item()}, sigma={sigma[0,0,0,0].item():.4f}, next_sigma={next_sigma[0,0,0,0].item():.4f}, d_sigma={d_sigma[0,0,0,0].item():.4f}")
            pred_v =net(x, sigma.reshape(x.shape[0],1))
            x_next = x + d_sigma *  pred_v
            x = x_next
            
        return x 
    @torch.no_grad()
    def consistency_sample(self, image_or_shape,num_inference_step ,net,device="cuda"):
        # simple_var for compatibility
        if isinstance(image_or_shape,torch.Tensor):
            x = image_or_shape.to(device)
        else:
            x = torch.randn(image_or_shape,device=device)
        sigmas = torch.linspace(0,1,num_inference_step+1,device=device).reshape(-1,1,1,1,1).repeat(1,x.shape[0],1,1,1)
        sigma_end = sigmas[-1]
        for i, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            pred_v = net(x, sigma.reshape(x.shape[0],1))
            pred_end = x + (sigma_end-sigma) * pred_v
            noise = torch.randn_like(x).to(x.device)
            x = sigma_next * pred_end + (1 - sigma_next) * noise 
        return pred_end