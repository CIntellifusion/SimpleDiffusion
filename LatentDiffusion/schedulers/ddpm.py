
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
