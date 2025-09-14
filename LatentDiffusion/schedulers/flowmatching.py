import torch 
from torch import nn 

class FlowMatching(nn.Module):
    def __init__(self, num_timesteps):
        super().__init__()
        self.num_timesteps = num_timesteps
        
    def sample_forward(self, x1, t):
        # x0 = noise ; x1 = data 
        # t \in [0,N)
        # from sigma=0 to sigma=1, the noise level decreases
        # linear interpolation 
        x0 = torch.randn_like(x1).to(x1.device)
        sigma = t.view(-1,1,1,1) / self.num_timesteps
        xt = sigma * x1 + (1-sigma) * x0
        # xt' = sigma' * x1 + (1-sigma') * x0
        # xt' - xt = (sigma'-sigma) * x1 + (sigma-sigma') * x0
        #        = (sigma'-sigma) * (x1 - x0)
        # xt' = xt + (sigma'-sigma) * (x1 - x0)
        return xt ,  x1-x0  
    
    def sample_backward_step(self, net, xt, t):
        pred_v = net(xt, t)
        x_next = xt + pred_v / self.num_timesteps
        return x_next
    
    @torch.no_grad()
    def sample_backward(self, image_or_shape ,net,device="cuda",simple_var=True):
        if isinstance(image_or_shape,torch.Tensor):
            x = image_or_shape.to(device)
        else:
            x = torch.randn(image_or_shape,device=device)
        for t in range(self.num_timesteps):
            t = torch.ones(x.shape[0],dtype=torch.long,device=x.device).reshape(-1,1) * t
            x = self.sample_backward_step(net, x, t)
        return x
