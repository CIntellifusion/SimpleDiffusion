import torch 
from torch import nn 

class FlowMatching(nn.Module):
    def __init__(self, num_timesteps, beta_start, beta_end):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)

    def sample_forward(self, x, t):
        # linear interpolation 
        x_0 = torch.randn_like(x)
        x_t = t * x + (1-t) * x_0 
        return x_t 
    
    def sample_backward_step(self, net, x_t, t):
        pred_v = net(x_t, t)
        x_next = x_t + pred_v / self.num_timesteps
        return x_next

    def sample(self, net, image_or_shape,device="cuda"):
        if isinstance(image_or_shape,torch.Tensor):
            x = image_or_shape.to(device)
        else:
            x = torch.randn(image_or_shape,device=device)
        for t in range(self.num_timesteps):
            x = self.sample_backward_step(net, x, t)
        return x
