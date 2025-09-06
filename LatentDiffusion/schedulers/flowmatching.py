import torch 
from torch import nn 

class FlowMatching(nn.Module):
    def __init__(self, num_timesteps):
        super().__init__()
        self.num_timesteps = num_timesteps
        
    def sample_forward(self, x, t,eps=None):
        # linear interpolation 
        x_0 = torch.randn_like(x)
        t = t.view(-1,1,1,1) / self.num_timesteps
        # print(x.shape,t.shape)
        # import pdb; pdb.set_trace()
        x_t = t * x + (1-t) * x_0 
        return x_t 
    
    def sample_backward_step(self, net, x_t, t):
        pred_v = net(x_t, t)
        x_next = x_t + pred_v / self.num_timesteps
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
