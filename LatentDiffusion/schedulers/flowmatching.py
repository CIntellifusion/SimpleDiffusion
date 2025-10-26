import torch 
from torch import nn 

class FlowMatching(nn.Module):
    def __init__(self):
        super().__init__()
        
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
    def sample_backward(self, image_or_shape,num_inference_step ,net,device="cuda",simple_var=True):
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
        print(f"sigmas shape : {sigmas.shape}")
        for i, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            pred_v = net(x, sigma.reshape(x.shape[0],1))
            print(f"Step {i+1}/{num_inference_step}, sigma: {sigma[0][0][0][0]} -> {sigma_next[0][0][0][0]}")
            pred_end = x + (sigma_end-sigma) * pred_v
            noise = torch.randn_like(x).to(x.device)
            x = sigma_next * pred_end + (1 - sigma_next) * noise 
        return pred_end
    
