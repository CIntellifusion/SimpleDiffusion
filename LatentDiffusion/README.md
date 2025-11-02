# Flow Matching with different time scheduler 

## support detailed log for large scale experiment 

-  training with different configs to conduct comparable experiments. 

# Support Simplify Consistent Distillation(sCM)

## Learning Rate and Learning Rate scheduler

The sCM training is very inconsistent due to many issues. To stablize training process: 
1. we need a relative low learning rate 
2. we need a proper learning rate scheduler for long-term training. 

Existing issue may include:
1. Training collapse after several thousand training iterations. 
2. Gradient Norm maybe vary at a large scale from 10 to 10^3

## Model Archtecture 

1. QK normalization in DiT archtecture. 
2. Use ema_model. 

### EMA model 
Follow the original formula in sCM we need a EMA model to ensure stablize training. 
Here we use a naive EMA instead of PostHocEMA(that was created but not used in FACM).
Using a EMA model in JVP significantly reduces the training memory, from 121GiB to 37GiB while all other setting remains the same. The reason is still unknown.  The training loss and grad norm also smaller and more stable. But the training speed decreases from 2.8 it/sec to 2.3 it/sec. 

## sCM hyperparameter 

1. we adjust r_factor_max from 1 to 0.5 to slow down training collapse time. 
 

## Distillation and Training from Scratch 

- [ ]distill consistency model from a pretrained flow matching model 
- [ ]train consistency model from scratch 


## Experiments Results 

### test training parameters 

- [ ] 200 epoch for pretrain on celeba 
- [ ] triaining with facm implementation and haoyu's implementation. training with and without ema_model. 


### FID 
FM: Frechet Inception Distance: 9.877043774564868
SCM(FACM implementation): reaches 11-12 or so on 1-step training. 

# Code Issues 

The distributed behavior in pytorch lightning 
1. the train_epoch_end is called by every process, other hood function the same. 
2. when sample images, sample function is executed on each device in parallel, just make sure the save file name are differnt for each process. 
3. usually don't explicit call barraier, which is kind of different from the frame work of accelerator. 
4. To customize on cluster training, just set `devices=auto` for different num of devices. 
5. `self.global_rank` could easily get current device id. We can easily control some behavior that you only want excuted on rank_zero. @zero_zero_only decorator is another option. 
