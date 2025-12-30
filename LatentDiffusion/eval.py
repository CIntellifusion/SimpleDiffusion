
import os
from util import instantiate_from_config
import torch
from torchvision.utils import save_image
import argparse
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch_fidelity

###  parse args 
def parse_args():
    parser = argparse.ArgumentParser(description='Training script')
    ## epoch 200 with loss 0.02 is enough to generate on mnist 
    parser.add_argument('--ckpt', type=str, default=None ,help='ckpt_path')
    parser.add_argument('--num_inference_steps', type=int, default=50 ,help='num_inference_steps')
    parser.add_argument("-o",'--output-dir', type=str, default='./outputs', help='directory to save generated images')
    parser.add_argument("-gt",'--ground-truth-dir', type=str, default='./outputs', help='directory to save generated images')
    parser.add_argument("-b", "--base", nargs="*", metavar="configs/train.yaml", help="paths to base configs. Loaded from left-to-right. Parameters can be overwritten or added with command-line options of the form `--key value`.", default=list())
    args = parser.parse_args()
    return args

def create_ground_truth_images(data_module,gt_dir):
    os.makedirs(gt_dir,exist_ok=True)
    train_dataloader = data_module.train_dataloader()
    for i, batch in enumerate(train_dataloader):
        imgs, _ = batch
        bs = imgs.shape[0]
        for j in range(bs):
            name = f"gt_samples={i*bs+j}.png"
            output_file = os.path.join(gt_dir,name)
            save_image(imgs[j],output_file, normalize=True)
            if i*bs + j >= 5000:
                return

#python eval.py -b configs/train_pixel_fm.yaml --ckpt checkpoints/fm_celeba_200epoch.ckpt -o samples/eval_fm_200epoch -gt ground-truth-celeba
#python eval.py -b configs/train_pixel_scm.yaml --ckpt logs/FixScmCollapse_1e-5_rmax0.3/version_0/checkpoints/model-epoch=04-val_loss=0.03708.ckpt -o samples/eval_scm_5epoch -gt ground-truth-celeba
if __name__ == "__main__":
    args= parse_args()
    configs = [OmegaConf.load(cfg) for cfg in args.base]
    # cli = OmegaConf.from_dotlist(unknown)
    # config = OmegaConf.merge(*configs, cli)
    config = OmegaConf.merge(*configs)
    model = instantiate_from_config(config.model)
    trainer_config = config.trainer.params
        
    trainer = pl.Trainer(
        **trainer_config,
        logger=False,  # 关闭日志记录器
    )
    if args.ckpt is not None: 
        model = instantiate_from_config(config.model)
        os.makedirs(args.output_dir,exist_ok=True)
        model.fix_val_noise = False 
        model.load_state_dict(torch.load(args.ckpt,weights_only=False)['state_dict'],strict=True)
        model.sample_images(args.output_dir,n_sample=5000,max_batch_size=10,num_inference_step=args.num_inference_steps,device="cuda",save_mode='single')

    # create ground truth images if necessary
    if not os.path.exists(os.path.join(args.ground_truth_dir,"gt_samples=0.png")):
        data_module = instantiate_from_config(config.data)
        data_module.prepare_data()
        data_module.setup('fit')
        create_ground_truth_images(data_module,args.ground_truth_dir)
        
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=args.output_dir,
        input2=args.ground_truth_dir,
        fid=True
    )
    
    print(metrics_dict)