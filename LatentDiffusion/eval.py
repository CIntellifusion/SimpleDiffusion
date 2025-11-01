
import os
from util import instantiate_from_config
import torch
from torchvision.utils import save_image
import argparse
from omegaconf import OmegaConf
import pytorch_lightning as pl

###  parse args 
def parse_args():
    parser = argparse.ArgumentParser(description='Training script')
    ## epoch 200 with loss 0.02 is enough to generate on mnist 
    parser.add_argument('--ckpt', type=str, default=None ,help='ckpt_path')
    parser.add_argument("-o",'--output-dir', type=str, default='./outputs', help='directory to save generated images')
    parser.add_argument("-b", "--base", nargs="*", metavar="configs/train.yaml", help="paths to base configs. Loaded from left-to-right. Parameters can be overwritten or added with command-line options of the form `--key value`.", default=list())
    args = parser.parse_args()
    return args

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
        model.sample_images(args.output_dir,n_sample=81,max_batch_size=9,device="cuda",save_mode='single')