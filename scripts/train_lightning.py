import sys
import os
import argparse
import yaml
import time

import torch
import numpy
import random
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

sys.path.append("../")
from src.models.submodels.interaction_gnn import InteractionGNN
from src.models.submodels.euclidnet import EuclidNet_SO3, EuclidNet_SO2, EuclidNet_SO2_Rec

import wandb

from pytorch_lightning import seed_everything


def set_random_seed(seed):
    torch.random.manual_seed(seed)
    print("Random seed:", seed)
    seed_everything(seed)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("train_gnn.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="default_config.yaml")
    add_arg("root_dir", nargs="?", default=None)
    add_arg("checkpoint", nargs="?", default=None)
    add_arg("random_seed", nargs="?", default=None)
    return parser.parse_args()


def train(config, root_dir, checkpoint, random_seed):
    print("Running train")

    if checkpoint is not None:
        default_configs = torch.load(checkpoint)["hyper_parameters"]
    else:
        default_configs = config

    # Set random seed
    if random_seed is not None:
        set_random_seed(random_seed)
        default_configs["random_seed"] = random_seed

    elif "random_seed" in default_configs.keys():
        set_random_seed(default_configs["random_seed"])

    print("Initialising model")
    print(time.ctime())
    model_name = eval(default_configs["model"])
    model = model_name(default_configs)

    checkpoint_callback = ModelCheckpoint(
        monitor="auc", mode="max", save_top_k=2, save_last=True
    )
    
    wandb.init(project=default_configs["project"], reinit=True) # Need this to avoid logging to the same wandb run
    logger = WandbLogger(
        project=default_configs["project"],
        save_dir=default_configs["artifacts"],
    )
    logger.watch(model, log="all")

    if root_dir is None: 
        if "SLURM_JOB_ID" in os.environ:
            default_root_dir = os.path.join(".", os.environ["SLURM_JOB_ID"])
        else:
            default_root_dir = None
    else:
        default_root_dir = os.path.join(".", root_dir)
        
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = Trainer(
        accelerator = accelerator,
        devices=default_configs["gpus"],
        num_nodes=default_configs["nodes"],
        auto_select_gpus=True,
        max_epochs=default_configs["max_epochs"],
        logger=logger,
        callbacks=[checkpoint_callback],
        default_root_dir=default_root_dir
    )
    trainer.fit(model, ckpt_path=checkpoint)

    return model, trainer

def test(model, trainer):

    return trainer.test(model)

def main():
    print(time.ctime())

    args = parse_args()

    with open(args.config) as file:
        print(f"Using config file: {args.config}")
        default_configs = yaml.load(file, Loader=yaml.FullLoader)

    model, trainer = train(default_configs, args.root_dir, args.checkpoint, args.random_seed)


if __name__ == "__main__":

    main()
