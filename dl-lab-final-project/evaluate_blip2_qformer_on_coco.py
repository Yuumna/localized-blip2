import os
import argparse
import pprint
import random
import math
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lavis.models import load_preprocess
from omegaconf import OmegaConf

from datasets.coco_data_module import CocoCaptionDataModule
from models.localized_blip2_qformer import LocalizedBlip2QFormer
from models.blip2_lightning import Blip2Model
from training_utils import SaveCheckpointAfterValidation, SaveCheckpointAtEpochEnd, make_training_dir
from general_utils import get_env_variables

pp = pprint.PrettyPrinter(indent=4)
evaluating_name = "Task: Evaluate Trained Localized BLIP2-Qformer on COCO"
ENV = get_env_variables()
STORAGE_DIR = ENV["STORAGE_DIR"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(args):
    print(f"Task: {evaluating_name}")

    # print args
    print("args:")
    pp.pprint(args.__dict__)

    # setup and print train config
    train_cfg = OmegaConf.load(args.training_config_file).train
    print("train config:")
    pp.pprint(train_cfg)

    set_seed(train_cfg.seed)

    # setup tensorboard logger
    logger = pl.loggers.TensorBoardLogger("output/lightning_logs/", name="Evaluate_blip2_stage1_qformer_coco")


    # load model and preprocesses
    print("Loading model...")
    start_time = datetime.now()
    model_config = OmegaConf.load(args.model_config_file)
    model = LocalizedBlip2QFormer.from_config(model_config.model)
    vis_preprocess, text_preprocess = load_preprocess(model_config.preprocess)
    print(f"Done loading model. Took {(datetime.now() - start_time).seconds} seconds")

    print("Loading datasets...") 
    start_time = datetime.now()
    
    # define lightning modules
    data_module = CocoCaptionDataModule(
        image_root_dir=os.path.join(STORAGE_DIR, "shared/datasets/coco2014/coco/images"),
        ann_files_dir=os.path.join(STORAGE_DIR, "shared/datasets/coco2014/coco/annotations"),
        vis_transform=vis_preprocess["eval"].transform,
        text_transform=text_preprocess["eval"],
        batch_size=100,
        num_workers=4,
    )
    lightning_model = Blip2Model(
        model,
        test_gt_file=os.path.join(STORAGE_DIR, "shared/datasets/coco2014/coco_gt/coco_karpathy_test_gt.json"),
        test_dataset_format="COCO",
        is_debug=True
    )

    # evaluate model
    trainer = pl.Trainer(accelerator="auto", logger=logger, precision=train_cfg.precision)
    trainer.test(model=lightning_model, datamodule=data_module, ckpt_path=args.ckpt_path)


if __name__ == "__main__":

    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fast_dev_run", action="store_true")
    parser.add_argument("--train_batches_num", type=int, default=None)
    parser.add_argument("--val_batches_num", type=int, default=None)
    parser.add_argument("--test_batches_num", type=int, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--model_config_file", type=str, default="configs/non_localized_qformer_model.yaml")
    parser.add_argument("--training_config_file", type=str, default="configs/localized_qformer_ft_vg.yaml")
    args = parser.parse_args()

    main(args)
