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

from models.localized_blip2_qformer import LocalizedBlip2QFormer
from models.blip2_lightning import Blip2Model
from datasets.visual_genome_dataset import VGDataset
from datasets.refcoco_dataset import RefCocoDataset
from training_utils import SaveCheckpointAfterValidation, SaveCheckpointAtEpochEnd, make_training_dir
from general_utils import get_env_variables

pp = pprint.PrettyPrinter(indent=4)
evaluating_name = "Task: Evaluate Untrained BLIP2-Qformer on VG"
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
    logger = pl.loggers.TensorBoardLogger("output/lightning_logs/", name="evaluate qformer on vg")

    # create subdirectory s.t. different runs don't overwrite each other
    if args.ckpt_path is None:
        ckpt_dir = make_training_dir("output/checkpoints", evaluating_name)
    else:
        if not os.path.exists(args.ckpt_path):
            raise ValueError(f"Checkpoint path {args.ckpt_path} does not exist.")
        ckpt_dir = os.path.dirname(args.ckpt_path)

    # load model and preprocesses
    print("Loading model...")
    start_time = datetime.now()
    model_config = OmegaConf.load(args.model_config_file)
    model = LocalizedBlip2QFormer.from_config(model_config.model)
    vis_preprocess, text_preprocess = load_preprocess(model_config.preprocess)
    print(f"Done loading model. Took {(datetime.now() - start_time).seconds} seconds")

    print("Loading datasets...") 
    start_time = datetime.now()
    vg_dataset = VGDataset(
        image_dir=os.path.join(STORAGE_DIR, "shared/datasets/vg/images"),
        index_file=os.path.join(STORAGE_DIR, "shared/datasets/vg/annotations/vg_bigger_than_median.json"),
        vis_processor=vis_preprocess,
        text_processor=text_preprocess,
        feed_image_mode=args.feed_image_mode
    )

    vg_val_dataloader = DataLoader(
        vg_dataset.get_split("val"),
        batch_size=train_cfg.batch_size_eval,
        num_workers=train_cfg.num_workers,
        collate_fn=VGDataset.collate_fn
    )

    print(f"Done loading datasets. Took {(datetime.now() - start_time).seconds} seconds")

    lightning_model = Blip2Model(
        model=model,
        val_gt_file=vg_dataset.get_ground_truth_file(split="val"),
        val_dataset_format="VG",
        training_name=evaluating_name,
        is_debug=True
    )

    # train model
    trainer = pl.Trainer(
        accelerator="auto",
        logger=logger,
        precision=train_cfg.precision,
                         
        # debug settings
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=args.train_batches_num,
        limit_val_batches=args.val_batches_num,
        limit_test_batches=args.test_batches_num,
    )
    trainer.validate(
        model=lightning_model,
        dataloaders=[vg_val_dataloader],
        ckpt_path=args.ckpt_path
    )



if __name__ == "__main__":

    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fast_dev_run", action="store_true")
    parser.add_argument("--train_batches_num", type=int, default=None)
    parser.add_argument("--val_batches_num", type=int, default=None)
    parser.add_argument("--test_batches_num", type=int, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--model_config_file", type=str, default="configs/localized_qformer_model.yaml")
    parser.add_argument("--training_config_file", type=str, default="configs/localized_qformer_ft_vg.yaml")
    parser.add_argument('--feed_image_mode', type=str, default='full', choices=['full', 'cropped', 'blackout'],
                        help='Mode for image handling. Options are "full", "cropped", and "blackout".')
    args = parser.parse_args()

    main(args)
