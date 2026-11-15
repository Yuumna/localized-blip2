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
training_name = "train_blip2_stage1_qformer_on_vg"
ENV = get_env_variables()
STORAGE_DIR = ENV["STORAGE_DIR"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(args):
    print(f"Task: {training_name}")

    # print args
    print("args:")
    pp.pprint(args.__dict__)

    # setup and print train config
    train_cfg = OmegaConf.load(args.training_config_file).train
    print("train config:")
    pp.pprint(train_cfg)

    set_seed(train_cfg.seed)

    # setup tensorboard logger
    logger = pl.loggers.TensorBoardLogger("output/lightning_logs/", name="train_blip2_stage1_qformer")

    # create subdirectory s.t. different runs don't overwrite each other
    if args.ckpt_path is None:
        ckpt_dir = make_training_dir("output/checkpoints", training_name)
    else:
        if not os.path.exists(args.ckpt_path):
            raise ValueError(f"Checkpoint path {args.ckpt_path} does not exist.")
        ckpt_dir = os.path.dirname(args.ckpt_path)

    # configure callbacks (checkpointing, LR monitor)
    callbacks = [
        SaveCheckpointAfterValidation(training_name, ckpt_dir),
        SaveCheckpointAtEpochEnd(training_name, ckpt_dir),
        ModelCheckpoint(dirpath=ckpt_dir, monitor="val_loss", mode="min", save_top_k=2),
        pl.callbacks.LearningRateMonitor(logging_interval="step")
    ]

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
        text_processor=text_preprocess
        #feed_image_mode="cropped"
    )
    refcocog_val = RefCocoDataset(
        vis_root=os.path.join(STORAGE_DIR, "shared/datasets/coco2014/coco/images"),
        instances_file=os.path.join(STORAGE_DIR, "shared/datasets/refcocog/instances.json"),
        refs_file=os.path.join(STORAGE_DIR, "shared/datasets/refcocog/refs(umd).p"),
        dataset_name="refcoco_g",
        set_type="val",
        vis_transform=vis_preprocess["eval"],
        text_transform=text_preprocess["eval"],
    )

    vg_train_dataloader = DataLoader(
        vg_dataset.get_split("train"),
        batch_size=train_cfg.batch_size_train,
        num_workers=train_cfg.num_workers,
        shuffle=True,
        collate_fn=VGDataset.collate_fn
    )
    vg_val_dataloader = DataLoader(
        vg_dataset.get_split("val"),
        batch_size=train_cfg.batch_size_eval,
        num_workers=train_cfg.num_workers,
        collate_fn=VGDataset.collate_fn
    )
    refcocog_val_dataloader = DataLoader(
        refcocog_val,
        batch_size=train_cfg.batch_size_eval,
        num_workers=train_cfg.num_workers,
        collate_fn=RefCocoDataset.collate_fn
    )
    print(f"Done loading datasets. Took {(datetime.now() - start_time).seconds} seconds")

    num_training_steps = None
    if args.train_batches_num:
        num_training_steps = math.ceil(args.train_batches_num / train_cfg.grad_accumulate_batches)
    else:
        num_training_steps = math.ceil(len(vg_train_dataloader) / train_cfg.grad_accumulate_batches)
    print(f"num_training_steps: {num_training_steps}")

    lightning_model = Blip2Model(
        model=model,
        train_cfg=train_cfg,
        ca_num_steps=num_training_steps,
        val_gt_file=vg_dataset.get_ground_truth_file(split="val"),
        val_dataset_format="VG",
        run_test_every_n_val_iters=train_cfg.run_test_every_n_val_iters,
        test_gt_file=os.path.join(STORAGE_DIR, train_cfg.test_gt_file),
        test_dataset_format=train_cfg.test_dataset_format,
        training_name=training_name,
        is_debug= True
    )

    # train model
    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=train_cfg.max_epoch,
        accumulate_grad_batches=train_cfg.grad_accumulate_batches,
        val_check_interval=train_cfg.val_check_interval,
        check_val_every_n_epoch=train_cfg.check_val_every_n_epoch,
        log_every_n_steps=train_cfg.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
        precision=train_cfg.precision,
        num_sanity_val_steps=0,
                         
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
    """"
    trainer.test(
        model=lightning_model,
        dataloaders=[refcocog_val_dataloader],
        ckpt_path=args.ckpt_path
    )
    trainer.fit(
        model=lightning_model,
        train_dataloaders=vg_train_dataloader,
        val_dataloaders=[vg_val_dataloader, refcocog_val_dataloader],
        ckpt_path=args.ckpt_path,
    )
    trainer.test(
        model=lightning_model,
        dataloaders=[refcocog_val_dataloader],
    )
    """

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
    args = parser.parse_args()

    main(args)
