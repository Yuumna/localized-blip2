import argparse
import pprint
import random

import torch
from torch.utils.data import DataLoader
import numpy as np
import lightning.pytorch as pl
from lavis.models import load_preprocess
from omegaconf import OmegaConf

from models.localized_blip2_qformer import LocalizedBlip2QFormer
from models.blip2_lightning import Blip2Model
from datasets.refcoco_dataset import RefCocoDataset

pp = pprint.PrettyPrinter(indent=4)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(args):
    print("Task: Train BLIP2-Qformer on RefCoco+ (to be updated to VG)")
    set_seed(42)

    # print args
    print("args:")
    pp.pprint(args.__dict__)

    # setup and print train config
    train_cfg = OmegaConf.load(args.config_file).train
    print("train config:")
    pp.pprint(train_cfg)

    # setup tensorboard logger
    logger = pl.loggers.TensorBoardLogger("lightning_logs/", name="train_blip2_stage1_qformer")

    # load model and preprocesses
    model_name = "blip2"
    model_type = "pretrain"     # pretrain / pretrain_vitL / coco
    model = LocalizedBlip2QFormer.from_pretrained(model_type=model_type)
    cfg = OmegaConf.load(LocalizedBlip2QFormer.default_config_path(model_type)).preprocess
    vis_preprocess, text_preprocess = load_preprocess(cfg)

    # define lightning modules
    refcoco_train = RefCocoDataset(
        vis_root="./datasets/coco2014/coco/images",
        instances_file="./datasets/refcoco+/instances.json",
        refs_file="./datasets/refcoco+/refs(unc).p",
        dataset_name="refcoco+",
        set_type="train",
        vis_transform=vis_preprocess["train"],
        text_transform=text_preprocess["train"],
    )
    train_dataloader = DataLoader(refcoco_train, batch_size=32, num_workers=4, collate_fn=RefCocoDataset.collate_fn)

    refcoco_val = RefCocoDataset(
        vis_root="./datasets/coco2014/coco/images",
        instances_file="./datasets/refcoco+/instances.json",
        refs_file="./datasets/refcoco+/refs(unc).p",
        dataset_name="refcoco+",
        set_type="val",
        vis_transform=vis_preprocess["eval"],
        text_transform=text_preprocess["eval"],
    )
    val_dataloader = DataLoader(refcoco_val, batch_size=32, num_workers=4, collate_fn=RefCocoDataset.collate_fn)

    lightning_model = Blip2Model(
        model,
        train_cfg,
        val_gt_file="./datasets/refcoco+/refs(unc).p",
        val_dataset_format="RefCOCO+"
    )


    # train model
    trainer = pl.Trainer(accelerator="auto",
                         max_epochs=args.max_epochs,
                         limit_train_batches=args.limit_train_samples_ratio,
                         limit_val_batches=args.limit_val_samples_ratio,
                         limit_test_batches=args.limit_test_samples_ratio,
                         val_check_interval=train_cfg.val_check_interval,
                         check_val_every_n_epoch=train_cfg.check_val_every_n_epoch,
                         log_every_n_steps=train_cfg.log_every_n_steps,
                         logger=logger)
    trainer.fit(model=lightning_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":

    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit_train_samples_ratio", type=float, default=1.0)
    parser.add_argument("--limit_val_samples_ratio", type=float, default=1.0)
    parser.add_argument("--limit_test_samples_ratio", type=float, default=1.0)
    parser.add_argument("-e", "--max_epochs", type=int, default=10)
    parser.add_argument("--config_file", type=str, default="configs/localized_qformer.yaml")
    args = parser.parse_args()

    main(args)
