import os
import argparse
import pprint
import random

import torch
import numpy as np
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lavis.models import load_preprocess
from omegaconf import OmegaConf

from models.localized_blip2_qformer import LocalizedBlip2QFormer
from models.blip2_lightning import Blip2Model
from datasets.refcoco_dataset import RefCocoDataset
from general_utils import get_env_variables

pp = pprint.PrettyPrinter(indent=4)
ENV = get_env_variables()
STORAGE_DIR = ENV["STORAGE_DIR"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(args):
    print("Task: Evaluate BLIP2-Qformer on RefCoco+")
    set_seed(42)

    # print args
    print("main args ---------------------")
    pp.pprint(args.__dict__)

    # setup tensorboard 
    logger = pl.loggers.TensorBoardLogger("lightning_logs/", name="evaluate_qformer_on_refcoco_plus")

    # load model and preprocesses
    model_name = "blip2"
    model_type = "pretrain"     # pretrain / pretrain_vitL / coco
    model = LocalizedBlip2QFormer.from_pretrained(model_type=model_type)
    cfg = OmegaConf.load(LocalizedBlip2QFormer.default_config_path(model_type)).preprocess
    vis_preprocess, text_preprocess = load_preprocess(cfg)

    # define lightning modules
    refcoco = RefCocoDataset(
        vis_root=os.path.join(STORAGE_DIR, "shared/datasets/coco2014/coco/images"),
        instances_file=os.path.join(STORAGE_DIR, "shared/datasets/refcoco+/instances.json"),
        refs_file=os.path.join(STORAGE_DIR, "shared/datasets/refcoco+/refs(unc).p"),
        dataset_name="refcoco+",
        set_type="val",
        vis_transform=vis_preprocess["eval"],
        text_transform=text_preprocess["eval"],
    )

    dataloader = DataLoader(refcoco, batch_size=64, num_workers=10, collate_fn=RefCocoDataset.collate_fn)

    lightning_model = Blip2Model(
        model,
        test_gt_file=os.path.join(STORAGE_DIR, "shared/datasets/refcoco+/refs(unc).p"),
        test_dataset_format="RefCOCO"
    )

    # evaluate model
    trainer = pl.Trainer(
        accelerator="auto",
        logger=logger,
        fast_dev_run=args.fast_dev_run,
        limit_test_batches=args.limit_samples_ratio
    )
    trainer.test(model=lightning_model, dataloaders=[dataloader])


if __name__ == "__main__":

    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fast_dev_run", action="store_true")
    parser.add_argument("--limit_samples_ratio", type=float, default=1.0)
    args = parser.parse_args()

    main(args)
