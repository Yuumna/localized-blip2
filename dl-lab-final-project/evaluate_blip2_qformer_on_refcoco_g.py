import argparse
import pprint
import random
import os 
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lavis.models import load_preprocess
from lavis.models.blip2_models.blip2_qformer import Blip2Qformer
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
    print("Task: Evaluate BLIP2-Qformer on RefCocog")

    # print args
    print("main args ---------------------")
    pp.pprint(args.__dict__)
    
    # setup and print train config
    train_cfg = OmegaConf.load(args.training_config_file).train
    print("train config:")
    pp.pprint(train_cfg)

    set_seed(train_cfg.seed)
    # setup tensorboard 
    logger = pl.loggers.TensorBoardLogger("output/lightning_logs/", name="Evaluate_blip2_stage1_qformer_refcocog")

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
    refcoco = RefCocoDataset(
        vis_root=os.path.join(STORAGE_DIR, "shared/datasets/coco2014/coco/images/"),
        instances_file=os.path.join(STORAGE_DIR, "shared/datasets/refcocog/instances.json"),
        refs_file=os.path.join(STORAGE_DIR, "shared/datasets/refcocog/refs(umd).p"),
        dataset_name="refcoco_g",
        set_type="val",
        vis_transform=vis_preprocess["eval"],
        text_transform=text_preprocess["eval"],
        feed_image_mode=args.feed_image_mode
    )

    dataloader = DataLoader(refcoco, batch_size=32, num_workers=4, collate_fn=RefCocoDataset.collate_fn)

    lightning_model = Blip2Model(
        model,
        test_gt_file=os.path.join(STORAGE_DIR, "shared/datasets/refcocog/refs(umd).p"),
        test_dataset_format="RefCOCO",
        is_debug=True
    )

    # evaluate model
    trainer = pl.Trainer(
        accelerator="auto",
        logger=logger,
        precision=train_cfg.precision,
        limit_test_batches=args.limit_samples_ratio,
        limit_val_batches=args.val_batches_num
    )
    trainer.test(model=lightning_model, dataloaders=[dataloader], ckpt_path=args.ckpt_path)


if __name__ == "__main__":

    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit_samples_ratio", type=float, default=1.0)
    parser.add_argument("--use_localized_qformer", action="store_true")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--val_batches_num", type=int, default=None)
    parser.add_argument("--test_batches_num", type=int, default=None)
    parser.add_argument("--model_config_file", type=str, default="configs/localized_qformer_model.yaml")
    parser.add_argument("--training_config_file", type=str, default="configs/localized_qformer_ft_vg.yaml")
    parser.add_argument('--feed_image_mode', type=str, default='full', choices=['full', 'cropped', 'blackout'],
                        help='Mode for image handling. Options are "full", "cropped", and "blackout".')
    args = parser.parse_args()

    main(args)
