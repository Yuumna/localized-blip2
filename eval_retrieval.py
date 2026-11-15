import os
import pprint
import random

from omegaconf import OmegaConf
import numpy as np
import torch
from torch.utils.data import DataLoader
from lavis.models import load_preprocess

from models.localized_blip2_qformer import LocalizedBlip2QFormer
from datasets.refcoco_dataset import RefCocoDataset
from datasets.coco_data_module import CocoCaptionDataModule
from retrieval_utils import preprocess_data, compute_retrieval_metrics
from general_utils import get_env_variables

pp = pprint.PrettyPrinter(indent=4)
ENV = get_env_variables()
STORAGE_DIR = ENV["STORAGE_DIR"]
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(args=None):
    print("Task: Evaluate retrieval on COCO")
    set_seed(42)

    # print args
    if args:
        print("args:")
        pp.pprint(args.__dict__)

    # load model and preprocesses
    model_name = "blip2"
    model_type = "pretrain"
    model = LocalizedBlip2QFormer.from_pretrained(model_type=model_type).to("cuda")
    model.is_localized_attention = False
    cfg = OmegaConf.load(LocalizedBlip2QFormer.default_config_path(model_type)).preprocess
    vis_preprocess, text_preprocess = load_preprocess(cfg)
    model.to(device)

    # refcoco = RefCocoDataset(
    #     vis_root=os.path.join(STORAGE_DIR, "shared/datasets/coco2014/coco/images/"),
    #     instances_file=os.path.join(STORAGE_DIR, "shared/datasets/refcocog/instances.json"),
    #     refs_file=os.path.join(STORAGE_DIR, "shared/datasets/refcocog/refs(umd).p"),
    #     dataset_name="refcoco_g",
    #     set_type="val",
    #     vis_transform=vis_preprocess["eval"],
    #     text_transform=text_preprocess["eval"],
    #     feed_image_mode="full"
    # )
    # refcoco_dataloader = DataLoader(refcoco, batch_size=32, num_workers=4, collate_fn=RefCocoDataset.collate_fn)

    coco_data_module = CocoCaptionDataModule(
        image_root_dir=os.path.join(STORAGE_DIR, "shared/datasets/coco2014/coco/images"),
        ann_files_dir=os.path.join(STORAGE_DIR, "shared/datasets/coco2014/coco/annotations"),
        vis_transform=vis_preprocess["eval"].transform,
        text_transform=text_preprocess["eval"],
        batch_size=100,
        num_workers=4,
    )
    coco_data_module.setup("test")
    coco_test_loader = coco_data_module.test_dataloader()

    all_images, all_texts, i2t, t2i = preprocess_data(coco_test_loader, group_by="image_id")
    with torch.cuda.amp.autocast():
        score_matrix_i2t, score_matrix_t21 = model.compute_sim_matrix(all_images, all_texts, topk=128)
        retrieval_metrics = compute_retrieval_metrics(score_matrix_i2t, score_matrix_t21, t2i, i2t)
    
    print("Retrieval metrics:")
    pp.pprint(retrieval_metrics)


if __name__ == "__main__":
    main()
    