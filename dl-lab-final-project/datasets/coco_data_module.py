import os
from typing import Callable

import lightning as L
from torch.utils.data import DataLoader

from datasets.default_data_module_config import DefaultDataModuleConfig
from datasets.coco_dataset import CocoCaptionDataset, CocoCaptionEvalDataset
    

class CocoCaptionDataModule(L.LightningDataModule):
    def __init__(
        self,
        image_root_dir: str,
        ann_files_dir: str,
        vis_transform: Callable,
        text_transform: Callable,
        batch_size: int = int(os.getenv("BATCH_SIZE", DefaultDataModuleConfig.BATCH_SIZE)),
        num_workers: int = int(DefaultDataModuleConfig.NUM_WORKERS),
    ):
        super().__init__()
        self.image_root_dir = image_root_dir
        self.ann_files_dir = ann_files_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.vis_transform = vis_transform
        self.text_transform = text_transform

    def prepare_data(self):
        # TODO download dataset if not existing (for the moment, dataset is downloaded manually)
        pass

        # # download dataset if needed
        # if not self.skip_download_dataset:
        #     foz.download_zoo_dataset(
        #         "coco-2014",
        #         split="validation",
        #         dataset_dir=self.data_dir
        #     )

    def setup(self, stage=None):
        if stage == "fit":
            raise NotImplementedError("CocoCaptionDataModule does not support training")

        if stage == "validate":
            print("Setting up for validation")
            self.val_set = CocoCaptionEvalDataset(
                vis_root=os.path.join(self.image_root_dir),
                ann_file=os.path.join(self.ann_files_dir, "coco_karpathy_val.json"),
                vis_transform=self.vis_transform,
                text_transform=self.text_transform
            )
            print(f"-- val_set size = {len(self.val_set)}")

        if stage == "test":
            print("Setting up for test")
            self.test_set = CocoCaptionEvalDataset(
                vis_root=os.path.join(self.image_root_dir),
                ann_file=os.path.join(self.ann_files_dir, "coco_karpathy_test.json"),
                vis_transform=self.vis_transform,
                text_transform=self.text_transform
            )
            print(f"-- test_set size = {len(self.test_set)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        print("num_workers = ", self.num_workers)

        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
