import os
import json
from typing import Optional, Callable

from torch.utils.data import Dataset
from PIL import Image


class CocoBaseDataset(Dataset):
    def __init__(
            self,
            vis_root: str,
            ann_file: str,
            vis_transform: Optional[Callable] = None,
            text_transform: Optional[Callable] = None
    ):
        self.vis_root = vis_root
        print(f"vis_root: {self.vis_root}")
        self.vis_transform = vis_transform
        self.text_transform = text_transform

        self.annotation = json.load(open(ann_file, "r"))

    def __len__(self):
        return len(self.annotation)


class CocoCaptionDataset(CocoBaseDataset):
    def __init__(
        self,
        vis_root: str,
        ann_file: str,
        vis_transform: Optional[Callable] = None,
        text_transform: Optional[Callable] = None
    ):
        super().__init__(vis_root, ann_file, vis_transform, text_transform)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_transform(image)
        caption = self.text_transform(ann["caption"])

        return {
            "image": image,
            "text_input": caption,
            "image_id": ann["image_id"]
        }


class CocoCaptionEvalDataset(CocoBaseDataset):
    def __init__(
        self,
        vis_root: str,
        ann_file: str,
        vis_transform: Optional[Callable] = None,
        text_transform: Optional[Callable] = None
    ):
        super().__init__(vis_root, ann_file, vis_transform, text_transform)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_transform(image)

        image_id = ann["image"].split("/")[-1].strip(".jpg").split("_")[-1]
        
        # strip leading zeros
        image_id = str(int(image_id))

        return {
            "image": image,
            "image_id": image_id
        }
