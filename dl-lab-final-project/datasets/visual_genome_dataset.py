import os
from pathlib import Path
import json
from typing import Callable, Optional, Literal, Dict, List
from tqdm import tqdm
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
import torch
from collections import Counter
from model_utils import get_normalized_bounding_box
from general_utils import load_jsonl


class VGDataset(Dataset):
    """Visual Genome Dataset"""

    def __getitem__(self, index):
        # This should not be called, use get_split instead
        raise NotImplementedError

    def __init__(
            self,
            image_dir: str,
            index_file: str,
            text_dir: Optional[str] = None,
            vis_processor: Optional[Dict[str, Callable]] = None,
            text_processor: Optional[Dict[str, Callable]] = None,
            feed_image_mode: Literal["full", "cropped", "blackout"] = "full",
    ) -> None:
        super().__init__()
        self.text_dir = Path(text_dir) if text_dir else None
        self.image_dir = Path(image_dir) if image_dir else None
        self.index_file = index_file
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.feed_image_mode = feed_image_mode
        self.data = None

        if not os.path.exists(self.index_file):
            if self.text_dir is None:
                raise ValueError(
                    "text_dir must be provided if index_file does not exist")

            print(f"{self.index_file} not found. Generating index file")
            all_data = self._generate_index_file()     # This takes very long

        else:
            print("Loading from index file")
            all_data = json.load(open(self.index_file, "r"))

        self.data = self._split_data(all_data)

        self.gt_output_files = {
            "train": os.path.join(os.path.dirname(self.index_file), "vg_ground_truth_train.json"),
            "val": os.path.join(os.path.dirname(self.index_file), "vg_ground_truth_val.json"),
        }

        for split in ["train", "val"]:
            if not os.path.exists(self.gt_output_files[split]):
                print(
                    f"{self.gt_output_files[split]} not found. Generating ground truth file")
                self._generate_ground_truth_file(split)

    def get_ground_truth_file(self, split: Literal["train", "val", "test"]):
        if not self._check_valid_split(split):
            raise ValueError(f"Invalid split type: {self.split}")

        return self.gt_output_files[split]

    def get_split(self, split: Literal["train", "val", "test"]):
        # check if the split type is valid
        if not self._check_valid_split(split):
            raise ValueError(f"Invalid split type: {self.split}")

        return VGSubset(
            data=self.data[split],
            image_dir=self.image_dir,
            text_dir=self.text_dir,
            vis_processor=self.vis_processor["train" if split ==
                                             "train" else "eval"],
            text_processor=self.text_processor["train" if split ==
                                               "train" else "eval"],
            feed_image_mode=self.feed_image_mode
        )

    def _split_data(self, all_data):
        # All data Splits:
        #   Train: 0:2637327
        #   Validation: 2637327:2646927
        #   Test: 2646927:

        # Big Image Split:
        #   Train: 0:717154
        #   Validation: 717154: 806791
        #   Test: 806791:

        # Only Train and Val for Large Images
        #   Train: [:-10001]
        #   Validation: [-10001:]

        # split_data = {
        #     "train": all_data[:-10001],
        #     "val": all_data[-10001:],
        # }
        split_data = {
            "train": all_data[:-10023],
            "val": all_data[-10023:],
            # retrieval eval used a halved val split to fit in memory:
            # "val": all_data[-5001:],
        }
        return split_data

    def _generate_index_file(self):
        """
        Generate index file from the entries in the directory,
            save the index file and return the data (list of dicts)
        """

        index_data = []
        for _, file in enumerate(tqdm(list(self.text_dir.glob("*.data")))):
            data = load_jsonl(file)
            for _, item in enumerate(data):
                assert len(item) == 3, f"Unexpected keys: {item.keys()}"
                image_id = item["image_id"]
                assert image_id not in index_data
                # keys are image, image_id and elems
                for elem in item["elems"]:
                    assert len(elem) == 3, f"Unexpected keys: {elem.keys()}"
                    sample_data = {
                        "image_id": image_id,
                        "ann_id": elem["region_id"],
                        "text": elem["caption"]["en"],
                        "bb": elem["bb"]
                    }
                    index_data.append(sample_data)

        with open(self.index_file, "w") as f:
            json.dump(index_data, f, indent=4)

        return index_data

    def _check_valid_split(self, split: Literal["train", "val", "test"]) -> bool:
        return split in ['train', 'val', 'test']

    def _generate_ground_truth_file(self, split: Literal["train", "val", "test"]):
        """
        Generate ground truth file for evaluation according to the following format:

        {
            "images": [{"id": int}, ...],
            "annotations": [{"image_id": int, "caption": str, "id": int}, ...]
        }
        """

        gt = {"images": [], "annotations": []}
        for sample in self.data[split]:
            gt["images"].append({"id": sample["image_id"]})
            gt["annotations"].append({
                "image_id": sample["image_id"],
                "caption": sample["text"],
                "id": sample["ann_id"]
            })

        with open(self.gt_output_files[split], "w") as f:
            json.dump(gt, f)

    # def _debug(self):
        # debugging
        # print("Train Length:", len(all_data[0:2357545]))
        # print("Validation Length:", len(all_data[2357545:2387545]))
        # print("Test Length:", len(all_data[2387545:]))
        # train_ids = [item["image_id"] for item in all_data[0:2357545]]
        # val_ids = [item["image_id"] for item in all_data[2357545:2387545]]
        # test_ids = [item["image_id"] for item in all_data[2387545:]]
        # tr_val_intersect = Counter(train_ids) & Counter(val_ids)
        # val_test_intersect = Counter(val_ids) & Counter(test_ids)
        # test_tr_intersect = Counter(test_ids) & Counter(train_ids)
        # print("Train Validation Intersection:", tr_val_intersect)
        # print("Validation Test Intersection:", val_test_intersect)
        # print("Train Test Intersection:", test_tr_intersect)

    @classmethod
    def collate_fn(cls, batch):
        image_ids, ann_ids, images, text_input, bboxes = [], [], [], [], []

        for b in batch:
            image_ids.append(b["image_id"])
            ann_ids.append(b["ann_id"])
            images.append(b["image"])
            text_input.append(b["text_input"])
            bboxes.append(torch.Tensor(b["bbox_normalized"]))

        return {
            "image_id": image_ids,
            "ann_id": ann_ids,
            "image": torch.stack(images),
            "text_input": text_input,
            "bbox_normalized": torch.stack(bboxes),
        }


class VGSubset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        image_dir: Path,
        text_dir: Path,
        vis_processor: Optional[Callable],
        text_processor: Optional[Callable],
        feed_image_mode: Literal["full", "cropped", "blackout"] = "full",
    ):
        self.data = data
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.vis_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            vis_processor.normalize,
        ])
        self.text_transform = text_processor
        self.feed_image_mode = feed_image_mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_item = self.data[index]
        image_id = data_item["image_id"]
        ann_id = data_item["ann_id"]
        text = data_item["text"]
        bb = data_item["bb"]

        image_file = self.image_dir / f"{image_id}.jpg"
        image = Image.open(image_file).convert("RGB")
        image_width, image_height = image.size
        bbox_normalized = get_normalized_bounding_box(
            bb, (image_width, image_height))
        # Convert bbox from normalized to pixel coordinates
        bbox_x_pixel = int(bbox_normalized[0] * image_width)
        bbox_y_pixel = int(bbox_normalized[1] * image_height)
        bbox_w_pixel = int(bbox_normalized[2] * image_width)
        bbox_h_pixel = int(bbox_normalized[3] * image_height)
        if self.feed_image_mode == "full":
            pass

        elif self.feed_image_mode == "cropped":
            # Crop the image based on the bounding box
            image = image.crop((
                bbox_x_pixel, bbox_y_pixel,
                bbox_x_pixel + bbox_w_pixel, bbox_y_pixel + bbox_h_pixel
            ))

        elif self.feed_image_mode == "blackout":
            # Blackout the area outside the bounding box
            black_image = Image.new("RGB", image.size, "black")
            pil_bbox = image.crop((
                bbox_x_pixel, bbox_y_pixel,
                bbox_x_pixel + bbox_w_pixel, bbox_y_pixel + bbox_h_pixel
            ))
            black_image.paste(pil_bbox, (bbox_x_pixel, bbox_y_pixel))
            image = black_image

        image_ten = self.vis_transform(image)

        if self.text_transform:
            text = self.text_transform(text)

        sample = {
            "image_id": image_id,
            "ann_id": ann_id,
            "image": image_ten,
            "text_input": text,
            "bbox_normalized": get_normalized_bounding_box(bb, (image_width, image_height))
        }
        return sample
