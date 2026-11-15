import os
import json
import pickle
import random
from typing import Optional, Callable, Literal

import torch
from torch.utils.data import Dataset
from PIL import Image

from model_utils import resize_bbox


class RefCocoDataset(Dataset):
    REFCOCO_PLUS_SPLIT = ["train", "val", "testA", "testB"]
    REFCOCO_G_SPLIT = ["train", "val", "test"]

    def __init__(
            self,
            vis_root: str,
            instances_file: str,
            refs_file: str,
            dataset_name: Literal["refcoco+", "refcoco_g"],
            set_type: Literal["train", "val", "testA", "testB", "test"],
            vis_transform: Optional[Callable] = None,
            text_transform: Optional[Callable] = None,
            feed_image_mode: Literal["full", "cropped", "blackout"] = "full",
            device=torch.device("cuda")
    ):
        if dataset_name == "refcoco+":
            assert set_type in self.REFCOCO_PLUS_SPLIT
        elif dataset_name == "refcoco_g":
            assert set_type in self.REFCOCO_G_SPLIT
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        self.vis_root = vis_root
        self.vis_transform = vis_transform
        self.text_transform = text_transform
        self.feed_image_mode = feed_image_mode
        self.device = device

        self.ann_id_to_ann = self._load_annotations(instances_file)
        self.refs = pickle.load(open(refs_file, "rb"))
        self.samples = self._get_data_based_on_set_type(set_type)

        self.ground_truth_output_file = os.path.join(
            os.path.dirname(instances_file),
            f"{dataset_name}_ground_truth_{set_type}.json",
        )

        # (re-)generate ground truth file
        # especially in the case where the refs or annotations are changed,
        if os.path.exists(self.ground_truth_output_file):
            os.remove(self.ground_truth_output_file)
        self._generate_ground_truth_file()

    @property
    def ground_truth_file(self):
        return self.ground_truth_output_file

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        item = self.samples[index]

        pil_image = Image.open(item["image_path"]).convert("RGB")
        image_width, image_height = pil_image.size
        bbox_normalized = self.ann_id_to_ann[item["ann_id"]]["bbox_normalized"]

        # Convert bbox from normalized to pixel coordinates
        bbox_x_pixel = int(bbox_normalized[0] * image_width)
        bbox_y_pixel = int(bbox_normalized[1] * image_height)
        bbox_w_pixel = int(bbox_normalized[2] * image_width)
        bbox_h_pixel = int(bbox_normalized[3] * image_height)

        if self.feed_image_mode == "full":
            pass

        elif self.feed_image_mode == "cropped":
            # Crop the image based on the bounding box
            pil_image = pil_image.crop((
                bbox_x_pixel, bbox_y_pixel,
                bbox_x_pixel + bbox_w_pixel, bbox_y_pixel + bbox_h_pixel
            ))

        elif self.feed_image_mode == "blackout":
            # Blackout the area outside the bounding box
            black_image = Image.new("RGB", pil_image.size, "black")
            pil_bbox = pil_image.crop((
                bbox_x_pixel, bbox_y_pixel,
                bbox_x_pixel + bbox_w_pixel, bbox_y_pixel + bbox_h_pixel
            ))
            black_image.paste(pil_bbox, (bbox_x_pixel, bbox_y_pixel))
            pil_image = black_image

        image = pil_image
        if self.vis_transform is not None:
            image = self.vis_transform(pil_image)

        caption = item["caption"]
        if self.text_transform is not None:
            caption = self.text_transform(caption)

        return {
            "image_id": item["image_id"],
            "ann_id": item["ann_id"],
            "image": image,
            "text_input": caption,
            "bbox_normalized": bbox_normalized,     # (x, y, w, h)
        }
    
    def _get_data_based_on_set_type(self, set_type):
        samples = []
        ref_in_split = []
        missing_ref_id = []
        for ref in self.refs:
            if ref["split"] == set_type:
                ref_in_split.append(ref)
                image_id = ref["image_id"]

                # get the real image filename, since in refcoco+ the filename
                # doesn't come with the corresponding dir. and is in the
                # following format:
                #   - "COCO_train2014_000000000025.jpg"
                #   - "COCO_train2014_000000000025_1.jpg"
                image_filename_split = ref["file_name"].split("_")
                image_dir = image_filename_split[1]
                image_path = "_".join(image_filename_split[:3]) + ".jpg"
                image_path = os.path.join(self.vis_root, image_dir, image_path)

                if not os.path.exists(image_path):
                    missing_ref_id.append((image_id, ref["ref_id"]))
                    continue

                caption = random.choice(ref["sentences"])['raw']

                samples.append({
                    "image_id": image_id,
                    "image_path": image_path,
                    "caption": caption,
                    "ann_id": ref["ann_id"],
                })

        print(f"Missing {len(missing_ref_id)}/{len(ref_in_split)} references for {set_type} set.")
        return samples

    def _load_annotations(self, instances_file):
        with open(instances_file, "r") as file:
            instances = json.load(file)
            annotations = instances["annotations"]
            images = instances["images"]

            images_dict = {}
            for image in images:
                images_dict[image["id"]] = image

            ann_id_to_ann = {}
            for annotation in annotations:
                image_id = annotation["image_id"]
                annotation["bbox_normalized"] = resize_bbox(
                    annotation["bbox"],
                    images_dict[image_id]["width"],
                    images_dict[image_id]["height"],
                    1,
                    1,
                )
                ann_id_to_ann[annotation["id"]] = annotation

        return ann_id_to_ann

    def _generate_ground_truth_file(self):
        """
        Generate ground truth file for evaluation according to the following format:

        {
            "images": [{"id": int}, ...],
            "annotations": [{"image_id": int, "caption": str, "id": int}, ...]
        }
        """
        gt = {"images": [], "annotations": []}
        for s in self.samples:
            gt["images"].append({"id": s["image_id"]})
            gt["annotations"].append({
                "image_id": s["image_id"],
                "caption": s["caption"],
                "id": s["ann_id"],
            })

        with open(self.ground_truth_output_file, "w") as file:
            json.dump(gt, file)
    
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
