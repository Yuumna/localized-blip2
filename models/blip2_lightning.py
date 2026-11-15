import os
import json
import random
from typing import Literal, Optional
from datetime import datetime
import tempfile

import torch
import pickle
import lightning.pytorch as pl
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

from model_utils import LookupTables, pretty_json
from training_utils import LinearWarmupConstantLRScheduler, CosineAnnealingLRByStep, make_training_dir


class Blip2Model(pl.LightningModule):
    COCO_EVAL_PERF_METRICS = "coco_eval_perf_metrics.txt"

    def __init__(
            self,
            model,
            train_cfg: Optional[str] = None,
            ca_num_steps: Optional[int] = None,
            val_gt_file: Optional[str] = None, 
            val_dataset_format: Optional[Literal["COCO", "RefCOCO", "VG"]] = None,
            output_dir="./output",
            training_name="train_blip2",
            test_gt_file: Optional[str] = None, 
            test_dataset_format: Optional[Literal["COCO", "RefCOCO"]] = None,

            # debug purpose
            is_debug=False,

            # hack to run test during the training
            run_test_every_n_val_iters=-1,
        ):
        super().__init__()
        self.model = model
        self.train_cfg = train_cfg
        self.ca_num_steps = ca_num_steps

        # Evaluation related attributes
        self.val_gt_file = val_gt_file
        self.val_dataset_format = val_dataset_format
        self.output_dir = output_dir
        # self.eval_perf_metrics_file = os.path.join(self.output_dir,
        #                                            self.COCO_EVAL_PERF_METRICS)
        self.generated_captions_dir = make_training_dir(os.path.join(self.output_dir, "generated_captions"), training_name)
        self.gt_captions_dir = make_training_dir(os.path.join(self.output_dir, "gt_captions"), training_name)
        self.is_debug = is_debug

        # hack to run test during the training and validation
        self.val_iter_within_epoch = 0
        self.run_test_every_n_val_iters = run_test_every_n_val_iters
        self.test_gt_file = test_gt_file
        self.test_dataset_format = test_dataset_format
        if self.run_test_every_n_val_iters > 0:
            assert self.test_gt_file is not None and self.test_dataset_format is not None
        
        os.makedirs(self.output_dir, exist_ok=True)

        self.val_capt_pred = []
        self.val_refcocog_pred = []
        self.test_step_outputs = []

    def configure_optimizers(self):
        num_parameters = 0
        p_wd, p_non_wd = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue  # frozen weights
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)
            num_parameters += p.data.nelement()
        print("number of trainable parameters: %d" % num_parameters)
        optim_params = [
            {
                "params": p_wd,
                "weight_decay": self.train_cfg.weight_decay,
            },
            {"params": p_non_wd, "weight_decay": 0},
        ]
        optimizer = torch.optim.AdamW(
            optim_params,
            lr=self.train_cfg.init_lr,
            weight_decay=self.train_cfg.weight_decay,
            betas=(0.9, self.train_cfg.get("beta2", 0.999)),
        )

        if self.train_cfg.scheduler == "linear_warmup_constant":
            scheduler_internal = LinearWarmupConstantLRScheduler(
                    optimizer=optimizer,
                    warmup_steps=self.train_cfg.warmup_steps,
                    warmup_lr=self.train_cfg.warmup_lr,
                    init_lr=self.train_cfg.init_lr,
                )
        elif self.train_cfg.scheduler == "cosine_annealing":
            scheduler_internal = CosineAnnealingLRByStep(
                    optimizer=optimizer,
                    num_step=self.ca_num_steps,
                    eta_min=self.train_cfg.eta_min,
                    override_lr=self.train_cfg.get("override_lr", None),
            )

        lr_scheduler = {
            "scheduler": scheduler_internal,
            "interval": "step",
            "frequency": 1,
        }
        
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        loss_dict = self.model(batch)

        self.log_dict(loss_dict, on_step=True)
        
        return loss_dict["loss"]
        
    def on_train_epoch_start(self):
        self.val_iter_within_epoch = 0

    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        if dataloader_idx == 0:
            # Compute validation loss
            loss_dict = self.model(batch)

            loss_dict = {f"val_{k}": v for k, v in loss_dict.items()}
            self.log_dict(loss_dict, on_step=False, on_epoch=True, add_dataloader_idx=False)

            # Generate captions for evaluation
            self.val_capt_pred.extend(self._predict(batch, self.val_dataset_format))

        if dataloader_idx == 1 \
            and self.run_test_every_n_val_iters > 0 \
            and (self.val_iter_within_epoch + 1) % self.run_test_every_n_val_iters == 0:

            # Run evaluation on test set
            self.val_refcocog_pred.extend(self._predict(batch, self.test_dataset_format))

    def on_validation_epoch_end(self):
        # debug
        # Compute validation metrics (e.g. BLEU, CIDEr, etc.)
        print("Calculating VG metrics...")
        metrics = self._compute_perf_metrics(
            self.val_capt_pred, self.val_gt_file, self.val_dataset_format,
            "val", self.current_epoch, self.global_step
        )
        metrics = {f"val_{k}": v for k, v in metrics.items()}
        self.log_dict(metrics, on_step=False, on_epoch=True)
        
        if self.val_refcocog_pred:
            print("Calculating RefCOCOg metrics...")
            refcocog_metrics = self._compute_perf_metrics(
                self.val_refcocog_pred, self.test_gt_file, self.test_dataset_format,
                "val_refcocog", self.current_epoch, self.global_step
            )
            # refcocog_metrics = {f"val_refcocog_{k}": v for k, v in refcocog_metrics.items()}
            refcocog_metrics = {f"test_{k}": v for k, v in refcocog_metrics.items()}
            self.log_dict(refcocog_metrics, on_step=False, on_epoch=True)

        self.val_capt_pred.clear()
        self.val_refcocog_pred.clear()
        self.val_iter_within_epoch += 1

    def test_step(self, batch, batch_idx):
        self.test_step_outputs.extend(self._predict(batch, self.test_dataset_format))

    def on_test_epoch_end(self) -> None:
        # Compute test metrics (e.g. BLEU, CIDEr, etc.)
        print("Calculating test metrics...")
        metrics = self._compute_perf_metrics(
            self.test_step_outputs, self.test_gt_file, self.test_dataset_format,
            "test", self.current_epoch, self.global_step
        )
        metrics = {f"test_{k}": v for k, v in metrics.items()}
        self.log_dict(metrics, on_step=False, on_epoch=True)

        self.test_step_outputs.clear()

    def _predict(self, batch, dataset_format):
        outputs = []
        captions = self.model.generate(batch)

        if dataset_format == "COCO":
            for caption, img_id in zip(captions, batch["image_id"]):
                outputs.append({"caption": caption, "image_id": int(img_id)})

        elif dataset_format in ["RefCOCO", "VG"]:
            for caption, img_id, ann_id in zip(captions, batch["image_id"], batch["ann_id"]):
                outputs.append({"caption": caption, "image_id": int(img_id), "ann_id": int(ann_id)})

        else:
            raise ValueError(f"Unknown eval_dataset_name: {dataset_format}")
        
        return outputs

    def _compute_perf_metrics(self, val_result, gt_file, dataset_format, name, epoch, step):
        

        # load ground truth
        if dataset_format == "COCO":
            gt = json.load(open(gt_file, "r"))
            processed_val_result = val_result

        elif dataset_format == "RefCOCO":
            gt = pickle.load(open(gt_file, "rb"))

            # modify gt_file and res_file to use the value of "ann_id" for "image_id"
            # instead of the "image_id" itself
            ann_id_to_captions = {}
            for instance in gt:
                ann_id_to_captions[instance["ann_id"]] = [sent["raw"] for sent in instance["sentences"]]

            modified_gt = { "images": [], "annotations": [] }
            for ann_id in ann_id_to_captions.keys():
                modified_gt["images"].append({"id": ann_id})
                modified_gt["annotations"].extend(
                    [{"image_id": ann_id, "caption": cap, "id": ann_id} for cap in ann_id_to_captions[ann_id]]
                )

            gt = modified_gt
            
            processed_val_result = [
                {
                    "image_id": instance["ann_id"],
                    "caption": instance["caption"],
                    "original_image_id": instance["image_id"]
                }
            for instance in val_result]

        elif dataset_format == "VG":
            gt = json.load(open(gt_file, "r"))

            # modify gt_file and res_file to use the value of "ann_id" for "image_id"
            # instead of the "image_id" itself
            modified_gt = { "images": [], "annotations": [] }
            for ann in gt["annotations"]:
                modified_gt["images"].append({"id": ann["id"]})
                modified_gt["annotations"].append(
                    {"image_id": ann["id"], "caption": ann["caption"], "id": ann["id"]}
                )
            gt = modified_gt

            processed_val_result = [
                {
                    "image_id": instance["ann_id"],
                    "caption": instance["caption"],
                    "original_image_id": instance["image_id"]
                }
            for instance in val_result]

        else:
            raise ValueError(f"Unknown dataset_format: {dataset_format}")
        
        # store val_result into a JSON file
        eval_result_file = save_result_to_json(
            result=processed_val_result,
            result_dir=self.generated_captions_dir,
            filename=f"{name}_epoch_{epoch}_step_{step}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Filter ground truth according to the image_ids in val_result
        filtered_gt = filter_gt(gt, [x["image_id"] for x in processed_val_result])

        # Randomly log some generated captions with their corresponding ground truth and image
        if self.is_debug:
            filtered_gt_ann_lookup = LookupTables(filtered_gt["annotations"])
            val_result_ann_lookup = LookupTables(val_result)

            gt_res_pair = []
            #selected_res = random.choices(processed_val_result, k=min(20, len(processed_val_result)))
            for res in processed_val_result:
                if dataset_format == "COCO":
                    gt_res_pair.append({
                        "gen_caption": res["caption"],
                        "gt_captions": filtered_gt_ann_lookup.get("image_id", "caption", res["image_id"]),
                        "image_id": res["image_id"]
                    })
                elif dataset_format in ["RefCOCO", "VG"]:
                    gt_res_pair.append({
                        "gen_caption": res["caption"],
                        "gt_captions": filtered_gt_ann_lookup.get("image_id", "caption", res["image_id"]),
                        "image_id": val_result_ann_lookup.get("ann_id", "image_id", res["image_id"])[0],
                        "ann_id": res["image_id"]
                    })
                else:
                    raise ValueError(f"Unknown dataset_format: {dataset_format}")
                
            
            print(pretty_json(gt_res_pair))
            self.logger.experiment.add_text(
                tag=f"{name}_epoch_{epoch}_gen_gt_captions",
                text_string=pretty_json(gt_res_pair),
                global_step=epoch
            )

        # store filtered_gt into a JSON file
        filtered_gt_file = save_result_to_json(
            result=filtered_gt,
            result_dir=self.gt_captions_dir,
            filename=f"filtered_{name}_epoch_{epoch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # compute coco metrics
        coco_eval = eval_captioning(filtered_gt_file, eval_result_file)

        agg_metrics = coco_eval.eval["CIDEr"] + coco_eval.eval["Bleu_4"]
        # log_stats = {name: {k: v for k, v in coco_eval.eval.items()}}

        # # save coco metrics to file
        # with open(self.eval_perf_metrics_file, "a") as f:
        #     f.write(json.dumps(log_stats) + "\n")

        # save coco metrics to tensorboard
        coco_res = {k: v for k, v in coco_eval.eval.items()}
        coco_res["agg_metrics"] = agg_metrics

        return coco_res

    
def filter_gt(coco_gt, img_ids):
    img_id_to_image = {}
    for img in coco_gt["images"]:
        img_id_to_image[img["id"]] = img

    img_id_to_anns = {}
    for ann in coco_gt["annotations"]:
        if img_id_to_anns.get(ann['image_id']) is None:
            img_id_to_anns[ann['image_id']] = [ann]
        else:
            img_id_to_anns[ann['image_id']].append(ann)

    assert(set(img_id_to_image.keys()) == set(img_id_to_anns.keys()))

    # filter images and annotations
    coco_filter_gt = {"images": [], "annotations": []}
    seen_images = set()
    for img_id in img_ids:
        if img_id in seen_images:
            continue

        coco_filter_gt["images"].append(img_id_to_image[img_id])
        seen_images.add(img_id)

        coco_filter_gt["annotations"].extend(img_id_to_anns[img_id])

    print("raw coco_gt size: ", len(coco_gt["images"]))
    print("res size: ", len(img_ids))
    print("filtered coco_gt['images'] size: ", len(coco_filter_gt["images"]))
    print("filtered coco_gt['annotations'] size: ", len(coco_filter_gt["annotations"]))
          
    return coco_filter_gt


def save_result_to_json(result, result_dir, filename):
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f"{filename}.json")
    json.dump(result, open(result_file, "w"), indent=4)

    return result_file


def eval_captioning(gt_file, res_file):
    """
    To evaluate the captioning performance
    (modified from lavis.tasks.captioning.coco_caption_eval(...)

    :param gt_file (str): json file that contains the following:  
        {
            "images": [{"id": int}, ...],
            "annotations": [{"image_id": int, "caption": str, "id": int}, ...]
        }

    :param res_file (str): json file that contains the following:
        [{"image_id": int, "caption": str}, ...]
    """

    # evaluate the metrics
    coco = COCO(gt_file)
    coco_res = coco.loadRes(res_file)
    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval
