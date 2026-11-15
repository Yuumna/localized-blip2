import json

from torch.utils.data import DataLoader
import numpy as np
import torch

from model_utils import LookupTables


def preprocess_coco_data(dataloader: DataLoader, ann_file: str):
    gt_anns = json.load(open(ann_file, "r"))["annotations"]
    print("gt_anns: ", gt_anns)
    gt_lookup_table = LookupTables(gt_anns)

    all_data = {}
    for batch in dataloader:
        for i, key in enumerate(batch["image_id"]):
            if key not in all_data.keys():
                all_data[key] = {"texts": [], "image": batch["image"][i]}

            # get captions of this image
            texts = gt_lookup_table.get("image_id", "caption", int(key))
            print("texts: ", texts)
            all_data[key]["texts"].extend(texts)

    all_texts, all_images = [], []
    for i, key in enumerate(all_data.keys()):
        all_texts.extend(all_data[key]["texts"])
        all_images.append(all_data[key]["image"])

    all_images = torch.stack(all_images)

    i2t_gt, t2i_gt = [], []
    text_counter = 0
    for i, key in enumerate(all_data.keys()):
        i2t_gt.append(list(range(text_counter, text_counter + len(all_data[key]["texts"]))))
        text_counter += len(all_data[key]["texts"])
        t2i_gt.extend([i] * len(all_data[key]["texts"]))

    return all_images, all_texts, i2t_gt, t2i_gt


def preprocess_cropped_data(dataloader: DataLoader, group_by: str):
    """ Reorder the data s.t. the data with the same group_by value are fed subsequently."""

    all_data = {}   # {group_by: {"text": [], "image": tensor}}
    print("preprocess data")
    for batch in dataloader:
        for i, key in enumerate(batch[group_by]):
            if key not in all_data.keys():
                all_data[key] = {"texts": [], "image": batch["image"][i]}
            all_data[key]["texts"].append(batch["text_input"][i])

    all_texts, all_images = [], []
    for i, key in enumerate(all_data.keys()):
        all_texts.extend(all_data[key]["texts"])
        all_images.append(all_data[key]["image"])
    all_images = torch.stack(all_images)

    # generate i2t and t2i ground truth
    i2t_gt = []
    t2i_gt = []
    text_counter = 0
    print("generate i2t and t2i ground truth")
    for i, key in enumerate(all_data.keys()):
        i2t_gt.append(list(range(text_counter, text_counter + len(all_data[key]["texts"]))))
        text_counter += len(all_data[key]["texts"])
        t2i_gt.extend([i] * len(all_data[key]["texts"]))

    return all_images, all_texts, i2t_gt, t2i_gt


def preprocess_localized_data(dataloader: DataLoader, group_by: str):
    """ Reorder the data s.t. the data with the same group_by value are fed subsequently."""

    all_data = {}   # {group_by: {"text": [], "image": tensor}}
    print("preprocess data")
    for batch in dataloader:
        for i, key in enumerate(batch[group_by]):
            if key not in all_data.keys():
                all_data[key] = {"texts": [], "image": batch["image"][i], "bbox": []}
            all_data[key]["texts"].append(batch["text_input"][i])
            all_data[key]["bbox"].append(batch["bbox_normalized"][i])

    all_texts, all_images, all_bbox, all_bbox_2_img = [], [], [], []
    for i, key in enumerate(all_data.keys()):
        all_texts.extend(all_data[key]["texts"])
        all_images.append(all_data[key]["image"])
        all_bbox.extend(all_data[key]["bbox"])
        all_bbox_2_img.extend([i] * len(all_data[key]["bbox"]))
    all_images = torch.stack(all_images)
    all_bbox = torch.stack(all_bbox)

    # generate i2t and t2i ground truth
    i2t_gt = []
    t2i_gt = []
    text_counter = 0
    print("generate i2t and t2i ground truth")
    for i, key in enumerate(all_data.keys()):
        i2t_gt.append(list(range(text_counter, text_counter + len(all_data[key]["texts"]))))
        text_counter += len(all_data[key]["texts"])
        t2i_gt.extend([i] * len(all_data[key]["texts"]))

    return all_images, all_texts, (all_bbox, all_bbox_2_img), i2t_gt, t2i_gt


def compute_retrieval_metrics(scores_i2t, scores_t2i, txt2img, img2txt):

    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    agg_metrics = (tr1 + tr5 + tr10) / 3

    eval_result = {
        "txt_r1": tr1,
        "txt_r5": tr5,
        "txt_r10": tr10,
        "txt_r_mean": tr_mean,
        "img_r1": ir1,
        "img_r5": ir5,
        "img_r10": ir10,
        "img_r_mean": ir_mean,
        "r_mean": r_mean,
        "agg_metrics": agg_metrics,
    }
    return eval_result
