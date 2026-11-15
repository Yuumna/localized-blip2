from typing import Any, List, Tuple
import json

import torch

from lavis.common.dist_utils import is_dist_avail_and_initialized
from lavis.models.base_model import GatherLayer

def resize_bbox(bbox, image_width, image_height, new_image_width, new_image_height):
    """
    Resize bounding boxes according to image resize.
    Args:
        bbox (list[float]): [x, y, w, h]
        image_width (int): width of the image
        image_height (int): height of the image
        new_image_width (int): resized image width
        new_image_height (int): resized image height
    Returns:
        list[float]: resized bounding box
    """
    new_bbox = [0, 0, 0, 0]
    new_bbox[0] = bbox[0] * new_image_width / image_width
    new_bbox[1] = bbox[1] * new_image_height / image_height
    new_bbox[2] = bbox[2] * new_image_width / image_width
    new_bbox[3] = bbox[3] * new_image_height / image_height

    return new_bbox


class LookupTables:
    """
    A class that stores lookup tables for a list of dictionaries

    Example:
        list_of_dict = [
            {"a": 1, "b": 2, "c": 3}
            {"a": 1, "b": 9, "c": 3}
            {"a": 4, "b": 5, "c": 6}
            {"a": 7, "b": 8, "c": 9}
        ]
        lookup_tables = LookupTables(list_of_dict)
        lookup_tables.get("a", "b", 1) # return [2, 9]
        lookup_tables.get("a", "c", 7) # return [9]
    """

    def __init__(self, list_of_dict):
        self.list_of_dict = list_of_dict
        self.all_keys = self.list_of_dict[0].keys()
        self.lookup_tables = {}

    def get(self, query_dict_key: str, target_dict_key: str, query_value: Any) -> List[Any]:
        lookup_table = self._get_lookup_table(query_dict_key, target_dict_key)
        return lookup_table[query_value]

    def _get_lookup_table(self, query_dict_key: str, target_dict_key: str):
        assert query_dict_key in self.all_keys
        assert target_dict_key in self.all_keys

        if (query_dict_key, target_dict_key) not in self.lookup_tables.keys():
            # create a lookup table from query_dict_key to target_dict_key
            table = {}
            for d in self.list_of_dict:
                if d[query_dict_key] not in table.keys():
                    table[d[query_dict_key]] = []
                table[d[query_dict_key]].append(d[target_dict_key])

            self.lookup_tables[(query_dict_key, target_dict_key)] = table

        return self.lookup_tables[(query_dict_key, target_dict_key)]


def pretty_json(hp):
    json_hp = json.dumps(hp, indent=2)
    return "".join("\t" + line for line in json_hp.splitlines(True))


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensors

    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    # tensor_all = GatherLayer.apply(tensors)
    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)


def get_normalized_bounding_box(bb_coords: List[int], image_dims: Tuple):
    im_width, im_height = image_dims
    x, y, w, h = bb_coords
    return [
        x / im_width,
        y / im_height,
        w / im_width,
        h / im_height
    ]
