from __future__ import annotations
from typing import Literal
from .cifar10 import make_cifar10
from .coco2017 import make_coco2017


def make_dataset(name: Literal["cifar10", "coco2017"], **kwargs):
    if name == "cifar10":
        return make_cifar10(**kwargs)
    if name == "coco2017":
        return make_coco2017(**kwargs)
    raise ValueError(f"Unknown dataset: {name}")
