# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .concat_dataset import ConcatDataset
from .cuhkpedes import CUHKPEDESDataset
from .market1501 import Market1501Dataset
from .dukemtmc import DukeMTMCDataset

__all__ = ["ConcatDataset",
           "CUHKPEDESDataset",
           "Market1501Dataset",
           "DukeMTMCDataset"]
