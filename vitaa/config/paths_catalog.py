# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = "datasets"
    DATASETS = {
        "cuhkpedes_train": {
            "img_dir": "cuhkpedes",
            "ann_file": "cuhkpedes/annotations/train.json"
        },
        "cuhkpedes_val": {
            "img_dir": "cuhkpedes",
            "ann_file": "cuhkpedes/annotations/val.json"
        },
        "cuhkpedes_test": {
            "img_dir": "cuhkpedes",
            "ann_file": "cuhkpedes/annotations/test.json"
        },
        # "market1501_train": {
        #     "img_dir": "market1501",
        #     "ann_dir": "market1501/annotations"
        # },
        # "market1501_test": {
        #     "img_dir": "market1501",
        #     "ann_dir": "market1501/annotations"
        # },
        # "dukemtmc_train": {
        #     "img_dir": "dukemtmc",
        #     "ann_dir": "dukemtmc/annotations"
        # },
        # "dukemtmc_test": {
        #     "img_dir": "dukemtmc",
        #     "ann_dir": "dukemtmc/annotations"
        # },
    }

    @staticmethod
    def get(name):
        if "cuhkpedes" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="CUHKPEDESDataset",
                args=args,
            )
        # elif "market1501" in name:
        #     data_dir = DatasetCatalog.DATA_DIR
        #     attrs = DatasetCatalog.DATASETS[name]
        #     args = dict(
        #         root=os.path.join(data_dir, attrs["img_dir"]),
        #         ann_root=os.path.join(data_dir, attrs["ann_dir"]),
        #     )
        #     return dict(
        #         factory="Market1501Dataset",
        #         args=args,
        #     )
        # elif "dukemtmc" in name:
        #     data_dir = DatasetCatalog.DATA_DIR
        #     attrs = DatasetCatalog.DATASETS[name]
        #     args = dict(
        #         root=os.path.join(data_dir, attrs["img_dir"]),
        #         ann_root=os.path.join(data_dir, attrs["ann_dir"]),
        #     )
        #     return dict(
        #         factory="DukeMTMCDataset",
        #         args=args,
        #     )
        raise RuntimeError("Dataset not available: {}".format(name))