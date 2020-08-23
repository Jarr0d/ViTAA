import argparse
import os
import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from vitaa.config import cfg
from vitaa.utils.comm import synchronize, get_rank
from vitaa.utils.logger import setup_logger
from vitaa.models.model import build_model
from vitaa.utils.checkpoint import Checkpointer
from vitaa.utils.directory import makedir
from vitaa.data import make_data_loader
from vitaa.engine.inference import inference


def main():
    parser = argparse.ArgumentParser(description="PyTorch Image-Text Matching Inference")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--checkpoint-file",
        default="",
        metavar="FILE",
        help="path to checkpoint file",
        type=str,
    )
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--load-result",
        help="Use saved reslut as prediction",
        action='store_true',
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("PersonSearch", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    model = build_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = Checkpointer(model, save_dir=output_dir, logger=logger)
    _ = checkpointer.load(args.checkpoint_file)

    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            makedir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            device=cfg.MODEL.DEVICE,
            output_folder=output_folder
        )
        synchronize()


if __name__ == "__main__":
    main()
