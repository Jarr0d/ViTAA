import torch.utils.data

from vitaa.utils.comm import get_world_size
from vitaa.config.paths_catalog import DatasetCatalog
from vitaa.utils.caption import Caption

from . import datasets as D
from . import samplers
from .transforms import build_transforms
from .transforms import build_crop_transforms
from .collate_batch import collate_fn


def build_dataset(dataset_list,
                  transforms,
                  crop_transforms,
                  dataset_catalog,
                  is_train=True):
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    datasets = []
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name)
        factory = getattr(D, data["factory"])
        args = data["args"]
        if data["factory"] == "CUHKPEDESDataset":
            args["max_length"] = 100
            args["max_attribute_length"] = 25
            args["crop_transforms"] = crop_transforms
        # elif data["factory"] == "Market1501Dataset":
        #     args["is_train"] = is_train
        #     args["max_length"] = 100
        #     args["max_attribute_length"] = 35
        #     args["crop_transforms"] = crop_transforms
        # elif data["factory"] == "DukeMTMCDataset":
        #     args["is_train"] = is_train
        #     args["max_length"] = 100
        #     args["max_attribute_length"] = 35
        #     args["crop_transforms"] = crop_transforms
        args["transforms"] = transforms
        # make dataset from factory
        dataset = factory(**args)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return [dataset]


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(dataset, sampler, images_per_batch, images_per_pid, is_train=True):
    if is_train:
    # if False:
        batch_sampler = samplers.TripletSampler(
            sampler, dataset, images_per_batch, images_per_pid, drop_last=True
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    return batch_sampler


def make_data_loader(cfg, is_train=True, is_distributed=False):
    num_gpus = get_world_size()
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True

    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST

    transforms = build_transforms(cfg, is_train)
    crop_transforms = build_crop_transforms(cfg)
    datasets = build_dataset(
        dataset_list, transforms, crop_transforms, DatasetCatalog, is_train
    )

    data_loaders = []
    for dataset in datasets:
        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, images_per_gpu, cfg.DATALOADER.IMS_PER_ID, is_train
        )
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn
        )
        data_loaders.append(data_loader)
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders
