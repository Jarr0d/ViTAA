from . import transforms as T


def build_transforms(cfg, is_train=True):
    height = cfg.INPUT.HEIGHT
    width = cfg.INPUT.WIDTH
    ratio = cfg.INPUT.DOWNSAMPLE_RATIO
    padding = cfg.INPUT.PADDING

    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
    )

    if is_train:
        transform = T.Compose(
            [
                T.Resize((height, width), ratio=ratio),
                T.RandomHorizontalFlip(0.5),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    else:
        transform = T.Compose(
            [
                T.Resize((height, width), ratio=ratio),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    return transform


def build_crop_transforms(cfg):
    num_parts = cfg.MODEL.NUM_PARTS
    transform = T.Split(num_parts, use_binary=False)
    return transform
