import torch
import torch.utils.data as data
import numpy as np
import os
import json
from PIL import Image

from vitaa.utils.caption import Caption


class CUHKPEDESDataset(data.Dataset):
    def __init__(self,
                 root,
                 ann_file,
                 max_length=100,
                 max_attribute_length=25,
                 transforms=None,
                 crop_transforms=None,
                 cap_transforms=None):
        self.root = root
        self.max_length = max_length
        self.max_attribute_length = max_attribute_length
        self.transforms = transforms
        self.crop_transforms = crop_transforms
        self.cap_transforms = cap_transforms

        self.img_dir = os.path.join(self.root, 'imgs')
        self.seg_dir = os.path.join(self.root, 'segs')

        print('loading annotations into memory...')
        self.dataset = json.load(open(ann_file, 'r'))

    def __getitem__(self, index):
        """
        Args:
              index(int): Index
        Returns:
              tuple: (images, labels, captions)
        """
        data = self.dataset['annotations'][index]

        img_path = data['file_path']
        img = Image.open(os.path.join(self.img_dir, img_path)).convert('RGB')
        seg = Image.open(os.path.join(self.seg_dir, img_path.split('.')[0]+'.png'))

        caption = data['onehot']
        caption = torch.tensor(caption)
        caption = Caption([caption], max_length=self.max_length, padded=False)
        caption.add_field("img_path", img_path)

        attribute = data['att_onehot']
        attribute_list = [torch.tensor(v) for k, v in attribute.items()]
        attribute = Caption(attribute_list, max_length=self.max_attribute_length, padded=False)
        attribute.add_field("mask", attribute.length > 0)
        attribute.length[attribute.length < 1] = 1
        caption.add_field("attribute", attribute)

        label = data['id']
        label = torch.tensor(label)
        caption.add_field("id", label)

        if self.transforms is not None:
            img, seg = self.transforms(img, seg)

        if self.crop_transforms is not None:
            crops, mask = self.crop_transforms(img, seg)
            caption.add_field("crops", crops)
            caption.add_field("mask", mask)

        if self.cap_transforms is not None:
            caption = self.cap_transforms(caption)

        return img, caption, index

    def __len__(self):
        return len(self.dataset['annotations'])

    def get_id_info(self, index):
        image_id = self.dataset['annotations'][index]['image_id']
        pid = self.dataset['annotations'][index]['id']
        return image_id, pid



