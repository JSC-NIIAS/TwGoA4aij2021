import os
import glob
import numpy as np
from PIL import Image, ImageFile
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
import cv2
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset,DataLoader
import torch
from utils import torch_distributed_zero_first
from torchvision import transforms

def create_dataloader(x,y,batch_size,mode="train",hyp=None, augment=False,
                      rank=-1, world_size=1, workers=8):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = TrainDataset(x,y,mode=mode,
                                      hyp=hyp,  # augmentation hyperparameters
                                      rank=rank)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    dataloader = InfiniteDataLoader(dataset,
                                    shuffle=True,
                                    batch_size=batch_size,
                                    num_workers=nw,
                                    sampler=sampler,
                                    pin_memory=True,
                                    collate_fn=TrainDataset.collate_fn,drop_last=True)  # torch.utils.data.DataLoader()
    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers
    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class TrainDataset(torch.utils.data.Dataset):


    def __init__(self,x,y,mode="train", hyp=None,rank=-1):
        self.mode=mode
        self.hyp = hyp
        self.img_paths = x
        self.labels = y
        self.transformations = {
            'train': self.get_train_transforms(),
            'val': self.get_val_transforms(),
        }

        print(self.transformations)

    def get_train_transforms(self):
        alb_transforms = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.Resize(height=self.hyp['img_size_base'][0],width=self.hyp["img_size_base"][1]),
            A.OneOf([
                 A.RandomCrop(int(self.hyp['img_size_base'][0]*0.5), int(self.hyp['img_size_base'][1]*0.5), p=1),  # 813, 2048
                 A.RandomCrop(int(self.hyp['img_size_base'][0]*0.75), int(self.hyp['img_size_base'][1]*0.75), p=1),    # 1220, 3072
                 A.RandomCrop(int(self.hyp['img_size_base'][0]*0.8), int(self.hyp['img_size_base'][1]*0.8), p=1),  # 1300, 3277
            ], p=0.9),
            A.Resize(height=self.hyp['img_size_target'][0],width=self.hyp["img_size_target"][1]),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=(0.8, 1.4), rotate_limit=0, p=0.3),
            A.OneOf([
                A.MotionBlur(blur_limit=25, p=1),
                A.Blur(blur_limit=25, p=1),
                # A.MedianBlur(blur_limit=25, p=1),
            ], p=0.2),
            A.GaussNoise(p=0.2),
            A.Normalize([self.hyp['mean0'], self.hyp['mean1'], self.hyp['mean2']], [self.hyp['std0'], self.hyp['std1'], self.hyp['std2']]),
            ToTensorV2(),
        ], additional_targets={'label': 'mask'})
        return alb_transforms

    def get_val_transforms(self):
        alb_transforms = A.Compose([
            A.Resize(height=self.hyp['img_size_base'][0],width=self.hyp["img_size_base"][1]),
            A.Resize(height=self.hyp['img_size_target'][0],width=self.hyp["img_size_target"][1]),
            A.Normalize([self.hyp['mean0'], self.hyp['mean1'], self.hyp['mean2']], [self.hyp['std0'], self.hyp['std1'], self.hyp['std2']]),
            ToTensorV2(),
        ], additional_targets={'label': 'mask'})
        return  alb_transforms

    def map_target(self,target):
        """
        Maps target encoded with classic id classes to target encoded in train_id classes
        """
        target = np.array(target, dtype=np.uint8)
        target[target==10] = 3
        target[target==7] = 2
        target[target==6] = 1
        return Image.fromarray(target)

    @staticmethod
    def pil2cv(sample):
        sample['image'] = np.array(sample['image'])[:, :, ::-1]
        sample['label'] = np.array(sample['label'])
        return sample

    @staticmethod
    def alb_transform_wrapper(transform, sample):
        sample = TrainDataset.pil2cv(sample)
        sample = transform(**sample)
        sample['label'] = sample['label'].squeeze()
        return sample

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index]).convert("RGB")
        target = Image.open(self.labels[index]).convert('L')
        target = self.map_target(target)
        sample={"image":image,"label":target}
        sample=TrainDataset.alb_transform_wrapper(self.transformations[self.mode],sample)
        image = sample["image"]
        target=sample["label"]
        #label = torch.tensor(list(self.labels[index]),dtype=torch.float32)
        #image=self.image_pipeline(image)
        return image,target.long()

    @staticmethod
    def collate_fn(batch):
        img, label = zip(*batch)  # transposed
        return torch.stack(img, 0), torch.stack(label, 0)