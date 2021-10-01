import torch
import os
from utils import torch_distributed_zero_first
import PIL
from PIL import Image
from torchvision import transforms
from PIL import Image, ImageFile
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
import cv2
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np

def create_dataloader(x,y,batch_size,hyp=None,mode="train",
                      rank=-1, world_size=1, workers=8):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = TrainDataset(x,y,mode="train",
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
                                    collate_fn=TrainDataset.collate_fn)  # torch.utils.data.DataLoader()
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
        self.mode = mode
        self.hyp = hyp
        self.img_paths = x
        self.labels = y
        self.transformations = {
            'train': self.get_train_transforms(),
            'val': self.get_val_transforms(),
        }

    def get_train_transforms(self):
        alb_transforms = A.Compose([
            A.RandomBrightnessContrast(p=0.8),
            A.HueSaturationValue(p=0.8),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=(0.8, 1.4), rotate_limit=0, p=0.1),
            A.OneOf([
                A.MotionBlur(blur_limit=25, p=1),
                A.Blur(blur_limit=25, p=1),
                # A.MedianBlur(blur_limit=25, p=1),
            ], p=0.3),
            A.GaussNoise(p=0.5),
            A.Normalize([self.hyp['mean0'], self.hyp['mean1'], self.hyp['mean2']], [self.hyp['std0'], self.hyp['std1'], self.hyp['std2']]),
            ToTensorV2(),
        ])
        return alb_transforms

    def get_val_transforms(self):
        alb_transforms = A.Compose([
            A.Normalize([self.hyp['mean0'], self.hyp['mean1'], self.hyp['mean2']], [self.hyp['std0'], self.hyp['std1'], self.hyp['std2']]),
            ToTensorV2(),
        ])
        return  alb_transforms

    @staticmethod
    def pil2cv(sample):
        sample['image'] = np.array(sample['image'])[:, :, ::-1]
        return sample

    @staticmethod
    def alb_transform_wrapper(transform, sample):
        sample = TrainDataset.pil2cv(sample)
        sample = transform(**sample)
        return sample

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        path = self.img_paths[index]
        image = Image.open(path).convert("RGB")
        label = torch.tensor(self.labels[index],dtype=torch.float32)
        sample = {"image": image}
        sample = TrainDataset.alb_transform_wrapper(self.transformations[self.mode], sample)
        image = sample["image"]
        return image,label.long()

    @staticmethod
    def collate_fn(batch):
        img, label = zip(*batch)  # transposed
        return torch.stack(img, 0), torch.stack(label, 0)
