import torch
from torch import tensor
import os
from utils import torch_distributed_zero_first
from PIL import Image, ImageFile
import cv2
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import random
import matplotlib.pyplot as plt

def create_dataloader(x, y, batch_size,n_classes,
                      rank=-1, world_size=1, workers=8):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = DataProcessor(x, y, n_classes)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    dataloader = InfiniteDataLoader(dataset,
                                    shuffle=True,
                                    batch_size=batch_size,
                                    num_workers=nw,
                                    sampler=sampler,
                                    pin_memory=True,
                                    collate_fn=DataProcessor.collate_fn)  # torch.utils.data.DataLoader()
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

class DataProcessor(torch.utils.data.Dataset):
    def __init__(self, x, y, n_classes, cell_size=0.25):
        self.img_paths = x
        self.labels = y
        self.img_size = self.get_image_size()
        self.n_classes = n_classes
        self.img_center = (self.img_size[0] // 2, self.img_size[1] // 2)
        self.indices = [i for i in range(self.__len__())]
        self.mosaic_p = 0.1
        self.cell_size = cell_size
        self.threshold = 0.3
        self.pad = self.calc_pad()
        self.additional_offset = 5

    def __len__(self):
        return len(self.img_paths)

    def calc_pad(self):
        np_size = np.array(self.img_size)
        desired_size = np.ceil(np_size / 32) * 32
        difference = desired_size - np_size

        pad = np.zeros((3, 2), dtype=int)
        for i, diff in enumerate(difference):
            pad[i, 0] = np.ceil(diff / 2)
            pad[i, 1] = diff // 2

        # pad = ((pad[0,0], pad[0, 1]), (pad[1, 0], pad[1, 1]), (0, 0))
        return pad

    def get_image_size(self):
        if len(self.img_paths):
            img = cv2.imread(self.img_paths[0])
            height, width, _ = img.shape
            return (height, width)
        else: return np.zeros(2)


    def create_labels(self, label):
        gt_keypoint_label = np.zeros((self.n_classes, self.img_size[0], self.img_size[1]))
        gt_reg_label = np.zeros((self.img_size[0], self.img_size[1], 3))
        for gt_l in label:
            center_y = int(self.img_size[0] - gt_l[0] / self.cell_size)
            center_x = int(self.img_size[1] / 2 - gt_l[1] / self.cell_size)
            gt_keypoint_label[int(gt_l[2])][center_y][center_x] = 1
            gt_reg_label[center_y][center_x][0] = gt_l[3]
            gt_reg_label[center_y][center_x][1] = gt_l[4]
            gt_reg_label[center_y][center_x][2] = gt_l[5]
        gt_keypoint_label = np.argmax(gt_keypoint_label, axis=0)
        return gt_keypoint_label, gt_reg_label

    def get_object_info(self, label):
        center_y = int(self.img_size[0] - label[0] / self.cell_size)
        center_x = int(self.img_size[1] / 2 - label[1] / self.cell_size)
        width = label[4] / self.cell_size
        height = label[5] / self.cell_size

        return center_x, center_y, width, height

    # rect = (x1a, y1a, x2a, y2a)
    def is_in_rect(self, rect, point):
        return point[0] >= rect[0] and point[1] >= rect[1] \
            and point[0] <= rect[2] and point[1] <= rect[3]

    def calc_intersect_percentage(self, candidate, rect):
        candidate_area = (candidate[0] - candidate[2]) * (candidate[1] - candidate[3])

        left = max(candidate[0], rect[0])
        top = max(candidate[1], rect[1])
        right = min(candidate[2], rect[2])
        bottom = min(candidate[3], rect[3])

        if left >= right or top >= bottom:
            return 0

        intersect_area = (right - left) * (bottom - top)

        return intersect_area / candidate_area

    def move_rect_along_axis(self, bound1, bound2, p, max):
        delta = 0
        if p < bound1:
            additional_offset = min(self.additional_offset, p)
            delta = bound1 - p + additional_offset
        elif p > bound2:
            additional_offset = min(self.additional_offset, max - p)
            delta = bound2 - p - additional_offset

        return bound1 - delta, bound2 - delta

    def fix_coords(self, x1, y1, x2, y2, labels):
        for label in labels:
            c_x, c_y, width, height = self.get_object_info(label)
            rect = (x1, y1, x2, y2)

            if self.is_in_rect(rect, (c_x, c_y)):
                continue
            
            left = c_x - width / 2
            right = c_x + width / 2
            top = c_y - height / 2
            down = c_y + height / 2

            candidate = (left, top, right, down)
            
            if (self.calc_intersect_percentage(candidate, rect) < self.threshold):
                continue

            x1, x2 = self.move_rect_along_axis(x1, x2, c_x, self.img_size[1] - 1)
            y1, y2 = self.move_rect_along_axis(y1, y2, c_y, self.img_size[0] - 1)

        return x1, y1, x2, y2


    def create_new_coords(self,x1a, y1a, x2a, y2a):
        delta_x = random.uniform(0, 200)
        delta_y = random.uniform(0, 200)
        lenght_x = x2a - x1a
        lenght_y = y2a - y1a
        if x1a==y1a:
            x2b = min(x2a + delta_x, self.img_size[1])
            y2b = min(y2a + delta_y, self.img_size[0])
            x1b = x2b - lenght_x
            y1b = y2b - lenght_y
        if x2a == self.img_size[1] and y1a==0:
            x1b = max(x1a - delta_x, 0)
            y2b = min(y2a + delta_y, self.img_size[0])
            x2b = x1b + lenght_x
            y1b = y2b - lenght_y
        if y2a ==  self.img_size[0] and x1a == 0:
            x2b = min(x2a + delta_x, self.img_size[1])
            x1b = x2b - lenght_x
            y1b = max(y1a - delta_y, 0)
            y2b = y1b + lenght_y
        if x2a == self.img_size[1] and y2a  ==  self.img_size[0]:
            x1b = max(x1a - delta_x, 0)
            y1b = max(y1a - delta_y, 0)
            x2b = x1b + lenght_x
            y2b = y1b + lenght_y
        return int(x1b),int(y1b),int(x2b),int(y2b)

    def __getitem__(self, index):
        image_out : np.ndarray
        keypoint_label : np.ndarray
        reg_label : np.ndarray

        if random.random() > self.mosaic_p:
            xc = int(random.uniform(int(self.img_center[1] // 2), self.img_size[1]))
            yc = int(random.uniform(int(self.img_center[0] // 2), self.img_size[0]))
            indices = [index] + random.choices(self.indices, k=3)
            images = [cv2.imread(self.img_paths[i]) for i in indices]
            labels = [self.create_labels(self.labels[i]) for i in indices]
            keypoint_label = np.zeros((self.img_size[0], self.img_size[1]))
            reg_label = np.zeros((self.img_size[0], self.img_size[1],3))
            image_out = np.full((self.img_size[0], self.img_size[1], 3), 114, dtype=np.uint8)
            for i, (image, label) in enumerate(zip(images, labels)):
                if i == 0:
                    x1a, y1a, x2a, y2a = 0, 0, xc, yc
                elif i == 1:
                    x1a, y1a, x2a, y2a = xc, 0, self.img_size[1], yc
                elif i == 2:
                    x1a, y1a, x2a, y2a = 0, yc, xc, self.img_size[0]
                elif i == 3:
                    x1a, y1a, x2a, y2a = xc, yc, self.img_size[1], self.img_size[0]

                x1b, y1b, x2b, y2b = self.create_new_coords(x1a, y1a, x2a, y2a)
                x1b, y1b, x2b, y2b = self.fix_coords( \
                    x1b, y1b, x2b, y2b, self.labels[indices[i]])
                    
                image_out[y1a:y2a, x1a:x2a, :3] = image[int(y1b):int(y2b), int(x1b):int(x2b), :3]
                keypoint_label[y1a:y2a, x1a:x2a] = label[0][int(y1b):int(y2b), int(x1b):int(x2b)]
                reg_label[y1a:y2a, x1a:x2a, :3] = label[1][int(y1b):int(y2b), int(x1b):int(x2b)]
        else:
            image_out = cv2.imread(self.img_paths[index])
            keypoint_label, reg_label = self.create_labels(self.labels[index])

        image_out = np.pad(image_out, self.pad, mode='constant', constant_values=0)
        keypoint_label = np.pad(keypoint_label, self.pad[:-1], mode='constant', constant_values=0)
        reg_label = np.pad(reg_label, self.pad, mode='constant', constant_values=0)
        image_out = np.rollaxis(image_out, 2)
        reg_label = np.rollaxis(reg_label, 2)
        return tensor(image_out), tensor(keypoint_label, dtype=torch.int64), tensor(reg_label, dtype=torch.float32)

    @staticmethod
    def collate_fn(batch):
        img, keypoint_label, reg_label = zip(*batch)  # transposed
        return torch.stack(img, 0), torch.stack(keypoint_label, 0), torch.stack(reg_label, 0)