from dataset import DataProcessor
import random
import numpy as np
import argparse
from utils import get_data_lists

def calculate_weights(X, y, n_classes, seed=42, sample_count=10):
    dataset = DataProcessor(X, y, n_classes)
    pad = dataset.pad
    pixel_count = (dataset.img_size[0] + np.sum(pad[0])) * (dataset.img_size[1] + np.sum(pad[1]))    

    random.seed(seed)

    class_count = np.zeros((n_classes, sample_count), dtype=int)

    sample_id = 0
    while sample_id < sample_count:
        for _, keypoint_label, _ in dataset:
            if (sample_id >= sample_count): break

            unique, counts = np.unique(keypoint_label, return_counts=True)
            for clazz, count in zip(unique, counts):
                class_count[clazz, sample_id] = count

            sample_id += 1

    median = np.median(class_count, axis=1)
    return 1 - median / pixel_count

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_count', type=int, default=100000, help='number of measurements')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--img_path', type=str, help='path to images')
    parser.add_argument('--label_path', type=str, help='path to labels')
    parser.add_argument('--n_classes', type=int, help='class numbers')
    opt = parser.parse_args()

    X, y = get_data_lists(opt.img_path,opt.label_path)
    weights = calculate_weights(X, y, opt.n_classes, opt.seed, opt.sample_count)
    np.savetxt("keypoints_weights.txt", weights) 