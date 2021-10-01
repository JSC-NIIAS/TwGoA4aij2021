import os
from dataset import TrainDataset
import yaml
import torch
with open("configs/test_config.yaml") as f:
    experiment_dict = yaml.load(f, Loader=yaml.FullLoader)
images_path="/files/hackhathon_baseline/SBER_RZD_COMPETITION_DATASET/train_data/images"
masks_path="/files/hackhathon_baseline/SBER_RZD_COMPETITION_DATASET/train_data/mask"
images=[file for file in os.listdir(images_path)]
images_paths=[os.path.join(images_path,file) for file in images]
masks_paths=[os.path.join(masks_path,file) for file in images]
dataset= TrainDataset(images_paths,masks_paths,mode="train",hyp=experiment_dict['dataset'])
image,label=dataset.__getitem__(0)
