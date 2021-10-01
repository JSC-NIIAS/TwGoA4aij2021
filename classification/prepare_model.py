import os
import yaml
import segmentation_models_pytorch as smp
import torch
import argparse
import torch.nn as nn
import timm
from model_wrapper import Classification_model

def prepare_model(opt):
    with open(opt.hyp) as f:
        experiment_dict = yaml.load(f, Loader=yaml.FullLoader)
    model_pretrained = timm.create_model(experiment_dict['model']['name'], pretrained=experiment_dict['model']['pretrained'],num_classes=experiment_dict['model']['num_classes'])
    model = Classification_model(model_pretrained,experiment_dict['model']['model_type'],experiment_dict['model']['num_classes_mt'])
    model=nn.DataParallel(model)
    model.load_state_dict(torch.load(experiment_dict["savepath"])["model_state_dict"])
    model=model.module
    torch.save(model, experiment_dict["final_model_path"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp', type=str, default='configs/baseline_trailing_switch.yaml', help='hyperparameters path')
    opt = parser.parse_args()
    prepare_model(opt)