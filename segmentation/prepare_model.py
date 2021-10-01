import os
import yaml
import segmentation_models_pytorch as smp
import torch
import argparse
import torch.nn as nn


def prepare_model(opt):
    with open(opt.hyp) as f:
        experiment_dict = yaml.load(f, Loader=yaml.FullLoader)
    model = smp.Linknet(encoder_name=experiment_dict["model"]["name"], encoder_depth=5, encoder_weights='imagenet',
                    decoder_use_batchnorm=True, in_channels=3, classes=experiment_dict["model"]["classes"],
                    aux_params=None)
    device=torch.device("cuda:0")
    model=nn.DataParallel(model)
    model.load_state_dict(torch.load(experiment_dict["savepath"])["model_state_dict"])
    model=model.module
    torch.save(model, experiment_dict["final_model_path"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp', type=str, default='configs/baseline_config.yaml', help='hyperparameters path')
    opt = parser.parse_args()
    prepare_model(opt)