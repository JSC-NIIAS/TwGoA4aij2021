from typing import Tuple
import torchmetrics
from typing import List, Union, Tuple
import torch
import numpy as np

def F1(input : Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], \
    target : Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]) -> torch.Tensor:
    keypount_input, _ = input
    keypount_target, _ = target
    return torchmetrics.functional.f1(keypount_input, keypount_target, mdmc_average='global')

def IoU(input : Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], \
    target : Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]) -> torch.Tensor:
    _, reg_input = input
    keypount_target, reg_target = target

    idx = keypount_target > 0
    out_input = np.rollaxis(reg_input.numpy(), axis=1, start=reg_input.ndim)
    out_input = out_input[idx]
    out_target = np.rollaxis(reg_target.numpy(), axis=1, start=reg_target.ndim)[idx]
    min_arr = np.zeros(out_input.shape)
    for i, (input, target) in enumerate(zip(out_input, out_target)):
        min_arr[i] = np.minimum(input, target)
    
    intersection = np.prod(min_arr, axis=1)
    input_area = np.prod(out_input, axis=1)
    target_area = np.prod(out_target, axis=1)
    union = input_area + target_area - intersection
    return torch.as_tensor(np.sum(intersection / union, where=(union != 0)), dtype=torch.float32)


if __name__=='__main__':
    input_reg = torch.zeros((1, 3, 4, 5))
    input_reg[0, 0, 0, 0] = 1
    input_reg[0, 1, 0, 0] = 5
    input_reg[0, 2, 0, 0] = 9
    input = (torch.rand((1, 3, 1, 1)), input_reg)
    target_reg = torch.zeros((1, 3, 4, 5))
    target_reg[0, 0, 0, 0] = 2
    target_reg[0, 1, 0, 0] = 4
    target_reg[0, 2, 0, 0] = 8
    target_keypoint = torch.ones((1, 4, 5))
    target = (target_keypoint, target_reg)
    print(IoU(input, target))

