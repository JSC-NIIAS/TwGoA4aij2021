import torch.nn as nn
import torch
from dataset import create_dataloader

class BoxRegressionLoss():
    def __init__(self):
        self.criterion = nn.SmoothL1Loss()

    def __call__(self, predict_reg, target_reg, target_keypoint):
        idx = target_keypoint > 0
        idx = torch.stack([idx]*3, axis=1)
        return self.criterion(predict_reg[idx], target_reg[idx])

class TotalLoss():
    def __init__(self, total_weights, keypoint_weights):
        self.weights = total_weights
        self.kp_loss = nn.CrossEntropyLoss(keypoint_weights)
        self.reg_loss = BoxRegressionLoss()

    def __call__(self, predict_keypoint, predict_reg, target_keypoint, target_reg):
        return self.kp_loss(predict_keypoint, target_keypoint) * self.weights[0] \
            + self.reg_loss(predict_reg, target_reg, target_keypoint) * self.weights[1]

def create_loss(keypoint_weights, total_weights):
    total_weights = torch.tensor(total_weights, dtype=torch.float32)
    keypoint_weights = torch.tensor(keypoint_weights, dtype=torch.float32)
    return TotalLoss(total_weights, keypoint_weights)



if __name__=='__main__':
    total_weights = torch.tensor([1., 1.])
    keypoint_weights = torch.tensor([1., 1.])
    loss = TotalLoss(total_weights, keypoint_weights)
    target_kp = torch.tensor([[[0, 1], [0, 0]], [[1, 1], [0, 1]]]).long()
    predict_kp = torch.tensor([[[[1, 0], [1, 1]], [[0, 0], [1, 0]]], [[[0, 1], [0, 0]], [[1, 1], [0, 1]]]]).float()
    target_reg = torch.tensor([[[[1, 2], [0, 2]], [[1, 1], [2, 2]]], [[[2, 1], [0, 1]], [[1, 0], [0, 0]]]]).float()
    predict_reg = torch.tensor([[[[2, 3], [1, 3]], [[2, 2], [3, 3]]], [[[3, 2], [1, 2]], [[2, 1], [1, 1]]]]).float()
    print(loss(predict_kp, predict_reg, target_kp, target_reg))