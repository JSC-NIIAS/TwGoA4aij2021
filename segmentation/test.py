from tqdm import tqdm
import torch
from sklearn.metrics import precision_score, f1_score
import numpy as np
import segmentation_models_pytorch as smp
SMOOTH = 1e-6


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):

    intersection = (outputs & labels).float().sum((1, 2))
    union = (outputs | labels).float().sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10

    return thresholded.mean()


def test_model(model,hyp,dataloader,wandb,rank,device,opt,metrics,amp,cuda,loss_1,loss_2):
    model.eval()
    f1 = smp.utils.metrics.Fscore().to(device)
    nb = len(dataloader)
    pbar = enumerate(dataloader)
    if rank in [-1, 0]:
        pbar = tqdm(pbar, total=nb)  # progress bar
    iou_scores=[]
    mean_iou_score=0
    mean_loss_val=0
    for i, (imgs, targets) in pbar:  # batch -------------------------------------------------------------
        with torch.no_grad():
            imgs = imgs.to(device, non_blocking=True).float()
            with amp.autocast(enabled=cuda):
               pred = model(imgs)  # forward
               loss_1_value = loss_1.forward(pred,targets.to(device))
               loss_2_value = loss_2.forward(pred,targets.to(device))
               loss = loss_1_value * hyp["train"]["weight_0"] + loss_2_value * hyp["train"]["weight_0"]
               mean_loss_val+=loss.item()
               pred = torch.argmax(torch.softmax(pred,dim=1),dim=1)
               iou_score=iou_pytorch(pred,targets.to(device))
               mean_iou_score+= iou_score.item()
        iou_scores.append(iou_score)
    results_dict = {}
    results_dict['mean_IOU']=mean_iou_score/len(iou_scores)
    results_dict['mean_val_loss']=mean_loss_val/len(dataloader)
    return results_dict
