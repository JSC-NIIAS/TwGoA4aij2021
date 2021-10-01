from tqdm import tqdm
import torch
from sklearn.metrics import precision_score, f1_score
import numpy as np

def test_model(model,hyp,dataloader,wandb,rank,device,opt,metrics,amp,cuda,loss_function):
    model.eval()
    nb = len(dataloader)
    pbar = enumerate(dataloader)
    if rank in [-1, 0]:
        pbar = tqdm(pbar, total=nb)  # progress bar
    f1_scores=[]
    mean_loss_val=0
    mean_f1_score=0
    results_dict={}
    for i, (imgs, targets) in pbar:  # batch -------------------------------------------------------------
        with torch.no_grad():
            imgs = imgs.to(device, non_blocking=True).float()
            with amp.autocast(enabled=cuda):
               pred = model(imgs)  # forward
               loss = loss_function(pred,targets.to(device))
               f1_one_batch=metrics(pred,targets.to(device))
               mean_loss_val+=loss.item()
               mean_f1_score+=f1_one_batch.item()
               f1_scores.append(f1_one_batch.item())
    results_dict['mean_F1_val']=mean_f1_score/len(f1_scores)
    results_dict['mean_val_loss']=mean_loss_val/len(dataloader)
    return results_dict
