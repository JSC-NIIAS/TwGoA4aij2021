from tqdm import tqdm
import torch

def test_model(model,hyp,dataloader,wandb,rank,device,opt,metrics,amp,cuda,loss_function):
    model.eval()
    nb = len(dataloader)
    pbar = enumerate(dataloader)
    if rank in [-1, 0]:
        pbar = tqdm(pbar, total=nb)  # progress bar
    mean_loss_val=0
    results_dict={'mean_val_loss': 0}
    for metric in metrics:
        results_dict[metric.__name__] = 0

    for i, (imgs, keypoint_label, reg_label) in pbar:  # batch -------------------------------------------------------------
        with torch.no_grad():
            imgs = imgs.to(device, non_blocking=True).float()
            keypoint_label, reg_label = keypoint_label.to(device), reg_label.to(device)
            with amp.autocast(enabled=cuda):
                pred_keypoint, pred_reg = model(imgs)  # forward
                loss = loss_function(pred_keypoint, pred_reg, \
                     keypoint_label, reg_label)\

                mean_loss_val+=loss.item()
                for metric in metrics:
                    results_dict[metric.__name__] += metric((pred_keypoint, pred_reg), \
                        (keypoint_label, reg_label))
                        
    results_dict['mean_val_loss']=mean_loss_val/len(dataloader)
    for metric in metrics:
        results_dict[metric.__name__] /= len(dataloader)
    return results_dict
