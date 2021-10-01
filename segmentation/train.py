import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import segmentation_models_pytorch as smp
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import timm
import logging
import os
from utils import ModelEmaV2, select_device, torch_distributed_zero_first,init_seeds,set_logging,get_x_y,createCheckpoint,get_data_lists
import argparse
import wandb
import yaml
from skmultilearn.model_selection import iterative_train_test_split,IterativeStratification
import sklearn
from sklearn.model_selection import train_test_split
from dataset import create_dataloader
from test import test_model
logger = logging.getLogger(__name__)

def train_model(hyp,opt,device,wandb):
    batch_size,total_batch_size =  opt.batch_size,opt.total_batch_size
    rank = opt.global_rank
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.hyp) as f:
        experiment_dict = yaml.load(f, Loader=yaml.FullLoader)
    model = smp.Linknet(encoder_name=experiment_dict["model"]["name"], encoder_depth=5, encoder_weights='imagenet', decoder_use_batchnorm=True, in_channels=3, classes=experiment_dict["model"]["classes"], aux_params=None).to(device)
    nbs = experiment_dict['train']['accumulate_batch_size']
    accumulate = max(round(nbs / total_batch_size), 1)
    experiment_dict['train']['weight_decay'] *= total_batch_size * accumulate / nbs
    loss_sce = smp.losses.SoftCrossEntropyLoss(smooth_factor=0.1).to(device)
    loss_dice = smp.losses.DiceLoss(mode="multiclass").to(device)
    metrics = smp.utils.metrics.IoU().to(device)

    if experiment_dict['train']['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=experiment_dict['train']['lr'], momentum=experiment_dict['train']['momentum'], nesterov=True,weight_decay=experiment_dict['train']['weight_decay'])

    if experiment_dict['train']['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=3e-4, betas=(experiment_dict['train']['momentum'], 0.999))
        # adjust beta1 to momentum
    if experiment_dict['train']['scheduler'] == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer,step_size=experiment_dict['train']['step_size'],gamma=experiment_dict['train']['gamma'])

    if experiment_dict['train']['ema']==True:
       ema = ModelEmaV2(model,decay=experiment_dict['train']['ema_decay']) if rank in [-1, 0] else None

    if cuda and rank == -1 and torch.cuda.device_count() > 1:
       model = torch.nn.DataParallel(model)

    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)

    if wandb and wandb.run is None:
        experiment_dict['batch_size']=batch_size
        experiment_dict['total_batch_size']=total_batch_size
        experiment_dict['workers']=opt.workers
        experiment_dict['sync_bn']=opt.sync_bn
        wandb_run = wandb.init(config=experiment_dict, resume="allow",
                               project=experiment_dict['project'],
                               name=experiment_dict['experiment_name'],
                               id=None)
    train_images,train_labels,val_images,val_labels = get_data_lists(experiment_dict["dataset"]["path_to_train_images"],experiment_dict["dataset"]["path_to_train_masks"],experiment_dict["dataset"]["path_to_val_masks"],experiment_dict["dataset"]["path_to_val_masks"])
    dataloader, dataset = create_dataloader(train_images,train_labels, batch_size,mode="train",
                                            hyp=experiment_dict['dataset'],
                                            rank=rank, world_size=opt.world_size, workers=opt.workers)
    start_epoch, best_fitness = 0, 0.0
    nb = len(dataloader)  # number of batches
    if rank in [-1, 0]:
        if experiment_dict['train']['ema']==True:
            ema.updates = start_epoch * nb // accumulate  # set EMA updates
        testloader,testset = create_dataloader(val_images,val_labels, batch_size,mode="val",
                                            hyp=experiment_dict['dataset'],
                                            rank=rank, world_size=opt.world_size, workers=opt.workers)

    scheduler.last_epoch = start_epoch - 1
    scaler = amp.GradScaler(enabled=cuda)
    epochs=experiment_dict['train']['epochs']
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        mean_train_loss = 0
        model.train()
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float()
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                loss_value_sce = loss_sce.forward(pred,targets.to(device))
                loss_value_dice = loss_dice.forward(pred, targets.to(device))
                loss_value = loss_value_sce * experiment_dict["train"]["weight_0"] + loss_value_dice * experiment_dict["train"]["weight_1"]
                if rank != -1:
                    loss_value *= opt.world_size  # gradient averaged between devices in DDP mode

            # Backward
            scaler.scale(loss_value).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if experiment_dict['train']['ema']==True:
                    ema.update(model)

            if rank in [-1, 0]:
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = str(epoch) +''+str(epochs - 1) +' '+str(mem) +' '+str(loss_value.item())
                mean_train_loss+=loss_value.item()
                pbar.set_description(s)
            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------
        scheduler.step()
        final_epoch = epoch + 1 == epochs
        results_dict = test_model(model,experiment_dict,testloader,wandb,rank,device,opt,metrics,amp,cuda,loss_sce,loss_dice)
        if results_dict['mean_IOU'] >= best_fitness:
            best_fitness = results_dict['mean_IOU']
            createCheckpoint(experiment_dict['savepath'],model,optimizer,epoch,scheduler)
        if wandb:
            results_dict['train_loss']=mean_train_loss/len(dataloader)
            wandb.log(results_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp', type=str, default='configs/baseline_config.yaml', help='hyperparameters path')
    parser.add_argument('--batch-size', type=int, default=12, help='total batch size for all GPUs')
    parser.add_argument('--device', default='0,1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    opt = parser.parse_args()

    opt.total_batch_size = opt.batch_size
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size
    logger.info(opt)
    train_model(opt.hyp,opt,device,wandb)