import torch
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from kindle import Model
import logging
import os
from utils import ModelEmaV2, select_device, init_seeds,set_logging,createCheckpoint,get_data_lists
import argparse
import wandb
import yaml
from loss import create_loss
from sklearn.model_selection import train_test_split
from dataset import create_dataloader
from test import test_model
import metric
logger = logging.getLogger(__name__)


def train_model(hyp,opt,device,wandb):
    batch_size,total_batch_size =  opt.batch_size,opt.total_batch_size
    rank = opt.global_rank
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.hyp) as f:
        experiment_dict = yaml.load(f, Loader=yaml.FullLoader)
    model = Model(experiment_dict['model']['config'],verbose=True).to(device)
    nbs = experiment_dict['train']['accumulate_batch_size']
    accumulate = max(round(nbs / total_batch_size), 1)
    experiment_dict['train']['weight_decay'] *= total_batch_size * accumulate / nbs


    if experiment_dict['train']['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=experiment_dict['train']['lr'], momentum=experiment_dict['train']['momentum'], nesterov=True,weight_decay=experiment_dict['train']['weight_decay'])

    if experiment_dict['train']['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=experiment_dict['train']['lr'], betas=(experiment_dict['train']['momentum'], 0.999))
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
    x_train,y_train=get_data_lists(experiment_dict["dataset"]["images_path"],experiment_dict["dataset"]["labels_path"])
    loss_function = create_loss(keypoint_weights=experiment_dict['loss']['keypoint_weights'], \
        total_weights=experiment_dict['loss']['total_weights'])
    metrics = [getattr(metric, met) for met in experiment_dict['metrics']]
    X_train, X_test, y_train, y_test =train_test_split(x_train, y_train, test_size=experiment_dict['dataset']['test_size'], random_state=2 + rank)
    dataloader, dataset = create_dataloader(X_train,y_train, batch_size,
                                            n_classes=experiment_dict['n_classes'],
                                            rank=rank, world_size=opt.world_size, workers=opt.workers)
    start_epoch, best_fitness = 0, 0
    nb = len(dataloader)  # number of batches
    if rank in [-1, 0]:
        if experiment_dict['train']['ema']==True:
            ema.updates = start_epoch * nb // accumulate  # set EMA updates
        testloader,testset = create_dataloader(X_test,y_test, batch_size,
                                            n_classes=experiment_dict['n_classes'],
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
        for i, (imgs, keypoint_label, reg_label) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float()
            with amp.autocast(enabled=cuda):
                pred_keypoint, pred_reg = model(imgs)  # forward
                loss = loss_function(pred_keypoint, pred_reg, \
                     keypoint_label.to(device), reg_label.to(device))
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if experiment_dict['train']['ema']==True:
                    ema.update(model)

            if rank in [-1, 0]:
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = str(epoch) +''+str(epochs - 1) +' '+str(mem) +' '+str(loss.item())
                mean_train_loss+=loss.item()
                pbar.set_description(s)
            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------
        scheduler.step()
        final_epoch = epoch + 1 == epochs
        results_dict = test_model(model,hyp,testloader,wandb,rank,device,opt,metrics,amp,cuda,loss_function)
        if results_dict['mean_val_loss'] >= best_fitness:
            best_fitness = results_dict['mean_val_loss']
            createCheckpoint(experiment_dict['savepath'],model,optimizer,epoch,scheduler)
        if wandb:
            results_dict['train_loss']=mean_train_loss/len(dataloader)
            wandb.log(results_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp', type=str, default='configs/experiment_config.yaml', help='hyperparameters path')
    parser.add_argument('--batch-size', type=int, default=64, help='total batch size for all GPUs')
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
    train_model(opt.hyp,opt,device, wandb)
