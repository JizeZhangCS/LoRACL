#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import os
from pathlib import Path
import random
import shutil
import time
import warnings
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from utils.dataset import MemoryDataset, MultipleDataset
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import NearestNeighbors
import simclr.builder
import simclr.loader
from utils.ddp import concat_all_gather_interlace
from utils.valid import ret_metrics, cls_metrics
from utils.meters import AverageMeter, ProgressMeter, _collect_lora_grad, accuracy
import lib.loralib as lora
from trainer.contrast_trainer_moco import ContrastTrainer
from lib.loralib.singleton_lora_piggybank import loraloss_pgbk
from datetime import datetime

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data',
                    default='../datasets/PACS/sketch',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--weights', default='',
                    help='default random init, use weights=\"DEFAULT\" to load default torchvision weights, other choices could be \"IMAGENET1K_V1\", etc.')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4096, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--lora_lr', default=0.03, type=float, help='initial lora learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--lora_wd', default=2e-4, type=float, help='lora weight decay (default: 2e-4)')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--enable_lora', action='store_true',
                    help='Use lora as supplement of augmentation')
parser.add_argument('--lora_start_epoch', default=1000, type=int,
                    help='epoch to start lora')
parser.add_argument('--lora_fix_epoch', default=1000, type=int,
                    help='epoch to fix lora')
parser.add_argument('--enable_aug', action='store_true',
                    help='enable moco v2\'s augmentation')
parser.add_argument('--iid_val_data', type=str, default='../datasets/PACS/sketch', 
                    help='path to file with source domain image paths and classes')
parser.add_argument('--ood_val_data', type=str, default='../datasets/PACS/art_painting,../datasets/PACS/cartoon,../datasets/PACS/photo',
                    help='path to file with target domain image paths and classes')

parser.add_argument('--valid_freq', default=10, type=int, help='frequency of validation on val set')
parser.add_argument('--save_freq', default=50, type=int, help='frequency of saving checkpoints')
parser.add_argument('--lora_layers', default='12', type=str,
                    help='the layers of ResNet that would attach lora.')
parser.add_argument('--timestamp', default='', type=str,
                    help='no need to change this')
parser.add_argument('--ckpt_folder', default='default', type=str,
                    help='the name of folder in experiments')

parser.add_argument('--rank_of_lora', default=8, type=int,
                    help='the rank of lora modules')
parser.add_argument('--zero_init', action='store_true',
                    help='initialize lora_B weight as zero')
parser.add_argument('--coop', action='store_true',
                    help='lora cooperate with encoder rather than adversary')
parser.add_argument('--lora_lr_adjust', action='store_true',
                    help='adjust lora lr together with base lr')
parser.add_argument('--valid_precent', default=0.1, type=float, help='how much of the data used to train KNN in trainset')
# parser.add_argument('--enable_lora_epoch', default=150, type=int,
                    # help='number of epochs before training with lora only and train with traditional simsiam')
parser.add_argument('--classifier', default='sgd', choices=["retrieval", "sgd", "logistic"], help='{"retrieval", "sgd", "logistic"}')
# moco specific configs:
parser.add_argument(
    "--dim", default=128, type=int, help="feature dimension (default: 128)"
)
parser.add_argument(
    "--temp", default=0.1, type=float, help="softmax temperature (default: 0.1)"
)
parser.add_argument(
    "--lora_scale", default=0.1, type=float, help="how large to the proportion could mat BA be, as compared with the original conv kernal"
)
parser.add_argument('--lora_type', default='none', type=str,
                    help='the way of regularization to lora')
parser.add_argument('--lr_sche', default='cos', choices=["cos", "none", "step", "cosend"], help='{"cos", "none", "step", "cosend"}')
parser.add_argument('--step_perc', default=0.5, type=float, help="for lr scheduler, how much does lr shrink to after each step")
parser.add_argument('--step_freq', default=100, type=int, help='how many epochs before each step of lr scheduler')

def main():
    args = parser.parse_args()
    
    while(True):
        timestamp = datetime.now()
        args.timestamp = "".join(str(timestamp.strftime("%D_%H%M%S")).split("/"))
        tgt = './experiments/'+args.ckpt_folder+"/"+str(args.timestamp)
        print(tgt)
        try:
            os.makedirs(tgt)
            break
        except FileExistsError:
            time.sleep(3)
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    # loraloss_pgbk.set_scale(args.lora_scale)
    loraloss_pgbk.init_pgbk(scale=args.lora_scale, type=args.lora_type)

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
    # create model
    print("=> creating model '{}'".format(args.arch))
    weights = args.weights
    if len(weights)==0:
        weights = None
    model = simclr.builder.SimCLR(
        models.__dict__[args.arch],
        args.dim,
        args.temp,
        weights=weights,
        args=args
    )

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256
    init_lora_lr = args.lora_lr * args.batch_size / 256

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    print(model) # print model after SyncBatchNorm

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # if args.enable_lora:
        # model.module.lora_init(args)

    optimizer = torch.optim.SGD(model.module.encoder_q.parameters(), init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    optimizer_lora = None
    if args.enable_lora:
        optimizer_lora = torch.optim.SGD(lora.separate_param(model=model.module.encoder_k)[1]
, init_lora_lr,
                            momentum=args.momentum,
                            weight_decay=args.lora_wd, maximize=not args.coop)
        if args.coop:
            print("WARNING: args.coop==True, now in coop mode")

    # optionally resume from a checkpoint
    
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)
            if (len(missing_keys) > 0) or (len(unexpected_keys) > 0):
                print(f'=> missing_keys={missing_keys}, unexpected_keys={unexpected_keys}')
            optimizer.load_state_dict(checkpoint['optimizer'])
            optimizer_lora.load_state_dict(checkpoint['optimizer_lora'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            raise NotImplementedError()

    cudnn.benchmark = True

    # Data loading code
    # traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_plain = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224), transforms.ToTensor(),normalize])
    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([simclr.loader.GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    traindir = args.data
    
    small_train_dataset = datasets.ImageFolder(traindir, transform_plain)
    small_train_sampler = torch.utils.data.distributed.DistributedSampler(small_train_dataset)
    small_train_loader = torch.utils.data.DataLoader(
        small_train_dataset, batch_size=256, shuffle=(small_train_sampler is None),
        num_workers=args.workers//2, pin_memory=True, sampler=small_train_sampler, drop_last=True)
    
    if args.enable_aug:
        train_dataset = datasets.ImageFolder(traindir, simclr.loader.TwoCropsTransform(transforms.Compose(augmentation)))
    else:
        train_dataset = small_train_dataset
        
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    valid_iid_dataset = MultipleDataset(args.iid_val_data, transform_plain)
    val_iid_ddp_sampler = torch.utils.data.distributed.DistributedSampler(valid_iid_dataset, shuffle=False)
    val_iid_loader = torch.utils.data.DataLoader(valid_iid_dataset, batch_size=args.batch_size, shuffle=False,
            sampler=val_iid_ddp_sampler, num_workers=args.workers//4, pin_memory=True)
    
    valid_ood_dataset = MultipleDataset(args.ood_val_data, transform_plain)
    val_ood_ddp_sampler = torch.utils.data.distributed.DistributedSampler(valid_ood_dataset, shuffle=False)
    val_ood_loader = torch.utils.data.DataLoader(valid_ood_dataset, batch_size=args.batch_size, shuffle=False,
            sampler=val_ood_ddp_sampler, num_workers=args.workers//4, pin_memory=True)

    # lora.lora_switch(model, use_lora=False)
    full_valid(small_train_loader, val_iid_loader, val_ood_loader, model.module.encoder_q, criterion, args.start_epoch, args)
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, init_lr, epoch, args)
        if args.enable_lora and args.lora_lr_adjust:
            print("adjusting lora learning rate...")
            adjust_learning_rate(optimizer_lora, init_lora_lr, epoch, args)
        
        # train for one epoch
        train(train_loader, model, criterion, optimizer, optimizer_lora, epoch, args)
        

        if epoch % args.save_freq == args.save_freq-1:
            if epoch == 199 or epoch == 249:
                full_valid(small_train_loader, val_iid_loader, val_ood_loader, model.module.encoder_q, criterion, epoch, args)
            else: 
                valid(small_train_loader, val_iid_loader, val_ood_loader, model.module.encoder_q, criterion, epoch, args)
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'optimizer_lora' : optimizer_lora.state_dict() if args.enable_lora else {},
                }, is_best=False, prefix ='./experiments/'+args.ckpt_folder+"/"+str(args.timestamp)+"/", filename='checkpoint_{:04d}.pth.tar'.format(epoch))

        elif epoch % args.valid_freq == args.valid_freq-1:
            # lora.lora_switch(model, use_lora=False)
            valid(small_train_loader, val_iid_loader, val_ood_loader, model.module.encoder_q, criterion, epoch, args)

def full_valid(train_loader, iid_loader, ood_loader, model, criterion, epoch, args):
    temp_vp = args.valid_precent
    temp_clsfr = args.classifier

    for clsfr in ["retrieval", "sgd"]:
        args.classifier = clsfr
        for perc in [0.01, 0.05, 0.1, 1.0]:
            args.valid_precent = perc
            valid(train_loader, iid_loader, ood_loader, model, criterion, epoch, args)
    args.valid_precent = temp_vp
    args.classifier = temp_clsfr

def valid(train_loader, iid_loader, ood_loader, model, criterion, epoch, args):
    model.eval()
    print("current used valid percentage:  " + str(args.valid_precent))
    num_train_samples = int(len(train_loader.dataset) * args.valid_precent)
    features_dim = 512 if args.arch == 'resnet18' else 2048
    device = next(model.parameters()).device
    is_root = dist.get_rank() == 0
    
    train_features = np.zeros((num_train_samples, features_dim), dtype=np.float16)
    train_labels = np.zeros(num_train_samples, dtype=np.int64)
    end_ind = 0
    for ind, (x, y) in tqdm(enumerate(train_loader)):
        with torch.no_grad():
            # features: output of layer4, before avgpool
            x = model.conv1(x.to(device))
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            features = model.layer4(x)
            features = torch.mean(features, dim=[-2, -1])
            features = nn.functional.normalize(features, dim=1)
            features = concat_all_gather_interlace(features)
            y = concat_all_gather_interlace(y.to(device))
            if not is_root:
                if (ind+1) * len(features) >= num_train_samples:
                    break
                continue
        y = y.cpu().numpy().astype(np.float16)
        features = features.cpu().numpy().astype(np.float16)
        begin_ind = end_ind
        end_ind = min(begin_ind + len(features), len(train_features))   # "drop_last=False", preventing shape issues, also for valid percent cutting
        train_features[begin_ind:end_ind, :] = features[:end_ind - begin_ind]
        train_labels[begin_ind:end_ind] = y[:end_ind - begin_ind]
        if end_ind >= num_train_samples:
            break

    if is_root:
        if args.classifier == 'sgd':
            cls = SGDClassifier(max_iter=1000, n_jobs=16, tol=1e-3).fit(train_features, train_labels)
        elif args.classifier == 'logistic':
            cls = LogisticRegression(max_iter=1000, n_jobs=16, tol=1e-3).fit(train_features, train_labels)
        elif args.classifier == 'retrieval':
            cls = NearestNeighbors(n_neighbors=min(20, train_features.shape[0]), algorithm='auto',
                                   n_jobs=-1, metric='correlation').fit(train_features)
        else:
            raise NotImplementedError()
    else:
        cls = None
        
    def _run_cls(dst_domain_pth, test_loader):
        print(f"Target domain {dst_domain_pth}")

        batch_time = AverageMeter('Time', ':6.5f')
        acc1 = AverageMeter('Acc@1', ':6.5f')
        acc10 = AverageMeter('Acc@10', ':6.5f')
        acc20 = AverageMeter('Acc@20', ':6.5f')
        precision1 = AverageMeter('p@1', ':6.5f')
        precision10 = AverageMeter('p@10', ':6.5f')
        precision20 = AverageMeter('p@20', ':6.5f')
        precision5 = AverageMeter('p@5', ':6.3f')
        precision15 = AverageMeter('p@15', ':6.3f')

        progress = ProgressMeter(
            len(test_loader),
            [batch_time, acc1, acc10, acc20, precision1, precision5, precision10, precision15, precision20],
            prefix=f"Train on {args.data} Test on {dst_domain_pth}")
        end = time.time()
        num_test_samples = len(test_loader.dataset)
        print(f'num_test_samples: {num_test_samples}')
        total_samples = 0
        all_features = []
        all_y = []
        for ind, (x, y) in enumerate(test_loader):
            with torch.no_grad():
                x = model.conv1(x.to(device))
                x = model.bn1(x)
                x = model.relu(x)
                x = model.maxpool(x)
                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                features = model.layer4(x)
                features = torch.mean(features, dim=[-2, -1])
                features = nn.functional.normalize(features, dim=1)
                features = concat_all_gather_interlace(features)
                y = concat_all_gather_interlace(y.to(device)).cpu()
                if not is_root:
                    continue
            features = features.cpu().numpy().astype(np.float16)
            y = y.numpy()
            total_samples += len(y)
            if total_samples > num_test_samples:
                diff = total_samples - num_test_samples
                features = features[:-diff]
                y = y[:-diff]
                all_features.append(features)
                all_y.append(y)
                print(f'diff {diff}')
            all_features.append(features)
            all_y.append(y)
            if args.classifier == 'retrieval':
                a1, a10, a20, p1, p5, p10, p15, p20 = ret_metrics(features, y, train_labels, cls)
            else:
                a1, a10, a20, p1, p5, p10, p15, p20 = cls_metrics(features, y, cls)
            acc1.update(a1, len(y))
            acc10.update(a10, len(y))
            acc20.update(a20, len(y))
            precision1.update(p1, len(y))
            precision10.update(p10, len(y))
            precision20.update(p20, len(y))
            precision5.update(p5, len(y))
            precision15.update(p15, len(y))
            batch_time.update(time.time() - end)
            end = time.time()

            if ind % 10 == 0:
                progress.display(ind)

        print('Final batch:')
        progress.display(len(test_loader))
        if args.gpu==0:
            print('Results:')
            print(','.join([
                            f'acc1={acc1.avg}',
                            f'acc10={acc10.avg}',
                            f'acc20={acc20.avg}',
                            f'precision1={precision1.avg}',
                            f'precision5={precision5.avg}',
                            f'precision10={precision10.avg}',
                            f'precision15={precision15.avg}',
                            f'precision20={precision20.avg}'
                            ]))
    _run_cls(args.iid_val_data, iid_loader)
    _run_cls(args.ood_val_data, ood_loader)


def train(train_loader, model, criterion, optimizer, optimizer_lora, epoch, args):
    if epoch < args.lora_start_epoch:
        args.enable_lora = False
    elif epoch == args.lora_start_epoch:
        args.enable_lora = True
    if epoch == args.lora_fix_epoch:
        loraloss_pgbk.disable()
        optimizer_lora = None
        args.enable_lora = False

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    lora_absolute = AverageMeter('LoRA Absolute', ':6.5f')
    lora_loss_meter = AverageMeter('LoRA Loss', ':6.5f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    contrast_step = ContrastTrainer(criterion, model, args, top1, top5)
    
    avg_meters = [batch_time, data_time, losses, top1, top5] if not args.enable_lora else [batch_time, data_time, losses, lora_absolute, lora_loss_meter, top1, top5]
    
    progress = ProgressMeter(
        len(train_loader),
        avg_meters,
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            if args.enable_aug:
                images[0] = images[0].cuda(args.gpu, non_blocking=True)
                images[1] = images[1].cuda(args.gpu, non_blocking=True)
            else:
                images = [images.cuda(args.gpu, non_blocking=True), None]
        
        loss = contrast_step.run(im_q=images[0], im_k=images[1])
        losses.update(loss.item(), images[0].size(0))
        
        if args.enable_lora:
            lora_loss = loraloss_pgbk.smash() 
            with torch.no_grad():
                lora_loss_meter.update(float(lora_loss))
                lora_abs = _collect_lora_grad(model)
                lora_absolute.update(float(lora_abs))
        
            # scaler = min(float(loss.detach()/(1e-7+lora_loss.detach())), 1)

        # compute gradient and do SGD step
            optimizer_lora.zero_grad()
            optimizer.zero_grad()
            loss -= lora_loss   # optimizer_lora.maximize==True, we need minimize lora_loss on lora parameter
            loss.backward()
            optimizer_lora.step()
            optimizer.step()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def ensure(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)

def save_checkpoint(state, is_best, prefix, filename='checkpoint.pth.tar'):
    ensure(prefix)
    torch.save(state, prefix+filename)
    if is_best:
        shutil.copyfile(filename, prefix +'model_best.pth.tar')

def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    if args.lr_sche == 'cos':
        cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    elif args.lr_sche == 'none':
        cur_lr = init_lr
    elif args.lr_sche == 'step':
        cur_lr = init_lr * (args.step_perc ** (epoch//args.step_freq))
    elif args.lr_sche == 'cosend':
        cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * (1000-args.epochs+epoch) / 1000))
    else:
        raise NotImplementedError()
    print("cur_lr: " + str(cur_lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


if __name__ == '__main__':
    main()