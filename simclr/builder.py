import itertools
import torch
import torch.nn as nn
import torch.nn.functional as func
import os
import glob
import numpy as np
import torch.distributed as dist
import lib.loralib as lora



class SimCLR(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, T=0.07, weights=None, args=None):
        """
        dim: feature dimension (default: 128)
        T: softmax temperature (default: 0.07)
        """
        super(SimCLR, self).__init__()

        self.args = args

        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=1000, weights=weights)
        self.encoder_k = base_encoder(num_classes=1000, weights=weights)

        dim_mlp = self.encoder_q.fc.weight.shape[1]
        self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp, bias=False), nn.BatchNorm1d(dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim))
        self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp, bias=False), nn.BatchNorm1d(dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim))

        # prev_dim = self.encoder_q.fc.weight.shape[1]
        # self.encoder_q.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
        #                         nn.BatchNorm1d(prev_dim),
        #                         nn.ReLU(inplace=True), # first layer
        #                         nn.Linear(prev_dim, prev_dim, bias=False),
        #                         nn.BatchNorm1d(prev_dim),
        #                         nn.ReLU(inplace=True), # second layer
        #                         # self.encoder.fc,
        #                         nn.Linear(prev_dim, dim, bias=False),
        #                         nn.BatchNorm1d(dim, affine=False)) # output layer
        # self.encoder_k.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
        #                 nn.BatchNorm1d(prev_dim),
        #                 nn.ReLU(inplace=True), # first layer
        #                 nn.Linear(prev_dim, prev_dim, bias=False),
        #                 nn.BatchNorm1d(prev_dim),
        #                 nn.ReLU(inplace=True), # second layer
        #                 # self.encoder.fc,
        #                 nn.Linear(prev_dim, dim, bias=False),
        #                 nn.BatchNorm1d(dim, affine=False)) # output layer
        
        self._param_init()
        if args.enable_lora:
            self._lora_init()

    def _param_init(self):
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
    

    def _lora_init(self):
        if "1" in self.args.lora_layers:
            self.encoder_k.layer1 = lora.attach_lora(self.encoder_k.layer1, r=self.args.rank_of_lora, zero_init=self.args.zero_init)
        if "2" in self.args.lora_layers:
            self.encoder_k.layer2 = lora.attach_lora(self.encoder_k.layer2, r=self.args.rank_of_lora, zero_init=self.args.zero_init)
        if "3" in self.args.lora_layers:
            self.encoder_k.layer3 = lora.attach_lora(self.encoder_k.layer3, r=self.args.rank_of_lora, zero_init=self.args.zero_init)        
        if "4" in self.args.lora_layers:
            self.encoder_k.layer4 = lora.attach_lora(self.encoder_k.layer4, r=self.args.rank_of_lora, zero_init=self.args.zero_init)

    @torch.no_grad()
    def _lora_encoder_update(self):
        """
        Momentum update of the base encoder attached with lora
        TODO: debug: check if name really the same!
        ATTENTION THAT BN RUNNING STAT ARE NOT CHANGED IN THIS WAY!!!
        """
        param_list = []
        for param_q in self.encoder_q.parameters():
            param_list.append(param_q)
        for name, param_k in self.encoder_k.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                continue
            else:
                param_k.data = param_list[0].data
                param_list = param_list[1:]

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        dist.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)  # also == batch_size_all - 1 - idx_shuffle

        # shuffled index for this gpu
        gpu_idx = dist.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = dist.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, img1, img2=None, use_lora=True):
        """
        Input:
            if img2 = None, img2 = img1
        Output:
            logits, targets
        """

        self._lora_encoder_update()
        model1 = self.encoder_q
        model2 = self.encoder_k
        if img2 == None:
            img2 = img1 

        # compute query features
        # img1.shape = 96, 3, 224, 224
        q = model1(img1)  # queries: NxC
        k = model2(img2)
        ft = torch.cat([q,k], dim=0)    # (2N, dim)
        ft = nn.functional.normalize(ft, dim=1)   
        sim_mat = torch.matmul(ft, ft.T)

        labels = torch.cat([torch.arange(q.shape[0]) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(sim_mat.device)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(sim_mat.device)
        labels = labels[~mask].view(labels.shape[0], -1)    # (2N, 2N-1), remove diagonal

        sim_mat = sim_mat[~mask].view(sim_mat.shape[0], -1)
        l_pos = sim_mat[labels.bool()].view(sim_mat.shape[0], -1)    # (2N, 1)
        l_neg = sim_mat[~labels.bool()].view(sim_mat.shape[0], -1)   # (2N, 2N-1)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits = logits / self.T

        # labels: positive key indicators, always being zero
        # "0" means that the logits[i][0] should be the largest (with highest prob) along logits[i]
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(sim_mat.device)
        return logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output
