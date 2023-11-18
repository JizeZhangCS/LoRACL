# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import lib.loralib as lora
import timm


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, dim=128, K=65536, m=0.999, T=0.2, mlp=True, weights=None, args=None):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.2)
        """
        super(MoCo, self).__init__()

        self.BIG_NUMBER = 100000
        self.K = K
        self.m = m
        self.T = T
        self.args = args
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = timm.create_model(weights, pretrained=not args.scratch)
        self.encoder_k = timm.create_model(weights, pretrained=not args.scratch)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.head.weight.shape[1]
            self.encoder_q.head = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.BatchNorm1d(dim_mlp), nn.ReLU(inplace=True), 
                nn.Linear(dim_mlp, dim_mlp), nn.BatchNorm1d(dim_mlp), nn.ReLU(inplace=True), 
                nn.Linear(dim_mlp, dim), nn.BatchNorm1d(dim, affine=False)
            )
            self.encoder_k.head = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.BatchNorm1d(dim_mlp), nn.ReLU(inplace=True), 
                nn.Linear(dim_mlp, dim_mlp), nn.BatchNorm1d(dim_mlp), nn.ReLU(inplace=True), 
                nn.Linear(dim_mlp, dim), nn.BatchNorm1d(dim, affine=False)
            )

        self._param_init()
        if args.enable_lora:
            self._lora_init()

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_idx_lst", - torch.ones(K))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        
    def _param_init(self):
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
    
    def _lora_init(self):
        for idx in range(len(self.encoder_q.blocks)):
            if "q" in self.args.weight_types or "k" in self.args.weight_types or "v" in self.args.weight_types:
                orig_fc = self.encoder_q.blocks[idx].attn.qkv
                self.encoder_q.blocks[idx].attn.qkv = lora.QKVLinear(orig_fc, r=self.args.rank_of_lora, zero_init=self.args.zero_init, weight_types=self.args.weight_types).to(orig_fc.weight.device)

            if "o" in self.args.weight_types:
                orig_fc = self.encoder_q.blocks[idx].attn.proj
                self.encoder_q.blocks[idx].attn.proj = lora.Linear(orig_fc, r=self.args.rank_of_lora, zero_init=self.args.zero_init).to(orig_fc.weight.device)
            
            if "m" in self.args.weight_types:
                orig_fc = self.encoder_q.blocks[idx].mlp.fc1
                self.encoder_q.blocks[idx].mlp.fc1 = lora.Linear(orig_fc, r=self.args.rank_of_lora, zero_init=self.args.zero_init).to(orig_fc.weight.device)
                orig_fc = self.encoder_q.blocks[idx].mlp.fc2
                self.encoder_q.blocks[idx].mlp.fc2 = lora.Linear(orig_fc, r=self.args.rank_of_lora, zero_init=self.args.zero_init).to(orig_fc.weight.device)
        
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        q_param_list = lora.separate_param(self.encoder_q)[0]
        for param_k in self.encoder_k.parameters():
            param_k.data = param_k.data * self.m + q_param_list[0].data * (1.0 - self.m)
            q_param_list = q_param_list[1:]

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, sample_idx):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        sample_idx = concat_all_gather(sample_idx)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        self.queue_idx_lst[ptr : ptr + batch_size] = sample_idx
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

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
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
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
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, sample_idx=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)
        
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        if sample_idx != None:
            batch_idx_mat = sample_idx.view(len(sample_idx), 1).repeat(1, self.K)
            bank_idx_mat = self.queue_idx_lst.view(1, self.K).repeat(len(sample_idx), 1)
            mask = (batch_idx_mat==bank_idx_mat) * -1 * self.BIG_NUMBER
            l_neg += mask

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, sample_idx)

        return logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
