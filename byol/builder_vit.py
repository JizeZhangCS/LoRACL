# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import lib.loralib as lora
import timm


class BYOL(nn.Module):
    """
    Build a BYOL model.
    """

    def __init__(self, dim=128, hidden_size=512, m=0.996, weights=None, args=None):
        """
        dim: feature dimension (default: 128)
        m: byol momentum of updating key encoder (default: 0.996)
        T: softmax temperature (default: 0.2)
        """
        super(BYOL, self).__init__()

        self.m = m
        self.args = args
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = timm.create_model(weights, pretrained=not args.scratch)
        self.encoder_k = timm.create_model(weights, pretrained=not args.scratch)

        dim_mlp = self.encoder_q.head.weight.shape[1]
        self.encoder_q.head = nn.Sequential(
            nn.Linear(dim_mlp, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(inplace=True), nn.Linear(hidden_size, dim)
        )
        self.encoder_k.head = nn.Sequential(
            nn.Linear(dim_mlp, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(inplace=True), nn.Linear(hidden_size, dim)
        )

        self.predictor = nn.Sequential(nn.Linear(dim, hidden_size),
                                        nn.BatchNorm1d(hidden_size),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(hidden_size, dim)) # output layer

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

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        p1 = self.predictor(self.encoder_q(x1)) # NxC
        p2 = self.predictor(self.encoder_q(x2)) # NxC

        m1 = self.encoder_k(x1)
        m2 = self.encoder_k(x2)

        return p1, p2, m1.detach(), m2.detach()
