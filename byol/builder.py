# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import lib.loralib as lora


class BYOL(nn.Module):
    """
    Build a BYOL model.
    """

    def __init__(self, base_encoder, dim=128, hidden_size=512, m=0.996, weights=None, args=None):
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
        self.encoder_q = base_encoder(num_classes=1000, weights=weights)
        self.encoder_k = base_encoder(num_classes=1000, weights=weights)

        dim_mlp = self.encoder_q.fc.weight.shape[1]
        self.encoder_q.fc = nn.Sequential(
            nn.Linear(dim_mlp, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(inplace=True), nn.Linear(hidden_size, dim)
        )
        self.encoder_k.fc = nn.Sequential(
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
        if "1" in self.args.lora_layers:
            self.encoder_q.layer1 = lora.attach_lora(self.encoder_q.layer1, r=self.args.rank_of_lora, zero_init=self.args.zero_init)
            # self.encoder_k.layer1 = lora.attach_lora(self.encoder_k.layer1, r=self.args.rank_of_lora)
        if "2" in self.args.lora_layers:
            self.encoder_q.layer2 = lora.attach_lora(self.encoder_q.layer2, r=self.args.rank_of_lora, zero_init=self.args.zero_init)
            # self.encoder_k.layer2 = lora.attach_lora(self.encoder_k.layer2, r=self.args.rank_of_lora)
        if "3" in self.args.lora_layers:
            self.encoder_q.layer3 = lora.attach_lora(self.encoder_q.layer3, r=self.args.rank_of_lora, zero_init=self.args.zero_init)  
            # self.encoder_k.layer3 = lora.attach_lora(self.encoder_k.layer3, r=self.args.rank_of_lora)        
        if "4" in self.args.lora_layers:
            self.encoder_q.layer4 = lora.attach_lora(self.encoder_q.layer4, r=self.args.rank_of_lora, zero_init=self.args.zero_init)
            # self.encoder_k.layer4 = lora.attach_lora(self.encoder_k.layer4, r=self.args.rank_of_lora)
        
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
