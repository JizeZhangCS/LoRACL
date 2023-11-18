# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import lib.loralib as lora
import timm


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, dim=2048, pred_dim=512, weights=None, id_predictor=False, args=None):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()
        self.args = args
        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        # self.encoder = base_encoder(num_classes=dim, zero_init_residual=True, weights=weights)
        self.encoder = timm.create_model(weights, pretrained=not args.scratch)

        # build a 3-layer projector
        prev_dim = self.encoder.head.weight.shape[1]
        self.encoder.head = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        # self.encoder.fc,
                                        nn.Linear(prev_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        # self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        
        if not id_predictor:
            self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                            nn.BatchNorm1d(pred_dim),
                                            nn.ReLU(inplace=True), # hidden layer
                                            nn.Linear(pred_dim, dim)) # output layer
        else:
            self.predictor = nn.Identity()
        

    def lora_init(self):
        for idx in range(len(self.encoder.blocks)):
            if "q" in self.args.weight_types or "k" in self.args.weight_types or "v" in self.args.weight_types:
                orig_fc = self.encoder.blocks[idx].attn.qkv
                self.encoder.blocks[idx].attn.qkv = lora.QKVLinear(orig_fc, r=self.args.rank_of_lora, zero_init=self.args.zero_init, weight_types=self.args.weight_types).to(orig_fc.weight.device)

            if "o" in self.args.weight_types:
                orig_fc = self.encoder.blocks[idx].attn.proj
                self.encoder.blocks[idx].attn.proj = lora.Linear(orig_fc, r=self.args.rank_of_lora, zero_init=self.args.zero_init).to(orig_fc.weight.device)
            
            if "m" in self.args.weight_types:
                orig_fc = self.encoder.blocks[idx].mlp.fc1
                self.encoder.blocks[idx].mlp.fc1 = lora.Linear(orig_fc, r=self.args.rank_of_lora, zero_init=self.args.zero_init).to(orig_fc.weight.device)
                orig_fc = self.encoder.blocks[idx].mlp.fc2
                self.encoder.blocks[idx].mlp.fc2 = lora.Linear(orig_fc, r=self.args.rank_of_lora, zero_init=self.args.zero_init).to(orig_fc.weight.device)

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        if x2 != None:
            # compute features for one view
            z1 = self.encoder(x1) # NxC
            z2 = self.encoder(x2) # NxC

            p1 = self.predictor(z1) # NxC
            p2 = self.predictor(z2) # NxC

            return p1, p2, z1.detach(), z2.detach()

        x2 = x1
        z_theta = self.encoder(x1)  # queries: NxC
        p_theta = self.predictor(z_theta)

        lora.lora_switch(self.encoder, use_lora=False)
        z = self.encoder(x2)
        lora.lora_switch(self.encoder, use_lora=True)

        return p_theta, None, None, z.detach()
