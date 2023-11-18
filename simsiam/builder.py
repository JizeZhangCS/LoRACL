# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import lib.loralib as lora


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512, weights=None, id_predictor=False):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        # self.encoder = base_encoder(num_classes=dim, zero_init_residual=True, weights=weights)
        self.encoder = base_encoder(num_classes=1000, zero_init_residual=True, weights=weights) # 1000 just for loading

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
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
        

    def lora_init(self, args):
        if "1" in args.lora_layers:
            self.encoder.layer1 = lora.attach_lora(self.encoder.layer1, r=args.rank_of_lora, zero_init=args.zero_init)
        if "2" in args.lora_layers:
            self.encoder.layer2 = lora.attach_lora(self.encoder.layer2, r=args.rank_of_lora, zero_init=args.zero_init)
        if "3" in args.lora_layers:
            self.encoder.layer3 = lora.attach_lora(self.encoder.layer3, r=args.rank_of_lora, zero_init=args.zero_init)        
        if "4" in args.lora_layers:
            self.encoder.layer4 = lora.attach_lora(self.encoder.layer4, r=args.rank_of_lora, zero_init=args.zero_init)

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
