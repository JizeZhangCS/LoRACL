#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F

import math
from lib.loralib.singleton_lora_piggybank import loraloss_pgbk

class LoRALayer():
    def __init__(self, r, lora_alpha, lora_dropout):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
            

class Linear(nn.Module, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(self, linear_module, r, lora_alpha=1, lora_dropout=0., zero_init=True):
        super(Linear, self).__init__()
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

        self.fc = linear_module
        self.use_lora = True
        out_features, in_features = self.fc.weight.shape

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.fc.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.fc.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            # self.fc.weight.requires_grad = False
        self.reset_parameters_lora(zero_init=zero_init)

    def reset_parameters_lora(self, zero_init=True):
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            if zero_init:
                nn.init.zeros_(self.lora_B)
            else:
                nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))

    def forward(self, x):
        result = F.linear(x, self.fc.weight, bias=self.fc.bias)
        if self.r > 0 and self.use_lora:
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        return result
    
    def __repr__(self):
        return "lora&" + super().__repr__() 

class QKVLinear(nn.Module, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(self, linear_module, r, lora_alpha=1, lora_dropout=0., zero_init=True, weight_types="qv"):
        super(QKVLinear, self).__init__()
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

        self.fc = linear_module
        self.use_lora = True
        self.weight_types = weight_types
        out_features, in_features = self.fc.weight.shape
        
        assert out_features % 3 == 0
        self.b_dim = out_features//3

        # Actual trainable parameters
        if r > 0:
            for type in self.weight_types:
                if type in "qkv":
                    self.register_parameter("lora_A_" + type, nn.Parameter(self.fc.weight.new_zeros((r, in_features))))
                    self.register_parameter("lora_B_" + type, nn.Parameter(self.fc.weight.new_zeros((self.b_dim, r))))
                elif type in "om":
                    continue
                else:
                    raise NotImplementedError("weight type could be only among /'qkvo/', but detected " + type)
            self.scaling = self.lora_alpha / self.r
        self.reset_parameters_lora(zero_init=zero_init)

    def reset_parameters_lora(self, zero_init=True):
        for n, p in self.named_parameters():
            if "lora_A_" in n:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            elif "lora_B_" in n:
                if zero_init:
                    nn.init.zeros_(p)
                else:
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
    
    def forward(self, x):
        result = F.linear(x, self.fc.weight, bias=self.fc.bias)
        if self.r > 0 and self.use_lora:
            if "q" in self.weight_types:
                result[:, :, :self.b_dim] += (self.lora_dropout(x) @ self.lora_A_q.transpose(0, 1) @ self.lora_B_q.transpose(0, 1)) * self.scaling
            if "k" in self.weight_types:
                result[:, :, self.b_dim:2*self.b_dim] += (self.lora_dropout(x) @ self.lora_A_k.transpose(0, 1) @ self.lora_B_k.transpose(0, 1)) * self.scaling
            if "v" in self.weight_types:
                result[:, :, 2*self.b_dim:] += (self.lora_dropout(x) @ self.lora_A_v.transpose(0, 1) @ self.lora_B_v.transpose(0, 1)) * self.scaling
            
        return result
    
    def __repr__(self):
        return self.weight_types + "lora&" + super().__repr__() 
    
# class Conv2d(nn.Module, LoRALayer):
#     def __init__(self, conv_module, r, lora_alpha=1, lora_dropout=0.):
#         super(Conv2d, self).__init__()
#         LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

#         self.conv = conv_module
#         self.use_lora = True
#         out_channels, in_channels, kernel_size, _ = self.conv.weight.shape
#         assert isinstance(kernel_size, int)
#         assert kernel_size == _

#         # Actual trainable parameters
#         if r > 0:
#             self.lora_A = nn.Conv2d(in_channels, r, kernel_size, self.conv.stride, self.conv.padding, bias=False)
#             self.lora_B = nn.Conv2d(r, out_channels, (1, 1), bias=False)
#             self.scaling = self.lora_alpha / self.r
#             # Freezing the pre-trained weight matrix
#             # self.conv.weight.requires_grad = False
#         self.reset_parameters()

#     def reset_parameters(self):
#         if hasattr(self, 'lora_A'):
#             # initialize A the same way as the default for nn.Linear and B to zero
#             nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
#             # nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
#             nn.init.zeros_(self.lora_B.weight)

#     def forward(self, x):
#         original = self.conv(x)
#         if self.r > 0 and self.use_lora:
#             mid = self.lora_A(self.lora_dropout(x))
#             return original + self.lora_B(mid) * self.scaling
#         return original
    
#     def __repr__(self):
#         return "lora&" + super().__repr__()

class Conv2d(nn.Module, LoRALayer):
    def __init__(self, conv_module, r, lora_alpha=1, lora_dropout=0., zero_init=True):
        super(Conv2d, self).__init__()
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

        self.conv = conv_module
        self.use_lora = True
        out_channels, in_channels, kernel_size, _ = self.conv.weight.shape
        assert isinstance(kernel_size, int)
        assert kernel_size == _

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
              self.conv.weight.new_zeros((out_channels//self.conv.groups*kernel_size, r*kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            # self.conv.weight.requires_grad = False
        self.reset_parameters(zero_init=zero_init)

    def reset_parameters(self, zero_init=True):
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            if zero_init:
                nn.init.zeros_(self.lora_B)
            else:
                nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
            # nn.init.normal_(self.lora_A, mean=0.0, std=0.2)
            # nn.init.normal_(self.lora_B, mean=0.0, std=0.1)

    def forward(self, x):
        if self.r > 0 and self.use_lora:
            lora_kernel = (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
            loraloss_pgbk.accumulate_loss(self.conv.weight, lora_kernel)
            return self.conv._conv_forward(
                x, 
                self.conv.weight + lora_kernel,
                self.conv.bias
            )
        return self.conv(x)
    
    def __repr__(self):
        return "lora&" + super().__repr__()
