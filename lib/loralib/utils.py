#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn

from typing import Dict

import lib.loralib as lora
from .layers import LoRALayer


__no_lora_list__ = ["activation", "pooling", "batchnorm"]


def lora_switch(model, use_lora=True):
    if model.__module__.split('.')[-2] == "loralib":
        model.use_lora = use_lora
    elif model.__module__.split('.')[-1] == "container":
        assert type(model)==nn.modules.container.Sequential, str(type(model)) + " is invalid, only Sequential is supported!"
        for submodel in model:
            lora_switch(submodel, use_lora=use_lora)
    else:
        for name, _ in model.named_children():
            lora_switch(getattr(model, name), use_lora=use_lora)
    # raise NotImplementedError(model.__module__ + " is not implemented yet!")

def attach_lora(model, r, lora_type='full', zero_init=True):

    if model.__module__.split('.')[-1] in __no_lora_list__:
        return model
    elif model.__module__.split('.')[-1] == "linear":
        if lora_type in ["linear", "full"]:
            model = lora.Linear(model, r=r, zero_init=zero_init).to(model.weight.device)
        return model
    elif model.__module__.split('.')[-1] == "conv":
        assert type(model)==nn.modules.conv.Conv2d, str(type(model)) + " is invalid, only conv2d is supported!"
        if lora_type in ["conv", "full"]:
            model = lora.Conv2d(model, r=r, zero_init=zero_init).to(model.weight.device)
        return model
    elif model.__module__.split('.')[-1] == "container":
        assert type(model)==nn.modules.container.Sequential, str(type(model)) + " is invalid, only Sequential is supported!"
        lora_seq = []
        for submodel in model:
            lora_seq.append(attach_lora(submodel, r, lora_type, zero_init=zero_init))
        return nn.Sequential(*lora_seq)
    else:
        for name, _ in model.named_children():
            new_child = attach_lora(getattr(model, name), r, lora_type, zero_init=zero_init)
            setattr(model, name, new_child)
        return model 
        # raise NotImplementedError(model.__module__ + " is not implemented yet!")

def freeze_param(model: nn.Module, freeze_lora=None, freeze_base=True, moco: bool = True) -> None:
    for n, p in model.named_parameters():
        if 'encoder_q_lora' in n and moco:
            p.requires_grad = False
            continue
        if 'lora_' not in n:        # base param
            if freeze_base != None:    
                p.requires_grad = not freeze_base
        else: 
            if freeze_lora != None:
                p.requires_grad = not freeze_lora

def separate_param(model: nn.Module, include_name=False):
    lora_param = []
    base_param = []
    if include_name:
        for n, p in model.named_parameters():
            if 'lora_' not in n:    # base param
                base_param.append((n,p))
            else:
                lora_param.append((n,p))
    else:
        for n, p in model.named_parameters():
            if 'lora_' not in n:    # base param
                base_param.append(p)
            else:
                lora_param.append(p)

    return base_param, lora_param

def state_dict(model: nn.Module, save_base: bool = 'True') -> Dict[str, torch.Tensor]:
    """Return the state dict of a lora model, containing EITHER base model param OR lora param

    Args:
        model (nn.Module): lora model
        save_base (bool, optional): True to save base model, False to save lora. Defaults to 'True'.

    Returns:
        Dict[str, torch.Tensor]: state dict of base model OR lora model
    """
    my_state_dict = model.state_dict()
    if save_base:
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' not in k}
    return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}

# def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
#     for n, p in model.named_parameters():
#         if 'lora_' not in n:
#             p.requires_grad = False
#     if bias == 'none':
#         return
#     elif bias == 'all':
#         for n, p in model.named_parameters():
#             if 'bias' in n:
#                 p.requires_grad = True
#     elif bias == 'lora_only':
#         for m in model.modules():
#             if isinstance(m, LoRALayer) and \
#                 hasattr(m, 'bias') and \
#                 m.bias is not None:
#                     m.bias.requires_grad = True
#     else:
#         raise NotImplementedError




# def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
#     my_state_dict = model.state_dict()
#     if bias == 'none':
#         return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
#     elif bias == 'all':
#         return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
#     elif bias == 'lora_only':
#         to_return = {}
#         for k in my_state_dict:
#             if 'lora_' in k:
#                 to_return[k] = my_state_dict[k]
#                 bias_name = k.split('lora_')[0]+'bias'
#                 if bias_name in my_state_dict:
#                     to_return[bias_name] = my_state_dict[bias_name]
#         return to_return
#     else:
#         raise NotImplementedError
