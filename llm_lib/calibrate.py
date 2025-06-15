import torch
import torch.nn as nn
from functools import partial
from tqdm import tqdm
import pickle
import os
import types
import math
from typing import Optional, Tuple
from transformers.models.phi3.modeling_phi3 import Phi3Attention, Phi3MLP, apply_rotary_pos_emb, logger, repeat_kv
from transformers.cache_utils import Cache

@torch.no_grad()
def calc_act_scales(model, tokenizer, dataset, seq_len=512, num_samples=256, mode="max"):
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}
    cnt = {}

    def stat_tensor(name, tensor, per_channel=True):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        if mode == "max":
            comming_max = torch.max(tensor, dim=0)[0].float()
            if not per_channel: comming_max = comming_max.max()
            if name in act_scales:
                act_scales[name] = torch.max(act_scales[name], comming_max)
            else:
                act_scales[name] = comming_max
        elif mode == "topk":
            k = 15
            comming_max = tensor.topk(min(k, tensor.shape[0]), dim=0)[0].float()
            if per_channel:
                if name in act_scales:
                    act_scales[name] = torch.concat([act_scales[name], comming_max], dim=0).topk(k, 0)[0]
                else:
                    act_scales[name] = comming_max
            else:
                comming_max = comming_max.max()
                if name in act_scales:
                    act_scales[name] = torch.max(act_scales[name], comming_max)
                else:
                    act_scales[name] = comming_max
        elif mode == "l1":
            comming_mean = torch.mean(tensor.abs(), dim=0).float()
            if not per_channel: comming_mean = comming_mean.mean()
            if name in act_scales:
                act_scales[name] = (act_scales[name] * cnt[name] + comming_mean) / (cnt[name] + 1)
                cnt[name] += 1
            else:
                act_scales[name] = comming_mean
                cnt[name] = 1
        elif mode == "l2":
            comming_l2 = tensor.abs().double().pow(2).mean(dim=0).sqrt().float()
            if not per_channel: comming_l2 = comming_l2.mean()
            if name in act_scales:
                act_scales[name] = (act_scales[name].double().pow(2)*cnt[name]/(cnt[name]+1) + comming_l2.double().pow(2)/(cnt[name]+1)).sqrt().float()
                cnt[name] += 1
            else:
                act_scales[name] = comming_l2
                cnt[name] = 1
        elif mode == "sim":
            comming_sim = tensor.transpose(-1, -2) @ tensor
            comming_l2 = tensor.abs().double().pow(2).mean(dim=0).sqrt().float()
            if not per_channel: comming_l2 = comming_l2.mean()
            if name in act_scales:
                act_scales[name] += comming_sim
                act_scales[name + "_norm"] = (act_scales[name + "_norm"].double().pow(2)*cnt[name]/(cnt[name]+1) + comming_l2.double().pow(2)/(cnt[name]+1)).sqrt().float()
                cnt[name] += 1
            else:
                act_scales[name] = comming_sim
                act_scales[name + "_norm"] = comming_l2
                cnt[name] = 1

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)
        stat_tensor(name + "_output", y)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear,)):
            hooks.append(
                m.register_forward_hook(partial(stat_input_hook, name=name))
            )

    if hasattr(dataset, 'input_ids'):
        inputs = [dataset.input_ids[:,i:i+seq_len] for i in range(0,dataset.input_ids.shape[1],seq_len)]
    else:
        inputs = []
        for data in dataset:
            if len(data["text"]) > 0:
            # if len(data["text"]) > 50:
                input_ids = tokenizer(
                    data["text"], return_tensors="pt", max_length=seq_len, truncation=True
                ).input_ids
                inputs.append(input_ids)

    inputs = inputs[:num_samples]
    bar = tqdm(total=len(inputs))
    for input_ids in inputs:
        model(input_ids.to(device))
        bar.update(1)

    for h in hooks:
        h.remove()

    if mode == "topk":
        for name in act_scales:
            #act_scales[name] = act_scales[name].min(dim=0)[0]
            act_scales[name] = act_scales[name].mean(dim=0)

    return act_scales

def get_act_scales(model, tokenizer, dataset, prefix, seq_len=512, num_samples=256, mode="max"):
    dir = os.path.dirname(__file__) + "/tmp"
    os.makedirs(dir, exist_ok=True)
    if prefix is None:
        act_scales = calc_act_scales(model, tokenizer, dataset, seq_len, num_samples=num_samples, mode=mode)
    else:
        path = f'{dir}/{prefix}_act_scales.pickle'
        print(path)
        if not os.path.exists(path):
            act_scales = calc_act_scales(model, tokenizer, dataset, seq_len, num_samples=num_samples, mode=mode)
            with open(path, 'wb') as f:
                pickle.dump(act_scales, f)
        else:
            with open(path, 'rb') as f:
                act_scales = pickle.load(f)
    
    device = next(model.parameters()).device
    for k in act_scales:
        act_scales[k] = act_scales[k].to(device)
    return act_scales
