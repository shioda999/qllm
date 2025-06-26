import torch
import torch.nn as nn
from .utils import calc_init_scale
import re
from functools import partial
from tqdm import tqdm
import math
import types
from typing import Optional, Tuple
from transformers.models.phi3.modeling_phi3 import Phi3Attention, Phi3MLP, apply_rotary_pos_emb, logger, repeat_kv
from transformers.cache_utils import Cache
from .squant import squant_flip, squant_flip_aware_act_scale
from .gptq import GPTQ
from .marlin_linear import MarlinLinear
import torch
import psutil
import marlin

# import torch.nn.quantized.functional as qF
# import torch.nn.quantized as nnq

def print_memory_usage():
    """CPUとGPUのメモリ使用量を表示"""
    # CPUメモリの使用量
    cpu_memory = psutil.virtual_memory()
    print(f"*******************\nCPU Memory Usage: {cpu_memory.used / (1024**3):.2f} GB / {cpu_memory.total / (1024**3):.2f} GB")

    # GPUメモリの使用量（CUDAが利用可能な場合）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
        gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"GPU Memory Allocated: {gpu_memory_allocated:.2f} GB")
        print(f"GPU Memory Reserved: {gpu_memory_reserved:.2f} GB")
    else:
        print("CUDA is not available.")
    print('*******************')


def round_ste(x):
    return (x.round() - x).detach() + x

def clamp_ste(x, min, max):
    return (x.clamp(min=min, max=max) - x).detach() + x

class Quantizer(nn.Module):
    def __init__(self, device, scale=1., zero_point=torch.tensor(0)):
        super(Quantizer, self).__init__()
        if type(scale) is not torch.Tensor: scale = torch.tensor(scale)
        self.scale = nn.Parameter(scale.clone().to(device), False)
        self.zero_point = nn.Parameter(zero_point.clone().to(device), False)
        self.Qp, self.Qn = 127, -128

    def forward(self, x):
        return torch.quantize_per_tensor(x.float(), self.scale.data, self.zero_point.data, dtype=torch.qint8)
        # return x.div(self.scale).clamp_(self.Qn, self.Qp).round_().mul_(self.scale)
        # y = (x.float() / self.scale).clamp(self.Qn, self.Qp).round() * self.scale
        # y = round_ste(x.float().div(self.scale)).clamp(self.Qn, self.Qp).mul(self.scale).to(x.dtype)
        # return y
    
class DynamicQuantizer(nn.Module):
    def forward(self, x):
        scale = x.abs().max().div_(127).clamp(min=1e-5)
        return (x.div_(scale)).clamp_(-128, 127).round_().mul_(scale)
    
class DynamicPerTokenQuantizer(nn.Module):
    def forward(self, x):
        Qp = 127
        scale = x.abs().max(dim=-1, keepdim=True)[0].div_(Qp).clamp(min=1e-5)
        return (x.div_(scale)).clamp_(-Qp-1, Qp).round_().mul_(scale)
    
def make_quantizer(device, scale):
    if scale == "none": return nn.Identity()
    elif scale == "dynamic": return DynamicQuantizer()
    elif scale == "dynamic_per_token": return DynamicPerTokenQuantizer()
    else: return Quantizer(device, scale)
    
class QLinear(nn.Linear):
    def add_quantizer(self, device, in_scale, w_scale, o_scale=None, name=None):
        # self.in_q = make_quantizer(device, in_scale)
        # self.w_q = make_quantizer(device, w_scale)
        # self.o_q = make_quantizer(device, o_scale)
        self.name = name
        # qw = self.w_q(self.weight)

        self.qlinear = MarlinLinear(self.weight.T)
        # groupsize = 128
        # maxq = 2 ** 4 - 1
        # w = self.weight
        # if groupsize != -1:
        #     w = w.reshape((-1, groupsize, w.shape[-1]))
        #     w = w.permute(1, 0, 2)
        #     w = w.reshape((groupsize, -1))
        # s = torch.max(torch.abs(w), 0, keepdim=True)[0]
        # s *= 2 / maxq
        # s = s.reshape((-1, w.shape[1])).contiguous()
        # self.qlinear = marlin.Layer(self.weight.shape[0], self.weight.shape[1], groupsize=groupsize)
        # self.qlinear.pack(self, s)
        # del self.weight

    def forward(self, x):
        # if self.weight.shape[0] > 30000:
          # return torch.nn.functional.linear(x, self.weight)
        # x = self.in_q(x)#.float())
        # if not hasattr(self, "qlinear"):
          # self.qlinear = MarlinLinear(product(x.shape[1:-1]), self.weight.T)
        o = self.qlinear(x)
        # o = torch.concat([self.qlinear(x[:,i:i+1]) for i in range(x.shape[1])], dim=1)
        # o = self.o_q(o).dequantize()
        # if hasattr(self, 'm'): o.mul_(self.m)
        # print(self.name, x.shape, w.shape, o.shape)
        # input()
        return o
    
def get_depth(name):
    m = re.match(r'model\.layers\.(\d+)\..*', name)
    if m is not None:
        return int(m.groups()[0])

def quantize_model(model, act_scales={}, mode="static", down_proj_in_scale_mul=0.5, name_prefix="", fp32=False):
    device = next(model.parameters()).device
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and m.weight.shape[0] < 30000:
            #print(name, get_depth(name))
            m.__class__ = QLinear
            m.name = name

            if fp32:
                in_scale = "none"
                w_scale = "none"
                o_scale = "none"
            
            else:
                scale = 1.0 # act_scales[name_prefix + name]
                o_scale = 1.0 # act_scales[name_prefix + name + "_output"]
                in_scale = 1.0 # calc_init_scale(scale, 127, method="minmax")
                w_scale = calc_init_scale(m.weight, 127, method="minmax")
                o_scale = 1.0 # calc_init_scale(o_scale, 127, method="minmax")

                if name.endswith('down_proj'):
                    in_scale *= down_proj_in_scale_mul
                    o_scale *= 0.5
                    w_scale *= 0.5

                if name.endswith('gate_proj'):
                    o_scale *= 0.5
                    w_scale *= 0.5

                if name.endswith('up_proj'):
                    o_scale *= 0.5
                    w_scale *= 0.5
                
                if mode == "down_proj_none":
                    if name.endswith('down_proj'):
                        o_scale = "none"
                elif mode == "out_none":
                    o_scale = "none"
                elif mode == "none":
                    in_scale = "none"
                    w_scale = "none"
                    o_scale = "none"

            m.add_quantizer(device, in_scale,  w_scale, o_scale, name=name)
            #m.add_quantizer(device, "dynamic_per_token",  w_scale, "dynamic_per_token")
            # m.__class__ = QLinearv2
            # o_scale = o_scale / in_scale
            # m.add_quantizer(device, "dynamic_per_token",  w_scale, o_scale)

def get_inputs(model, layer_name, tokenizer, dataset, seq_len=512, n_samples=None):
    model.eval()
    device = next(model.parameters()).device
    inputs = {}

    def store_tensor(name, tensor):
        if name not in inputs:
            inputs[name] = []
        inputs[name].append(tensor)

    def store_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        store_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if name == layer_name:
            hooks.append(
                m.register_forward_hook(partial(store_input_hook, name=name))
            )

    if hasattr(dataset, 'input_ids'):
        calibrate_inputs = [dataset.input_ids[:,i:i+seq_len] for i in range(0,dataset.input_ids.shape[1],seq_len)]
    else:
        calibrate_inputs = []
        for data in dataset:
            if len(data["text"]) > 50:
                input_ids = tokenizer(
                    data["text"], return_tensors="pt", max_length=seq_len, truncation=True
                ).input_ids
                calibrate_inputs.append(input_ids)

    with torch.no_grad():
        if n_samples is not None: calibrate_inputs = calibrate_inputs[:n_samples]
        bar = tqdm(total=len(calibrate_inputs))
        for input_ids in calibrate_inputs:
            model(input_ids.to(device))
            bar.update(1)

    for h in hooks:
        h.remove()

    return inputs

def optimize_act_scales(model, tokenizer, act_scales, dataset, seq_len=512):
    
    def quantize(tensor, scale):
        return round_ste(tensor / scale).clamp(-128, 127) * scale

    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and name.endswith('down_proj'):
            inputs = get_inputs(model, name, tokenizer, dataset, seq_len, n_samples=30)
            scale = nn.Parameter(m.in_q.scale.data, True)
            optim = torch.optim.Adam([scale], 0.001)
            attempts = 100
            depth = get_depth(name)

            for name in inputs:
                inputs[name] = torch.cat(inputs[name], 1).reshape(-1, inputs[name][0].shape[-1])

            for i in range(attempts):
                tensor = inputs[name]
                qtensor = quantize(tensor, scale)
                loss = (qtensor - tensor).pow(2).sum(dim=-1).mean()
                # if depth == 31:
                #     loss = (qtensor - tensor).pow(2).sum(dim=-1).mean()
                # else:
                #     loss = clamp_ste(qtensor - tensor, -scale * 3, scale * 3).pow(2).sum(dim=-1).mean()
                optim.zero_grad()
                loss.backward()
                optim.step()
                if i == 0: first_loss = float(loss)
                if (i+1) % (attempts//5) == 0:
                    loss = float(loss)
                    print(name, i, float(scale), loss, loss / first_loss)

            m.in_q.scale.data = scale

class SoftmaxQuantizer(nn.Module):
    def __init__(self, th=0.025):
        super(SoftmaxQuantizer, self).__init__()
        self.th = th

    def forward(self, x):
        x1 = (x.float() * 127 / self.th).clamp(0, 127).round() / 127 * self.th
        x2 = ((x.float() - self.th) * 127 / (1-self.th)).clamp(0, 127).round() / 127 * (1-self.th)
        return x1.to(x.dtype), x2.to(x.dtype)
    
class SoftmaxQuantizer_v2(nn.Module):
    def __init__(self, th=0.5):
        super(SoftmaxQuantizer_v2, self).__init__()
        self.th = th

    def forward(self, x):
        return (x.float() * 127 / self.th).clamp(0, 127).round() / 127 * self.th

def hack_attn_forward(model, act_scales, softmax_quant_mode="attn_split"):
    device = next(model.parameters()).device
    cnt = -1
    for name, m in model.named_modules():
        if isinstance(m, Phi3Attention):
            cnt += 1
            #m.qk_quantizer = make_quantizer(device, "dynamic_per_token")
            m.mode = softmax_quant_mode#"attn_split"
            # m.mode = "none"
            if m.mode == "none": qk_scale = "none"
            else:
                qk_scale = calc_init_scale(act_scales[name + ".qk_observer"], 127, method="minmax")
            m.qk_quantizer = make_quantizer(device, qk_scale)
            if m.mode == "attn_split":
                m.soft_quantizer = SoftmaxQuantizer()
            elif m.mode == "softmax_sup":
                m.soft_quantizer = make_quantizer(device, 1/127)
                # m.soft_quantizer = SoftmaxQuantizer_v2()
            elif m.mode == "per_tensor":
                m.soft_quantizer = make_quantizer(device, 1/127)
            elif m.mode == "per_tensor_255":
                m.soft_quantizer = make_quantizer(device, 1/255)
                m.soft_quantizer.Qp, m.soft_quantizer.Qn = 255, 0
            elif m.mode == "none":
                pass
            else:
                assert ValueError(), m.mode

def plot_bin(t):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(5,5))
    step = 512
    t = t.reshape(-1)
    t = torch.bincount((t*step).round().int())
    # t = torch.bincount((t*step).round().add(1).log().int())
    t = t.detach().cpu().numpy()
    x = np.arange(step+1)
    # x = np.linspace(0,1,step+1)
    # plt.bar(x, t)
    plt.bar(x, t)
    path = "xxx.png"
    # plt.bar(x[1:], t[1:])
    # path = "yyy.png"
    plt.savefig(path)
    print("save as %s" % path)
    exit(0)

def apply_squant(model):
    for m in model.modules():
        if isinstance(m, QLinear):
            m.weight.data = squant_flip(m.weight.data, m.w_q.scale, -128, 127)

def apply_squant_aware_act_scale(model, act_scales):
    for name, m in model.named_modules():
        if isinstance(m, QLinear):
            # m.weight.data = squant_flip_aware_act_scale(m.weight.data, m.w_q.scale, act_scales[name].pow(-1.), -128, 127)
            m.weight.data = squant_flip_aware_act_scale(m.weight.data, m.w_q.scale, act_scales[name].pow(0.25), -128, 127)

@torch.no_grad()
def apply_gptq(model, tokenizer, dataset, seqlen=512, nsamples=256):
    dev = model.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    print_memory_usage()
    print('cpu offload')
    model.cpu()
    print_memory_usage()

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            # print(cache['i'], inp.shape)
            if cache['i'] < nsamples:
                inps[cache['i']] = inp
                cache['i'] += 1
                cache['attention_mask'] = kwargs['attention_mask']
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])

    if hasattr(dataset, 'input_ids'):
        input_ids = [dataset.input_ids[:,i:i+seqlen] for i in range(0,dataset.input_ids.shape[1],seqlen)]
        input_ids = input_ids[:nsamples]
    else:
        text = ""
        cnt = 0
        for data in dataset:
            if len(data["text"]) > 0:
                text += data["text"]
                cnt += 1
                if cnt > nsamples*seqlen//50: break

        input_ids = tokenizer(
            text, return_tensors="pt", max_length=nsamples*seqlen, truncation=True
        ).input_ids
        input_ids = input_ids[:input_ids.numel()//seqlen*seqlen].reshape(-1, seqlen)

    for batch in input_ids:
        try:
            model(batch.to(dev)[None])
        except ValueError:
            pass

    layers[0] = layers[0].module.cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    def weight_quantizer(w):
        w_s = calc_init_scale(w, 127, method="minmax")
        return w.div(w_s).round().clamp(-128, 127).mul(w_s)

    bar = tqdm(total=len(layers))
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        sequential = [
            ['self_attn.qkv_proj'],
            # ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
            ['self_attn.o_proj'],
            ['mlp.up_proj', 'mlp.gate_proj'],
            ['mlp.down_proj']
        ]

        modules = {}
        for name, m in layer.named_modules():
            for names in sequential:
                if name in names: modules[name] = m
       
        for names in sequential:
            subset = {n: modules[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name], weight_quantizer)

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                # print(i, name)
                gptq[name].fasterquant(128, percdamp=0.001)
                # gptq[name].fasterquant(128, percdamp=0.001)
                gptq[name].free()

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

        bar.update(1)


    model.config.use_cache = use_cache

    print('reload to gpu')
    model.to(dev, dtype=dtype)
    print_memory_usage()
