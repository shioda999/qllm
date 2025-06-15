import torch
import torch.nn as nn
from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer, Phi3MLP, Phi3Attention, Phi3RMSNorm
from sklearn.cluster import KMeans
from llm_lib.hadamard import generate_hadamard_matrix
import math

@torch.no_grad()
def apply_rotate_qk_proj(model, act_scales=None):
    Q = torch.tensor([[1,-1],[1,1]], device=model.device)/math.sqrt(2)
    for name, module in model.named_modules():
        if isinstance(module, Phi3DecoderLayer):
            n_heads = module.self_attn.num_key_value_heads
            # s_q, s_k, s_v = act_scales[name + '.self_attn.qkv_proj_output'].chunk(3,0)
            w_q, w_k, w_v = module.self_attn.qkv_proj.weight.chunk(3,0)
            dim = w_q.shape[-1]
            w_q = (Q @ w_q.reshape(n_heads,2,-1,dim).transpose(1,2).to(Q.dtype)).to(w_q.dtype).transpose(1,2).reshape(-1,dim)
            w_k = (Q @ w_k.reshape(n_heads,2,-1,dim).transpose(1,2).to(Q.dtype)).to(w_k.dtype).transpose(1,2).reshape(-1,dim)
            module.self_attn.qkv_proj.weight.data = torch.concat([w_q, w_k, w_v], 0)

@torch.no_grad()
def apply_rotate_vo_proj(model, act_scales):
    dtype = next(model.parameters()).dtype
    for name, module in model.named_modules():
        if isinstance(module, Phi3DecoderLayer):
            n_heads = module.self_attn.num_key_value_heads
            w_q, w_k, w_v = module.self_attn.qkv_proj.weight.chunk(3,0)
            w_o = module.self_attn.o_proj.weight
            metric = act_scales[name + '.self_attn.o_proj'].reshape(n_heads, -1)
            Q = torch.block_diag(*[get_block_diag_rotate_mat(model, metric[i], k=1) for i in range(n_heads)])
            # Q = torch.block_diag(*[get_block_diag_rotate_mat_v2(model, metric[i], k=1) for i in range(n_heads)])
            w_v = (Q.T @ w_v.to(Q.dtype)).to(w_v.dtype)
            w_o = (w_o.to(Q.dtype) @ Q).to(w_o.dtype)
            module.self_attn.qkv_proj.weight.data = torch.concat([w_q, w_k, w_v], 0)
            module.self_attn.o_proj.weight.data = w_o

@torch.no_grad()
def rotate_ln_qkv(model, Q):
    dtype = next(model.parameters()).dtype
    for name, module in model.named_modules():
        if isinstance(module, Phi3DecoderLayer):
            attn_ln = module.input_layernorm
            qkv = module.self_attn.qkv_proj
            w = qkv.weight.float() * attn_ln.weight.float() @ Q
            qkv.weight.data = w.to(dtype)
            attn_ln.weight.data = torch.ones_like(attn_ln.weight.data)

@torch.no_grad()
def rotate_o_proj_out(model, Q):
    dtype = next(model.parameters()).dtype
    for name, module in model.named_modules():
        if isinstance(module, Phi3DecoderLayer):
            o_proj = module.self_attn.o_proj
            w = (Q.T @ o_proj.weight.float()).to(dtype)
            o_proj.weight.data = w.to(dtype)

@torch.no_grad()
def rotate_ln_gate_up(model, Q):
    dtype = next(model.parameters()).dtype
    for name, module in model.named_modules():
        if isinstance(module, Phi3DecoderLayer):
            ffn_ln = module.post_attention_layernorm
            fcs = [module.mlp.gate_proj, module.mlp.up_proj]
            w0 = (fcs[0].weight * ffn_ln.weight).float() @ Q
            w1 = (fcs[1].weight * ffn_ln.weight).float() @ Q
            fcs[0].weight.data = w0.to(dtype)
            fcs[1].weight.data = w1.to(dtype)
            ffn_ln.weight.data = torch.ones_like(ffn_ln.weight.data)
               
@torch.no_grad()
def rotate_down_proj_out(model, Q):
    dtype = next(model.parameters()).dtype
    for name, m in model.named_modules():
        if isinstance(m, Phi3MLP):
            m.down_proj.weight.data = (Q.T @ (m.down_proj.m.float()[:,None] * m.down_proj.weight.float())).to(dtype)
            m.down_proj.m.data = torch.ones_like(m.down_proj.m.data)

def random_orthogonal_matrix(size, device):
    # return generate_hadamard_matrix(size, device)
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q

def get_metric_scales(model, act_scales):
    scales = []
    for name, m in model.named_modules():
        if isinstance(m, Phi3Attention):
            s = act_scales[name + ".qkv_proj"].abs()
            s = s / s.norm(p=2)
            scales.append(s)
            s = act_scales[name + ".o_proj_output"].abs()
            s = s / s.norm(p=2)
            scales.append(s)
        if isinstance(m, Phi3MLP):
            s = act_scales[name + ".up_proj"].abs()
            s = s / s.norm(p=2)
            scales.append(s)
            s = act_scales[name + ".down_proj_output"].abs()
            s = s / s.norm(p=2)
            scales.append(s)
    scales = torch.stack(scales)
    scales = scales.norm(p=2, dim=0)
    return scales

def get_block_diag_rotate_mat(model, metric, k=64):
    device = next(model.parameters()).device
    # scales = get_metric_scales(model, act_scales)
    idx = metric.argsort(dim=-1, descending=True)
    group_sz = idx.shape[0] // k
    tmp_idx = torch.arange(idx.shape[0])
    tmp_idx = tmp_idx % group_sz * k + tmp_idx // group_sz
    # for i in range(idx.shape[0]):
    #     tmp_idx[(((i % k) if i // k % 2 == 0 else (k - 1 - i % k)) * group_sz) + i // k] = i
    idx = idx[tmp_idx]
    Q = torch.block_diag(*[generate_hadamard_matrix(group_sz, device) for _ in range(k)])
    # Q = torch.block_diag(*[random_orthogonal_matrix(group_sz, device) for _ in range(k)])
    # Q = Q.gather(1, idx[None].expand_as(Q))
    Q = Q.gather(0, idx[:,None].expand_as(Q))
    return Q

def get_block_diag_rotate_mat_v2(model, metric, k=64, protect=0):
    device = next(model.parameters()).device
    # scales = get_metric_scales(model, act_scales)
    idx = metric.argsort(dim=-1, descending=True)

    group_sz = idx.shape[0] // k
    if k == 1: protect = 0
    n = idx.shape[0] - protect
    topk_scale = metric[idx[:k]].pow(-0.1)#.add(1).log()
    each_group_sz_cumsum = torch.round(torch.concat([torch.tensor([0], device=device), torch.cumsum(topk_scale, dim=0)]) / topk_scale.sum() * n).int()
    each_group_sz = torch.diff(each_group_sz_cumsum)
    if each_group_sz.sum() != n:
        each_group_sz[0] += n - each_group_sz.sum()
    if len(each_group_sz) > 1:
        print(each_group_sz)
        
    each_group_cnt = [0 for _ in each_group_sz]
    idx2 = [0 for _ in idx]
    j = 0
    for i in range(n):
        while each_group_cnt[j] >= each_group_sz[j]:
            j = (j + 1) % k
        idx2[each_group_cnt[j] + each_group_sz_cumsum[j]] = idx[i]
        each_group_cnt[j] += 1
        j = (j + 1) % k
    idx = torch.tensor(idx2, device=device)

    Q = [generate_hadamard_matrix(v, device) for v in each_group_sz]
    if protect > 0: Q.append(torch.eye(protect, device=device))
    Q = torch.block_diag(*Q)
    Q = Q.gather(0, idx[:,None].expand_as(Q))
    return Q

def get_rotate_mat(model, act_scales, size=3072):
    # return torch.block_diag(*[generate_hadamard_matrix(64, device=model.device) for i in range(size//64)]).float()
    metric = get_metric_scales(model, act_scales)
    Q = get_block_diag_rotate_mat(model, metric)
    # Q = get_block_diag_rotate_mat_v2(model, metric)
    return Q.float()

def apply_rotate(model, Q):
    dtype = next(model.parameters()).dtype
    for name, m in model.named_modules():
        if isinstance(m, nn.Embedding):
            m.weight.data = (m.weight.float() @ Q).to(dtype)

    w = (model.lm_head.weight.float() * model.model.norm.weight) @ Q
    model.lm_head.weight.data = w.to(dtype)
    model.model.norm.weight.data = torch.ones_like(model.model.norm.weight.data)

    rotate_ln_qkv(model, Q)
    rotate_o_proj_out(model, Q)
    rotate_ln_gate_up(model, Q)
    rotate_down_proj_out(model, Q)