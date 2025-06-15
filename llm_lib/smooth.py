import torch
import torch.nn as nn
from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer, Phi3MLP, Phi3Attention
from typing import Optional, Tuple
from transformers.cache_utils import Cache

class LinearMul(nn.Linear):
    '''
    down_proj の出力にスケーリングファクタを導入。
    '''
    def set_mul_param(self, mul):
        self.m = nn.Parameter(mul.clone().to(self.weight.device).to(self.weight.dtype), False)
    def forward(self, x):
        return super().forward(x) * self.m
    
class Phi3MLPv2(Phi3MLP):
    '''
    gate_proj, up_projの二つに分割したバージョンのMLPブロック。
    '''
    def divide_gate_up_proj(self):
        config = self.config
        w_g, w_u = self.gate_up_proj.weight.data.chunk(2, 0)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.gate_proj.weight.data = w_g
        self.up_proj.weight.data = w_u
        del self.gate_up_proj

    def append_mul_operator(self):
        self.down_proj.__class__ = LinearMul
        self.down_proj.set_mul_param(self.down_proj.weight[:,0].pow(0.))

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = self.up_proj(hidden_states) * self.activation_fn(self.gate_proj(hidden_states))
        o = self.down_proj(hidden_states)
        return o

# class Phi3Attentionv2(Phi3Attention):
#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Cache] = None,
#         output_attentions: bool = False,
#         use_cache: bool = False,
#         cache_position: Optional[torch.LongTensor] = None,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
def convert_mlp(model):
    '''
    デフォルトでは、二つの線形層がgate_up_projという一つの線形層に統合されているため、
    gate_proj, up_projの二つに分割します。
    '''
    for m in model.modules():
        if isinstance(m, Phi3MLP):
            m.__class__ = Phi3MLPv2
            m.divide_gate_up_proj()
            m.append_mul_operator()

@torch.no_grad()
def smooth_ln_fcs(ln, fcs, scale, alpha=0.5, beta=0.5):
    '''
    LayerNorm層とFC層間でスムーズ化
    '''
    if not isinstance(fcs, list):
        fcs = [fcs]
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == scale.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    scale = scale.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    #(act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
    scales = (
        (scale.pow(alpha) / weight_scales.pow(beta))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    ln.weight.div_(scales)
    if hasattr(ln, 'bias') and ln.bias is not False: ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))
    return scales


@torch.no_grad()
def smooth_act(pre_fc, fcs, scale, alpha=0.5, beta=0.,gamma=0.):
    '''
    LayerNorm/FC層とFC層間でスムーズ化
    '''
    if not isinstance(fcs, list):
        fcs = [fcs]
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert pre_fc.weight.shape[0] == fc.in_features == scale.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    scale = scale.to(device=device, dtype=dtype)

    pre_weight_scales = pre_fc.weight.abs()#
    if len(pre_fc.weight.shape) > 1: pre_weight_scales = pre_weight_scales.max(dim=1)[0]
    post_weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    post_weight_scales = post_weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (
        (scale.pow(alpha) / post_weight_scales.pow(beta) * pre_weight_scales.pow(gamma))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    if len(pre_fc.weight.shape) > 1: pre_fc.weight.div_(scales.reshape(-1,1))
    else: pre_fc.weight.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))
    return scales

@torch.no_grad()
def smooth_ln_qkv(model, scales, alpha=0.5, beta=0.5, gamma=0.):
    '''
    一つ目のLayerNormとqkv_projのスムーズ化
    '''
    for name, module in model.named_modules():
        if isinstance(module, Phi3DecoderLayer):
            attn_ln = module.input_layernorm
            qkv = [module.self_attn.qkv_proj]
            s = smooth_act(attn_ln, qkv, scales[name + ".self_attn.qkv_proj"], alpha, beta, gamma)
            scales[name + ".self_attn.qkv_proj"].div_(s)

@torch.no_grad()
def smooth_ln_gate_up(model, scales, alpha=0.5, beta=0.5, gamma=0.):
    for name, module in model.named_modules():
        if isinstance(module, Phi3DecoderLayer):
            ffn_ln = module.post_attention_layernorm
            fcs = [module.mlp.gate_proj, module.mlp.up_proj]
            s = torch.maximum(scales[name + ".mlp.gate_proj"], scales[name + ".mlp.up_proj"])
            s = smooth_act(ffn_ln, fcs, s, alpha, beta, gamma)
            scales[name + ".mlp.gate_proj"].div_(s)
            scales[name + ".mlp.up_proj"].div_(s)

@torch.no_grad()
def smooth_down_proj(model, scales, alpha=0.5, beta=0.25, gamma=0.):
    for name, module in model.named_modules():
        if isinstance(module, Phi3DecoderLayer):
            s = smooth_act(module.mlp.up_proj, [module.mlp.down_proj], scales[name + ".mlp.down_proj"], alpha, beta, gamma)
            scales[name + ".mlp.up_proj_output"].div_(s)
            scales[name + ".mlp.down_proj"].div_(s)

@torch.no_grad()
def smooth_down_proj_gate(model, scales, alpha=0.2, beta=0.0):
    for name, module in model.named_modules():
        if isinstance(module, Phi3DecoderLayer):
            s = smooth_act(module.mlp.gate_proj, [module.mlp.down_proj], scales[name + ".mlp.down_proj"], alpha, beta)
            scales[name + ".mlp.gate_proj_output"].div_(s)
            scales[name + ".mlp.down_proj"].div_(s)
    
@torch.no_grad()
def smooth_down_proj_out(model, act_scales, alpha=1.0, beta=0.25):
    #return
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for name, m in model.named_modules():
        if isinstance(m, Phi3MLP):
            assert isinstance(m.down_proj, LinearMul)
            m.down_proj.weight.data = m.down_proj.weight.data * m.down_proj.m[:,None]
            s_act = act_scales[name + ".down_proj_output"]
            s_w = m.down_proj.weight.data.max(dim=1)[0]
            s = s_act.pow(alpha) * s_w.pow(beta)
            m.down_proj.weight.data /= s[:,None].to(dtype)
            act_scales[name + ".down_proj_output"] /= s
            # m.down_proj.__class__ = LinearMul
            m.down_proj.set_mul_param(s)

@torch.no_grad()
def smooth_qk_proj(model, act_scales, alpha=0.5, beta=0.):
    for name, m in model.named_modules():
        if isinstance(m, Phi3Attention):
            # pass
            num_heads = m.num_heads
            w = m.qkv_proj.weight.data
            s_act = act_scales[name + ".qkv_proj_output"]
            s_w = w.abs().max(dim=1)[0]
            
            s_q_act, s_k_act, s_v_act = s_act.reshape(3, -1).unbind()
            s_q_w, s_k_w, s_v_w = s_w.reshape(3, -1).unbind()
            # def fold(t):
            #     return torch.maximum(*t.chunk(2, 0))
            # def unfold(t):
            #     return torch.cat([t, t])
            def fold(t):
                return torch.maximum(*t.reshape(num_heads, -1).chunk(2, 1)).reshape(-1)
            def unfold(t):
                return torch.cat([t.reshape(num_heads, -1) for _ in range(2)], -1).reshape(-1)
            s_q_act2, s_k_act2, s_q_w, s_k_w = [fold(t) for t in [s_q_act, s_k_act, s_q_w, s_k_w]]
            w = w.reshape(3, -1, w.shape[-1])
            w_q, w_k, w_v = w.unbind(0)
            s = s_q_act2.pow(-alpha) * s_k_act2.pow(alpha) * s_q_w.pow(-beta) * s_k_w.pow(beta)
            s = unfold(s)
            w_q.mul_(s[:,None])
            w_k.div_(s[:,None])
            s_q_act.mul_(s)
            s_k_act.div_(s)

            m.qkv_proj.weight.data = torch.cat([w_q, w_k, w_v])
            act_scales[name + ".qkv_proj_output"] = torch.cat([s_q_act, s_k_act, s_v_act])


@torch.no_grad()
def smooth_vo_proj(model, act_scales, alpha=0.5, beta=0., gamma=0.0):
    for name, m in model.named_modules():
        if isinstance(m, Phi3Attention):
            pass
            w = m.qkv_proj.weight.data
            s_act = act_scales[name + ".qkv_proj_output"]
            s_w = w.abs().max(dim=1)[0]
            
            s_q_act, s_k_act, s_v_act = s_act.reshape(3, -1).unbind()
            s_q_w, s_k_w, s_v_w = s_w.reshape(3, -1).unbind()

            w = w.reshape(3, -1, w.shape[-1])
            w_q, w_k, w_v = w.unbind(0)

            w_o = m.o_proj.weight.data
            s_o_w = w_o.abs().max(dim=0)[0]

            s = s_v_w.pow(-alpha) * s_v_act.pow(-beta) * s_o_w.pow(gamma)
            w_v.mul_(s[:,None])
            s_v_act.mul_(s)
            w_o.div_(s[:,None])

            m.qkv_proj.weight.data = torch.cat([w_q, w_k, w_v])
            act_scales[name + ".qkv_proj_output"] = torch.cat([s_q_act, s_k_act, s_v_act])
            m.o_proj.weight.data = w_o
            act_scales[name + ".o_proj"].mul_(s)


@torch.no_grad()
def smooth_head(model, act_scales, alpha=0.5, beta=0.5, gamma=0.0):
    s = smooth_act(model.model.norm, [model.lm_head], act_scales["lm_head"], alpha, beta, gamma)
    act_scales["lm_head"].div_(s)
