import torch

def my_round(input):
    sign = torch.sign(input)
    output = sign * torch.floor(torch.abs(input) + 0.5)
    return output

@torch.no_grad()
def squant_flip(x, alpha, Qn=None, Qp=None):
    x_q = x / alpha
    rounding_number = my_round(x_q)
    # print(x_q - rounding_number)
    #rounding_number = squant_flip_per_dim(rounding_number, x_q, 1)
    rounding_number = squant_flip_per_dim(rounding_number, x_q, 0) # squant-c
    rounding_number = squant_flip_per_dim(rounding_number[None], x_q[None], 0)[0] # squant-e
    rounding_number = torch.clamp(rounding_number, Qn, Qp)
    # print(float((x_q - rounding_number).mean()))
    return rounding_number * alpha

def squant_flip_aware_act_scale(x, alpha, scale, Qn=None, Qp=None):
    x_q = x / alpha
    rounding_number = my_round(x_q)
    rounding_number = squant_flip_per_dim_aware_act_scale(rounding_number, x_q, scale, 0) # squant-c
    rounding_number = squant_flip_per_dim_aware_act_scale(rounding_number[None], x_q[None], scale, 0)[0] # squant-e
    rounding_number = torch.clamp(rounding_number, Qn, Qp)
    # print(float((x_q - rounding_number).mean()))
    return rounding_number * alpha

@torch.no_grad()
def squant_flip_for_conv(x, alpha, Qn=None, Qp=None):
    x_q = x / alpha
    rounding_number = my_round(x_q)
    rounding_number_origin = rounding_number
    #rounding_number = squant_flip_per_dim(rounding_number, x_q, 2)
    rounding_number = squant_flip_per_dim(rounding_number, x_q, 1) # squant-k
    rounding_number = squant_flip_per_dim(rounding_number, x_q, 0) # squant-c
    rounding_number = squant_flip_per_dim(rounding_number[None], x_q[None], 0)[0] # squant-e
    rounding_number = torch.clamp(rounding_number, Qn, Qp)
    rounding_number_delta = rounding_number - rounding_number_origin
    x += rounding_number_delta * alpha
    #x += torch.where(rounding_number_delta != 0, (rounding_number - rounding_number.sign() * 0.4) * alpha - x, 0)
    return x

def squant_flip_per_dim(w, origin_w, dim=1, k=1):
    delta = w - origin_w
    # delta = delta / torch.clamp(origin_w.abs(), 1)
    #delta = delta * scale.reshape(scale_reshape)
    delta = delta.flatten(dim+1).flatten(0,dim)
    n_flip = my_round(delta.sum(dim=-1) * k).int()
    # print('n_flip', n_flip, delta.size())
    w = w.clone()
    w = w.flatten(dim+1).flatten(0, dim)
    order = delta.argsort(dim=1, descending=True)

    # for i in range(n_flip.shape[0]):
    #     if n_flip[i] > 0: # down
    #         index = order[i,:int(n_flip[i])]
    #         w[i,index] = w[i,index] - 1
    #     elif n_flip[i] < 0: # up
    #         index = order[i,int(n_flip[i]):]
    #         w[i,index] = w[i,index] + 1

    idx = order.argsort(dim=1)
    n_flip = n_flip[...,None].expand_as(idx)
    w = torch.where(idx < n_flip, w-1, w)
    w = torch.where(idx >= idx.shape[-1]+n_flip, w+1, w)
    
    w = w.reshape(origin_w.shape)
    return w

def squant_flip_per_dim_aware_act_scale(w, origin_w, act_scale, dim=1, k=1.):
    delta = (w - origin_w).mul(act_scale)
    delta = delta.flatten(dim+1).flatten(0,dim)
    delta_flip = delta.sum(dim=-1) * k
    flip = -delta_flip.sign()[:,None]
    w = w.clone().flatten(dim+1).flatten(0, dim)
    act_scale = torch.ones_like(origin_w).mul(act_scale).reshape(w.shape)
    delta = delta.add(flip*act_scale).mul(flip)
    delta = torch.where(delta <= 0, torch.inf, delta)
    delta, order = delta.sort(dim=1)
    cumsum_delta = torch.cumsum(act_scale.gather(1, order), 1)
    
    # for i in range(delta_flip.shape[0]):
    #     if delta_flip[i] > 0: # down
    #         index = order[i][cumsum_delta[i] < delta_flip[i]]
    #         w[i,index] = w[i,index] - 1
    #     elif delta_flip[i] < 0: # up
    #         index = order[i][cumsum_delta[i] < -delta_flip[i]]
    #         w[i,index] = w[i,index] + 1

    cumsum_delta = cumsum_delta.gather(1, order.argsort(dim=1))
    delta_flip = delta_flip[...,None].expand_as(w)
    w = torch.where(cumsum_delta < delta_flip.abs(), w+flip, w)

    w = w.reshape(origin_w.shape)
    return w

def _squant_flip_per_dim_aware_act_scale(w, origin_w, act_scale, dim=1, k=1):
    delta = w - origin_w
    delta_s = delta * act_scale
    delta = delta.flatten(dim+1).flatten(0,dim)
    delta_s = delta_s.flatten(dim+1).flatten(0,dim)
    n_flip = my_round(delta.sum(dim=-1) * k).int()
    w = w.clone().flatten(dim+1).flatten(0, dim)
    order = delta_s.argsort(dim=1, descending=True)

    idx = order.argsort(dim=1)
    n_flip = n_flip[...,None].expand_as(idx)
    w = torch.where(idx < n_flip, w-1, w)
    w = torch.where(idx >= idx.shape[-1]+n_flip, w+1, w)
    
    w = w.reshape(origin_w.shape)
    return w