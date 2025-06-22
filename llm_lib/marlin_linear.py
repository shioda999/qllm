import marlin
import torch
import functools
import operator


def prod(shape):
    return functools.reduce(operator.mul, shape, 1)

def gen_quant4(w, groupsize=-1, dev=torch.device("cuda")):
    tile = 16
    maxq = 2 ** 4 - 1
    m, n = w.shape
    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq
    w = torch.round(w / s).int()
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)
    ref = (w - (maxq + 1) // 2).half() * s
    if groupsize != -1:
        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((m, n)).contiguous()
            return w
        ref = reshape(ref)
        w = reshape(w)
    s = s.reshape((-1, n)).contiguous()
    linear = torch.nn.Linear(m, n)
    linear.weight.data = ref.t()
    # Workaround to test some special cases that are forbidden by the API
    layer = marlin.Layer(256, 256, groupsize=groupsize)
    if groupsize == -1:
        groupsize = m
    layer.k = m
    layer.n = n
    layer.groupsize = groupsize
    layer.B = torch.empty((m // 16, n * 16 // 8), dtype=torch.int, device=dev)
    layer.s = torch.empty((m // groupsize, n), dtype=torch.half, device=dev)
    layer.pack(linear, s.t())
    q = layer.B
    s = layer.s
    return ref, q, s

class MarlinLinear:
  def __init__(self, w):
      k, m = w.shape
      self.workspace = torch.zeros(m // 128 * 16, device=w.device)
      self.groupsize = 128 # -1 # 128
      
      gpu = torch.cuda.get_device_name(0)
      if 'A100' in gpu:
          SMS = 108
      elif 'A10' in gpu:
          SMS = 72
      elif '3090' in gpu:
          SMS = 82
      elif 'A6000' in gpu:
          SMS = 84
      elif 'L4' in gpu:
          SMS = 48
      else:
          SMS = -1
      # thread_k, thread_n, sms = 256, 256, SMS
      thread_k, thread_n, sms = -1, -1, SMS
      self.thread_k = thread_k
      self.thread_n = thread_n
      self.sms = sms
      self.m = m
      self.w_ref, self.qw, self.s = gen_quant4(w, groupsize=self.groupsize, dev=w.device)
    
  def __call__(self, x):
      C = torch.zeros((prod(x.shape[:-1]), self.m), dtype=torch.half, device=x.device)
      marlin.mul(x.view(-1, x.shape[-1]), self.qw, C, self.s, self.workspace, self.thread_k, self.thread_n, -1)  # 修正: A → x
      return C.view(x.shape[:-1] + (-1,))