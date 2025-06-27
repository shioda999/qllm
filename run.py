# python run_quantization.py --use_flash_attn --eval_benchmark --lm_eval_tasks openbookqa
 
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from datasets import load_dataset, load_from_disk
import argparse

from llm_lib.calibrate import get_act_scales
from llm_lib.utils import set_seed
from llm_lib.smooth import convert_mlp, smooth_ln_qkv, smooth_ln_gate_up, smooth_down_proj,\
      smooth_down_proj_out, smooth_qk_proj, smooth_vo_proj, smooth_head
from llm_lib.rotate import get_rotate_mat, apply_rotate, apply_rotate_qk_proj, apply_rotate_vo_proj
from llm_lib.quantize import quantize_model, hack_attn_forward, apply_gptq
# from llm_lib.export_onnx import export_onnx, convert_quantizer, convert_for_onnx
# from llm_lib.debug import check_matmul_shape

from llm_lib.eval import eval_ppl
from llm_lib.marlin_linear import MarlinLinear
# from llm_lib.run_qa_task import run_qa_task
# from llm_lib.run_lm_eval import run_lm_eval
import time
import re

def str2bool(s):
     return s.lower() in ["true", "t", "yes", "1", "on"]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42)
    parser.add_argument('--model', default='microsoft/Phi-3-mini-4k-instruct')
    # 量子化設定
    parser.add_argument('--smooth', type=str2bool, default=True) # smoothquant のスムーズ化
    parser.add_argument('--rotate', type=str2bool, default=True) # quarotの回転スムーズ化
    parser.add_argument('--gptq', action='store_true') # GPTQ 遅い上に効果が微妙
    
    # benchmark
    parser.add_argument('--text_gen', type=str2bool, default=True) # 自己紹介文生成
    parser.add_argument('--eval_ppl', type=str2bool, default=True) # wikitext
    return parser.parse_args()

def load_model(model_name):
    kwargs = { "torch_dtype": torch.float16, "device_map": "auto" }
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer

@torch.no_grad()
def test_text_generation(model, tokenizer):
    messages = [
        {"role": "system", "content": "You are chatbot."},
        {"role": "user", "content": "List numbers from 1 to 10, each numbers is separated by comma."},
        # {"role": "user", "content": "Please Introduce yourself."},
        # {"role": "user", "content": "Please talk about global warming as long as you can."},
    ]
    # messages = [
    #     {"role": "user", "content": "こんにちは。何か適当に自己紹介して。"}
    # ]
    pipe = pipeline("text-generation", model, tokenizer=tokenizer)
    ret = pipe(messages, max_length=100)
    print(f"\nOUTPUT:=====\n{ret[0]['generated_text'][-1]['content']}\n============")


class ExecutionTimer:
    def __init__(self, label="Execution time"):
        self.label = label

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self  # withブロック内で `as` で使う場合

    def __exit__(self, exc_type, exc_value, traceback):
        end_time = time.perf_counter()
        elapsed_time = end_time - self.start_time
        print(f"{self.label}: {elapsed_time:.6f} seconds") 

def save_compact_dataset(dataset):
    import os
    sub_dataset = dataset.select(range(512))
    save_dir = "./subset_c4"
    os.makedirs(save_dir, exist_ok=True)
    sub_dataset.save_to_disk(save_dir)

@torch.no_grad()
def quantize(args, model ,tokenizer):
    seq_len = 2048
    # dataset = load_dataset(
    #     'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation',
    #     # download_mode="force_redownload"
    # )
    dataset = load_from_disk("./data/subset_c4")

    # convert_mlp(model)
    # act_scales = get_act_scales(model, tokenizer, dataset, "calib3", seq_len=seq_len, mode="topk")
    act_scales = {}
        
    if args.rotate:
        apply_rotate_vo_proj(model, act_scales)
        smooth_qk_proj(model, act_scales, 0.5)
        apply_rotate_qk_proj(model)
        Q = get_rotate_mat(model, act_scales)
        apply_rotate(model, Q)
        act_scales = get_act_scales(model, tokenizer, dataset, None, seq_len=seq_len, mode="topk")
        # act_scales = get_act_scales(model, tokenizer, dataset, "calib1", seq_len=seq_len, mode="topk")

    if args.smooth:
        smooth_qk_proj(model, act_scales, 0.5)
        smooth_vo_proj(model, act_scales, 0.25)
        smooth_ln_qkv(model, act_scales, 0.5, 0.5)
        # smooth_ln_gate_up(model, act_scales, 0.5, 0.5)
        # smooth_down_proj_out(model, act_scales, 1.0, 0.25)
        # smooth_down_proj(model, act_scales, 0.5, 0.25)
        # smooth_head(model, act_scales, 0.5, 0.5)
        act_scales = get_act_scales(model, tokenizer, dataset, None, seq_len=seq_len, mode="topk")
        # act_scales = get_act_scales(model, tokenizer, dataset, "calib2", seq_len=seq_len, mode="topk")

        smooth_down_proj_out(model, act_scales, 1.0, 0.25)
        smooth_down_proj(model, act_scales, 0.5, 0.25)
        smooth_ln_gate_up(model, act_scales, 0.5, 0.5)
        smooth_head(model, act_scales, 0.5, 0.5)
    
    quantize_model(model, act_scales, down_proj_in_scale_mul=0.5)
    
    if args.gptq:
        apply_gptq(model, tokenizer, dataset)

def eval(args, model, tokenizer):
    if args.text_gen:
        with ExecutionTimer():
            test_text_generation(model, tokenizer)
            
    if args.eval_ppl:
        with ExecutionTimer():
            model.seqlen = 2048
            eval_ppl(model, tokenizer, datasets=["wikitext2"])

def benchmark(f, warmup=1, iter=10):
    for i in range(warmup + iter):
        f()
        if i == warmup - 1:
            torch.cuda.synchronize()
            tick = time.time()
    torch.cuda.synchronize()
    res = (time.time() - tick) / iter
    time.sleep(1.)
    return res

def test(model, tokenizer):
  input_text = "Hello. I am sleepy. This is test script. " * 10
  print("!!!", input_text)
  input_ids = tokenizer.encode(input_text, return_tensors="pt").cuda()
  # with ExecutionTimer():
  #     model.model(input_ids)
  print("exection time:", benchmark(lambda: model.model(input_ids)))

def call_linear(model, x):
  for m in model.modules():
    if isinstance(m, torch.nn.Linear):
        m(x[:,:m.weight.shape[-1]])

def call_decoder(model, x):
  for m in model.modules():
    if hasattr(m, "self_attn"):
        m(x[:,None,:3072], position_embeddings=(torch.zeros((1,)).half().cuda(), torch.zeros((1,)).half().cuda()), causal_mask=torch.zeros((1,1,1,1)).half().cuda())

def call_attn(model, x):
  for m in model.modules():
    if hasattr(m, "qkv_proj"):
        m(x[:,None,:3072], (torch.zeros((1,)).half().cuda(), torch.zeros((1,)).half().cuda()), torch.zeros((1,1,1,1)).half().cuda())

def call_mlp(model, x):
  for m in model.modules():
    if hasattr(m, "down_proj"):
        m(x[:,:3072])
        
def call_norm(model, x):
  for m in model.modules():
    if hasattr(m, "variance_epsilon"):
        m(x[:,:3072])

def test2(model):
  x = torch.randn((1,9216), dtype=torch.half).cuda()
  print("exection time  linear:", benchmark(lambda: call_linear(model, x)))
  print("exection time decoder:", benchmark(lambda: call_decoder(model, x)))
  print("exection time    attn:", benchmark(lambda: call_attn(model, x)))
  print("exection time     mlp:", benchmark(lambda: call_mlp(model, x)))
  print("exection time    norm:", benchmark(lambda: call_norm(model, x)))

def main():
    args = get_args()
    set_seed(args.seed)
    model, tokenizer = load_model(args.model)
    torch.compile(model.model)
    
    # test(model, tokenizer)
    # test2(model)
    eval(args, model, tokenizer)
    # eval(args, model, tokenizer)

    log_dir = "./log"
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
        record_shapes=True,
        with_stack=True
    ) as prof:
        eval(args, model, tokenizer)
        prof.step()

    test2(model)
    
    quantize(args, model, tokenizer)
    torch.compile(model.model)

    # test(model, tokenizer)

    eval(args, model, tokenizer)
    # eval(args, model, tokenizer)
    log_dir = "./log2"
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
        record_shapes=True,
        with_stack=True
    ) as prof:
        eval(args, model, tokenizer)
        prof.step()

    test2(model)
    
if __name__ == '__main__':
    main()



# TODO
# attn, mlp, norm, linearごとでtimeをとる