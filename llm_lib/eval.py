# Code adapted from https://github.com/locuslab/wanda/blob/main/lib/eval.py

# Import necessary modules
import time
import torch
import torch.nn as nn

# Import get_loaders function from data module within the same directory
from .data import get_loaders 
from datasets import load_from_disk

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
@torch.no_grad()
def eval_ppl(model, tokenizer, device=torch.device("cuda:0"), datasets=["wikitext2", "ptb", "c4"]):
    if not hasattr(model, 'seqlen'): model.seqlen = 2048
    results = {}
    for dataset in datasets:
        if dataset == "wikitext2":
            testdata = load_from_disk("./data/wikitext_test")
            if testdata is not None:
                testloader = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
                
        else:
          _, testloader = get_loaders(
              dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer
          )
        # Print status
        print(f"evaluating on {dataset}")
        # Evaluate ppl in no grad context to avoid updating the model
        with torch.no_grad():
            results[dataset] = eval_ppl_wikitext(model, testloader, 1, device)
    for key, value in results.items():
        print(f"{key}: {str(value)}", flush=True)
    return results 

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    nlls = []
    print(f"nsamples {nsamples}")

    for i in range(0,nsamples,bs):

        j = min(i+bs, nsamples)

        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        lm_logits = model(inputs).logits

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        nlls.append(neg_log_likelihood)
        
        # print(i, torch.exp(torch.stack(nlls).sum() / ((i+1) * model.seqlen)).item(), flush=True)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    torch.cuda.empty_cache()

    return ppl.item()


def run_lm_eval(model_name, save_model, task_list=["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"], 
        num_fewshot=0, use_accelerate=False, add_special_tokens=False):
    from lm_eval import tasks, evaluator 
    task_manager = tasks.TaskManager()
    task_names = task_manager.match_tasks(task_list)
    print(task_names)
    model_args = f"pretrained={save_model}"
    limit = None 
    if "70b" in model_name or "65b" in model_name:
        limit = 2000
    if use_accelerate:
        model_args = f"pretrained={save_model},use_accelerate=True"
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=None,
        device=None,
        limit=limit
    )
    
    return results['results']