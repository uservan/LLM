import torch
from tqdm import tqdm
import os
import sys
project_dir = os.path.join(os.getcwd(),'other_projects','streaming-llm')
project_dir = os.path.join(os.getcwd(),'..')
sys.path.append(project_dir)
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from streaming_llm.kv_cache import StartRecentKVCache
from streaming_llm.utils import parse_args, load
import time
import yaml
import numpy as np

device = "cuda:1"
args = parse_args()

data = load_dataset(args.dataset_name, args.task, split=args.split)

model, tokenizer = load(args.model_name_or_path)

nlls = []
loss_fn = CrossEntropyLoss(reduction="none")
past_key_values = None

if args.enable_start_recent_kv_cache:
    if "llama" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
    elif "mpt" in model.config.model_type:
        v_seq_dim = 2
        k_seq_dim = 3
    elif "pythia" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
    elif "falcon" in model.config.model_type:
        v_seq_dim = 1
        k_seq_dim = 1
    else:
        raise ValueError(f"got {model.config.model_type}")
    kv_cache = StartRecentKVCache(
        start_size=args.start_size,
        recent_size=args.recent_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
    )
else:
    kv_cache = None

if args.enable_pos_shift:
    if "llama" in model.config.model_type:
        from streaming_llm.pos_shift.modify_llama import enable_llama_pos_shift_attention

        enable_llama_pos_shift_attention(model)
    elif "falcon" in model.config.model_type:
        from streaming_llm.pos_shift.modify_falcon import (
            enable_falcon_pos_shift_attention,
        )

        enable_falcon_pos_shift_attention(model)
    elif "gpt_neox" in model.config.model_type:
        from streaming_llm.pos_shift.modify_gpt_neox import (
            enable_gpt_neox_pos_shift_attention,
        )

        enable_gpt_neox_pos_shift_attention(model)
    elif "mpt" in model.config.model_type:
        pass
    else:
        raise ValueError(f"got {model.config.model_type}")

args.output_dir = os.path.join(args.output_dir, str(time.time()))
os.makedirs(os.path.join(project_dir,args.output_dir), exist_ok=True)
yaml.dump(args.__dict__ ,open(os.path.join(project_dir,f"{args.output_dir}/configuration.yml"), "w"))
f = open(os.path.join(project_dir,f"{args.output_dir}/log.txt"), "w")

percent_attn_dict = dict()
num_eval_tokens = 0
# print(type(data["text"][: args.num_samples]))
dataset = [args.sink_token] * args.start_size +data["text"][: args.num_samples] if args.sink_token else data["text"][: args.num_samples]
for text in dataset:
    encodings = tokenizer(text, return_tensors="pt")

    print(encodings.input_ids[:, :10])

    seq_len = encodings.input_ids.size(1)
    print(f"seq_len: {seq_len}")
    pbar = tqdm(range(0, seq_len - 1))

    for idx in pbar:
        input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
        with torch.no_grad():
            outputs = model(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                 output_attentions=True
            )
            logits = outputs.logits.view(-1, model.config.vocab_size)
            past_key_values = outputs.past_key_values
            label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
            neg_log_likelihood = loss_fn(logits, label)
            if kv_cache is not None:
                past_key_values = kv_cache(past_key_values)
        nlls.append(neg_log_likelihood)
        pbar.set_description(
            f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
        )
        print(neg_log_likelihood.item(), file=f, flush=True)
        num_eval_tokens += 1

        if args.percent_token_nums and num_eval_tokens in args.percent_token_nums:
            ground_attentions = torch.cat(outputs.attentions, dim=0).cpu().detach().numpy()[:,:,-1,:]
            attn_percent = np.mean(ground_attentions.reshape((-1, ground_attentions.shape[2])), axis=0)
            percent_attn_dict[num_eval_tokens] = {
                    'first token percent':float(attn_percent[0]),
                    'the most token':float(np.argmax(attn_percent)),
                    'the most token percent':float(np.max(attn_percent)),
                    'mean for each token': 1/num_eval_tokens
                    }
            yaml.dump(percent_attn_dict ,open(os.path.join(project_dir, f"{args.output_dir}/attn_percent.yml"), "w"))

        if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
            break
    if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
        break

f.close()

ppl = torch.exp(torch.stack(nlls).mean())
print(ppl.item())
with open(os.path.join(project_dir, f"{args.output_dir}/ppl.txt"), "w") as f:
    f.write(f"{ppl.item()}\n")
