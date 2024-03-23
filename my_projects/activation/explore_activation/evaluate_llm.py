import torch
import numpy as np
import pandas as pd
torch.set_grad_enabled(False)
import sys
sys.path.append('../')
from utils.trace_utils import TraceDict2
import matplotlib.pyplot as plt
import os
import seaborn as sns
from transformers import AutoConfig, AutoTokenizer
from counterfact import CounterFactDataset
import torch
from utils.evaluation_lm_eval import run_eval_harness
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM

def modify_1(token_id, layer_ids):
    def modify_output(output, layer_name, inputs):
        # current_layer = int(layer_name.split(".")[2])
        # if current_layer == edit_layer:
        #     if isinstance(output, tuple):
        #         output[0][:, idx] += fv_vector.to(device)
        #         return output
        #     else:
        #         return output
        # else:
        #     return output
        return output

    def modify_input(input, layer_name:str):
        if layer_name.find('wte') != -1:
            pass
            #print(layer_name)
        elif layer_name.find('lm_head') != -1:
            pass
            #print(layer_name)
        else:
            # print(layer_name)
            for layer_id in layer_ids:
                if str(layer_id) in layer_name.split('.'):
            # heads_range = range(n_heads)
                    input[:, :, 1:, token_id] = 0
            # sum_input = torch.unsqueeze(torch.sum(input, dim=-1), dim=-1)
            # sum_input[:,:,0,:] = 1
            # input = input / sum_input
        return input

    return modify_output, modify_input

def load_gpt_model_and_tokenizer(model_name: str, device='cuda', low_cpu_mem_usage=False):
    """
    Loads a huggingface model and its tokenizer

    Parameters:
    model_name: huggingface name of the model to load (e.g. GPTJ: "EleutherAI/gpt-j-6B", or "EleutherAI/gpt-j-6b")
    device: 'cuda' or 'cpu'

    Returns:
    model: huggingface model
    tokenizer: huggingface tokenizer
    MODEL_CONFIG: config variables w/ standardized names

    """
    assert model_name is not None

    print("Loading: ", model_name)

    if model_name == 'gpt2-xl':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=low_cpu_mem_usage).to(device)

        MODEL_CONFIG = {"n_heads": model.config.n_head,
                        "n_layers": model.config.n_layer,
                        "resid_dim": model.config.n_embd,
                        "name_or_path": model.config.name_or_path,
                        "attn_hook_names": [f'transformer.h.{layer}.attn.c_proj' for layer in
                                            range(model.config.n_layer)],
                        "layer_hook_names": [f'transformer.h.{layer}' for layer in range(model.config.n_layer)]
                        }

    elif 'gpt-j' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=low_cpu_mem_usage).to(device)

        layer_hook_names = [f'transformer.h.{layer}.attn.attn_dropout' for layer in range(model.config.n_layer)]
        # layer_hook_names.append('transformer.wte')
        # layer_hook_names.append('lm_head')
        MODEL_CONFIG = {"n_heads": model.config.n_head,
                        "n_layers": model.config.n_layer,
                        "resid_dim": model.config.n_embd,
                        "name_or_path": model.config.name_or_path,
                        "attn_hook_names": [f'transformer.h.{layer}.attn.out_proj' for layer in
                                            range(model.config.n_layer)],
                        "layer_hook_names": layer_hook_names
                        }

    elif 'gpt-neox' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

        MODEL_CONFIG = {"n_heads": model.config.num_attention_heads,
                        "n_layers": model.config.num_hidden_layers,
                        "resid_dim": model.config.hidden_size,
                        "name_or_path": model.config.name_or_path,
                        "attn_hook_names": [f'gpt_neox.layers.{layer}.attention.dense' for layer in
                                            range(model.config.num_hidden_layers)],
                        "layer_hook_names": [f'gpt_neox.layers.{layer}' for layer in
                                             range(model.config.num_hidden_layers)]}

    elif 'llama' in model_name.lower():
        if '70b' in model_name.lower():
            # use quantization. requires `bitsandbytes` library
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            tokenizer = LlamaTokenizer.from_pretrained(model_name)
            model = LlamaForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                quantization_config=bnb_config
            )
        else:
            if '7b' in model_name.lower():
                model_dtype = torch.float32
            else:  # half precision for bigger llama models
                model_dtype = torch.float16
            tokenizer = LlamaTokenizer.from_pretrained(model_name)
            model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=model_dtype).to(device)

        MODEL_CONFIG = {"n_heads": model.config.num_attention_heads,
                        "n_layers": model.config.num_hidden_layers,
                        "resid_dim": model.config.hidden_size,
                        "name_or_path": model.config._name_or_path,
                        "attn_hook_names": [f'model.layers.{layer}.self_attn.o_proj' for layer in
                                            range(model.config.num_hidden_layers)],
                        "layer_hook_names": [f'model.layers.{layer}' for layer in
                                             range(model.config.num_hidden_layers)]}
    else:
        raise NotImplementedError("Still working to get this model available!")

    return model, tokenizer, MODEL_CONFIG


if __name__ == "__main__":
    device = 'cuda:2'
    model_name = 'EleutherAI/gpt-j-6b'  # # 'EleutherAI/gpt-j-6b' 'meta-llama/Llama-2-7b'
    task = None
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device, True)
    # result = run_eval_harness(model, tokenizer, "normal_gpt_j", ["winogrande"], torch.device(device), 4)
    sink_token = '\n'
    layer_ids = range(4,model_config['n_layers']-2)
    print(layer_ids)
    modify_output, modify_input = modify_1(token_id=0, layer_ids=layer_ids)
    with TraceDict2(model, layers=model_config['layer_hook_names'], edit_input=modify_input,
                    edit_output=modify_output, retain_output=False) as ret:
        result2 = run_eval_harness(model, tokenizer, "test_gpt_j",
                                   task,torch.device(device), 4, sink_token=sink_token)
    result3 = run_eval_harness(model, tokenizer, "normal_gpt_j",
                               task,torch.device(device), 4, sink_token=sink_token)
    print(result3)
    print(result2)