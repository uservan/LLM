import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
import os
import random
from typing import *


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

        MODEL_CONFIG = {"n_heads": model.config.n_head,
                        "n_layers": model.config.n_layer,
                        "resid_dim": model.config.n_embd,
                        "name_or_path": model.config.name_or_path,
                        "attn_hook_names": [f'transformer.h.{layer}.attn.out_proj' for layer in
                                            range(model.config.n_layer)],
                        "layer_hook_names": [f'transformer.h.{layer}.attn.attn_dropout' for layer in range(model.config.n_layer)]
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


def make_inputs(tokenizer, prompts, device="cuda"):
    token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )

def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]

def predict_from_input(model, inp):
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.max(probs, dim=1)
    return preds, p