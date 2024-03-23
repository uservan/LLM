from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
import plotly.graph_objects as go
import matplotlib as mpl
import plotly.io as pio
from transformers import GPT2Config,AutoConfig
from datasets import *
import pickle
import uuid
import glob

class Emotion_Activation_Save:
    def __init__(self, activation, results, text, label):
        self.activation = activation
        self.results = results
        self.text = text
        self.label =label
def extract_hs_include_prefix(list_inputs, list_outputs, info='', max_len=256):
    def hook(module, input, output):
        # print(module)
        if list_inputs is not None:
            # print('input[0].shape', input[0].shape, f'[{info}]')
            last_tokens = input[0].clone().detach().squeeze().cpu()
            while len(last_tokens.shape) > 2:
                last_tokens = last_tokens[0]

            # print('last_tokens.shape', last_tokens.shape, f'[{info}]')
            for last_token in last_tokens:
                last_token = last_token.squeeze().view(-1, 1)
                list_inputs.append(last_token)

                if len(list_inputs) > max_len:
                    list_inputs.pop(0)

        if list_outputs is not None:
            last_tokens = output[0].clone().detach().squeeze().cpu()
            while len(last_tokens.shape) > 2:
                last_tokens = last_tokens[0]

            # print('last_tokens.shape', last_tokens.shape, f'[{info}]')
            for last_token in last_tokens:
                last_token = last_token.squeeze().view(-1, 1)
                # print('last_token.shape', last_token.shape, f'[{info}]')
                list_outputs.append(last_token)

                if len(list_inputs) > max_len:
                    list_inputs.pop(0)

    return hook

def get_activation(model, tokenizer, device, layers_to_get, line, max_len):
    module_dict = dict(model.named_modules())
    layers_to_check = [
        '.mlp', '.mlp.c_fc', '.mlp.c_proj',
        '.attn', '.attn.c_attn', '.attn.c_proj',
        '.ln_1', '.ln_2', '', ]
    hs_collector = {}

    for layer_idx in range(model.config.n_layer):
        for layer_type in layers_to_check:
            list_inputs = []
            list_outputs = []

            layer_with_idx = f'{layer_idx}{layer_type}'
            layer_pointer = module_dict[f"transformer.h.{layer_with_idx}"]
            layer_pointer.register_forward_hook(extract_hs_include_prefix(list_inputs=list_inputs,
                                                                          list_outputs=list_outputs,
                                                                          info=layer_with_idx,
                                                                          max_len=max_len))
            if layer_idx not in hs_collector:
                hs_collector[layer_idx] = {}

            layer_key = layer_type.strip('.')
            if layer_key not in hs_collector[layer_idx]:
                hs_collector[layer_idx][layer_key] = {}

            hs_collector[layer_idx][layer_key]['input'] = list_inputs
            hs_collector[layer_idx][layer_key]['output'] = list_outputs

    model.to(device)
    model.eval()
    encoded_line = tokenizer.encode(line.rstrip(), return_tensors='pt').to(device)
    output_and_cache = model(encoded_line, output_hidden_states=True, output_attentions=True, use_cache=True)

    outputs = dict()
    outputs['past_key_values'] = output_and_cache.past_key_values # the "attentnion memory"
    outputs['attentions'] = output_and_cache.attentions
    outputs['hidden_states'] = torch.cat(output_and_cache.hidden_states, dim=0)[1:].cpu()
    for layer_type in layers_to_get:
        layer_list = []
        for layer_idx in range(model.config.n_layer):
            hidden_state_list = hs_collector[layer_idx][layer_type]['output']
            h = torch.cat(hidden_state_list, dim=-1).permute(1, 0).cpu()
            layer_list.append(h)
        outputs[layer_type] = torch.stack(layer_list, dim=0)

    # the final answer token is:
    r = tokenizer.decode(output_and_cache.logits[0, :, :].argmax(dim=-1).tolist())
    # model_answer = tokenizer.decode(output_and_cache.logits[0, -1, :].argmax().item())
    # print(f'model_answer: "{model_answer}"')
    return outputs, r

def draw_atten(heads_show,token_i,model_name, dir ,title, codes):
    plt.figure(figsize=(30, 30), dpi=80)
    ax = plt.matshow(heads_show.numpy().T, cmap=plt.cm.Reds)
    plt.colorbar(ax.colorbar, fraction=0.2)
    plt.yticks(np.arange(len(codes)), codes)
    plt.title(f"-token_id:{token_i}-{model_name}")
    dirs = os.path.join(f'./test/{model_name}', dir)
    if not os.path.exists(dirs): os.makedirs(dirs)
    plt.savefig(os.path.join(dirs, f"{title}-token_id:{token_i}-{model_name}.png"))
    # plt.show()

if __name__ == "__main__":
    device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')

    # example 1
    model_name = 'gpt2-xl' # 'gpt2-xl' #'gpt2-large' #'gpt2-medium'  'EleutherAI/gpt-j-6b'

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.requires_grad_(False)
    max_len = 256

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    layers_to_get = ['mlp', 'attn']

    line = 'When Mary and John went to the store, John gave a drink to'
    # '1120 adds 5520 equals to' 'When Mary and John went to the store, John gave a drink to'
    # 'The Space Needle is in downtown' 'Beats Music is owned by'
    target_token = 'Mary' #  '6640' 'Mary' 'Seattle' 'Apple'
    activation, results = get_activation(model, tokenizer, device, layers_to_get, line, max_len)

    encoded_line = tokenizer.encode(line.rstrip())
    codes = tokenizer.convert_ids_to_tokens(encoded_line)

    attentions = activation['attentions']
    token_num = attentions[0].shape[-1]
    for token_i in range(10, attentions[0].shape[-1]):
        heads = torch.cat(attentions, dim=0)[:, :, token_i, :].cpu()
        heads_show = torch.mean(heads, dim=1)
        o = heads_show[0]
        draw_atten(heads_show,token_i,model_name, target_token,"cat-heads", codes)

        heads_show = torch.where(heads_show>=(1/(token_i+1)), 1, 0)
        draw_atten(heads_show, token_i, model_name, target_token,"cat-heads-standard", codes)

        head_num = heads.shape[1]
        heads_show = heads.reshape(heads.shape[0], -1) # torch.exp(heads * 1)/torch.exp(torch.ones([1]))
        o = heads_show[0]
        new_codes = []
        for i, c in enumerate(codes):
            new_codes.append(c)
            for _ in range(head_num - 1):
                new_codes.append("")
        draw_atten(heads_show, token_i, model_name, target_token,"each-heads", new_codes)

        heads_show = torch.where(heads_show >= (1 / (token_i + 1)), 1, 0)
        draw_atten(heads_show, token_i, model_name, target_token,"each-heads-standard", new_codes)

    print(activation,results)