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


def draw_matrix(layer_activations, y, dataset_dict, model_name, layer_id, key, model_key):
    index = np.argsort(layer_activations, axis=0)
    results = np.zeros_like(index)
    label = np.array(y)
    for i in range(results.shape[0]):
        results[i] = label[index[i]]
    plt.figure(figsize=(30, 10), dpi=80)
    ax = plt.matshow(results, cmap=plt.cm.Reds)
    plt.colorbar(ax.colorbar, fraction=0.025)
    plt.title(f"{model_name} - layer_id:{layer_id} - {model_key} - {dataset_dict[key]}")
    plt.show()
    dirs = f'./sort_pic/{model_name}/{key}'
    if not os.path.exists(dirs): os.makedirs(dirs)
    plt.savefig(os.path.join(dirs, f"{model_name}-layer_id:{layer_id}-{model_key}.png"))

if __name__ == "__main__":
    dataset_dict = {
        'emotion': "dair-ai/emotion",
        "math": 'camel-ai/math',
        'paraphrase': 'embedding-data/WikiAnswers',
        'language': 'opus_wikipedia'
    }

    key = 'emotion'
    module_keys = {'mlp': {"text": [],"label": [],"activation": []},
                   'attn': {"text": [],"label": [],"activation": []},
                   'hidden_states': {"text": [] ,"label": [], "activation": []}}
    model_list = {'gpt2-xl': [],
                  'gpt2-large': [],
                  'gpt2-medium': []} # ,  'EleutherAI/gpt-j-6b',
    for model_name, module_keys in model_list.items():
        model_config = AutoConfig.from_pretrained(model_name)
        files = glob.glob(os.path.join(f'./hiddenStates/{model_name}/{key}', "*.dat"))
        for file_i, file in enumerate(files):
            with open(file, 'rb') as f:
                obj = pickle.load(f)
                attentions = obj.activation['attentions']
                token_num = attentions[0].shape[-1]
                for token_i in range(attentions[0].shape[-1]):
                    heads = torch.cat(attentions, dim=0)[:, :, token_i, :].reshape(model_config.n_layer, -1).cpu()
                    heads_show = heads # torch.exp(heads * 1)/torch.exp(torch.ones([1]))
                    o = heads_show[0]
                    ax = plt.matshow(heads_show.numpy(), cmap=plt.cm.Reds, figsize=(30, 20), dpi=80)
                    plt.colorbar(ax.colorbar, fraction=0.025)
                    ax.set_title(f"{obj.text} - token_id:{token_i}-{model_name}")
                    plt.savefig(os.path.join(dirs, f"{file_i} - token_id:{token_i}-{model_name}.png"))
                    plt.show()
                    dirs = f'./attention/{model_name}/{key}'
                    if not os.path.exists(dirs): os.makedirs(dirs)

                print(obj)