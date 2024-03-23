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
from utils import *

device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')
max_len = 256

layers_to_get = ['mlp', 'attn']

# dataset
# camel-ai/math
# embedding-data/WikiAnswers
# opus_wikipedia
# dair-ai/emotion  datalabel = ["sadness", "joy", "love", "anger", "fear", "surprise"]

dataset_dict = {
    'emotion': "dair-ai/emotion",
    "math": 'baber/hendrycks_math',
    'paraphrase': 'embedding-data/WikiAnswers',
    'language': 'opus_wikipedia'
}

model_name = 'gpt2-large' #'gpt2-xl', 'gpt2-large', 'gpt2-medium' 'EleutherAI/gpt-j-6b',
# model
model = AutoModelForCausalLM.from_pretrained(model_name)
model.requires_grad_(False)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# dataset
begin = 45

# emotion
datasets = load_dataset("dair-ai/emotion", split=f"test[{begin}:201]")
key = 'emotion'
for i, d in enumerate(datasets):
    line = d['text']
    activation, results = get_activation(model, tokenizer, device, layers_to_get, line, max_len)
    a = Emotion_Activation_Save(activation, results, d['text'], d['label'])
    print(f"{begin+i}-{model_name} finished: {line}")
    dirs = f'./hiddenStates/{model_name}/{key}'
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    with open(os.path.join(dirs,f'{uuid.uuid4().hex}.dat'), 'wb') as f:
        pickle.dump(a, f)


# math
# datasets = load_dataset('baber/hendrycks_math', "number_theory", split=f"test[{begin}:201]")
# datasets = datasets.filter(lambda example: len(example["problem"]) < 200)
# key = 'math'
# for i, d in enumerate(datasets):
#     line = d['problem']
#     activation, results = get_activation(model, tokenizer, device, layers_to_get, line, max_len)
#     a = Emotion_Activation_Save(activation, results, line, 0)
#     dirs = f'./hiddenStates/{model_name}/{key}'
#     if not os.path.exists(dirs):
#         os.makedirs(dirs)
#     with open(os.path.join(dirs,f'{uuid.uuid4().hex}.dat'), 'wb') as f:
#         pickle.dump(a, f)
#     print(f"{begin + i}-{model_name} finished: {line}")
#
#     line = "Let's think step by step:" + line
#     activation, results = get_activation(model, tokenizer, device, layers_to_get, line, max_len)
#     a = Emotion_Activation_Save(activation, results, line, 1)
#     with open(os.path.join(dirs,f'{uuid.uuid4().hex}.dat'), 'wb') as f:
#         pickle.dump(a, f)
#     print(f"{begin + i}-{model_name} finished: {line}")


# language
# en_ru_datasets = load_dataset('opus_wikipedia', "en-ru", split=f"train[{begin}:35]")
# en_sl_datasets = load_dataset('opus_wikipedia', "en-sl", split=f"train[{begin}:35]")
#
# language_label = {'en':1, 'ru':2, 'sl':3}
# key = 'language'
# for i in range(35):
#     line = en_sl_datasets[i]['translation']['sl']
#     activation, results = get_activation(model, tokenizer, device, layers_to_get, line, max_len)
#     a = Emotion_Activation_Save(activation, results, line, language_label['sl'])
#     dirs = f'./hiddenStates/{model_name}/{key}'
#     if not os.path.exists(dirs):
#         os.makedirs(dirs)
#     with open(os.path.join(dirs,f'{uuid.uuid4().hex}.dat'), 'wb') as f:
#         pickle.dump(a, f)
#
#     line = en_ru_datasets[i]['translation']['en']
#     activation, results = get_activation(model, tokenizer, device, layers_to_get, line, max_len)
#     a = Emotion_Activation_Save(activation, results, line, language_label['en'])
#     dirs = f'./hiddenStates/{model_name}/{key}'
#     if not os.path.exists(dirs):
#         os.makedirs(dirs)
#     with open(os.path.join(dirs, f'{uuid.uuid4().hex}.dat'), 'wb') as f:
#         pickle.dump(a, f)
#
#     line = en_ru_datasets[i]['translation']['ru']
#     activation, results = get_activation(model, tokenizer, device, layers_to_get, line, max_len)
#     a = Emotion_Activation_Save(activation, results, line, language_label['ru'])
#     dirs = f'./hiddenStates/{model_name}/{key}'
#     if not os.path.exists(dirs):
#         os.makedirs(dirs)
#     with open(os.path.join(dirs, f'{uuid.uuid4().hex}.dat'), 'wb') as f:
#         pickle.dump(a, f)
#     print(f"{begin + i }-{model_name} finished: {line}")











