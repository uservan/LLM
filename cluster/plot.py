from collections import Counter
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from copy import deepcopy
from tqdm.auto import tqdm, trange
import re
import torch
from torch import nn
import torch.nn.functional as F
from collections import defaultdict
from utils import load_imdb
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn import cluster, datasets, mixture
from sklearn import datasets, manifold
import matplotlib.colors as mcolors
import collections
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from datasets import load_dataset
from tabulate import tabulate

# Parameter Extraction
device = 'cpu'#'cuda:2'
model = AutoModelForCausalLM.from_pretrained("gpt2-large")
tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
my_tokenizer = tokenizer
emb = model.get_output_embeddings().weight.data.T.detach()

num_layers = model.config.n_layer
num_heads = model.config.n_head
hidden_dim = model.config.n_embd
head_size = hidden_dim // num_heads

K = torch.cat([model.get_parameter(f"transformer.h.{j}.mlp.c_fc.weight").T
                           for j in range(num_layers)]).detach()
V = torch.cat([model.get_parameter(f"transformer.h.{j}.mlp.c_proj.weight")
                           for j in range(num_layers)]).detach()

W_Q, W_K, W_V = torch.cat([model.get_parameter(f"transformer.h.{j}.attn.c_attn.weight")
                           for j in range(num_layers)]).detach().chunk(3, dim=-1)
W_O = torch.cat([model.get_parameter(f"transformer.h.{j}.attn.c_proj.weight")
                           for j in range(num_layers)]).detach()
K_heads = K.reshape(num_layers, -1, hidden_dim)
V_heads = V.reshape(num_layers, -1, hidden_dim)
d_int = K_heads.shape[1]

W_Q_heads = W_Q.reshape(num_layers, -1, head_size).permute(0, 2, 1) #.permute(0, 2, 1, 3)
W_K_heads = W_K.reshape(num_layers, -1, head_size).permute(0, 2, 1) #.permute(0, 2, 1, 3)
W_V_heads = W_V.reshape(num_layers, -1, head_size).permute(0, 2, 1) #.reshape(num_layers, hidden_dim, num_heads, head_size).permute(0, 2, 1, 3)
W_O_heads = W_O.reshape(num_layers, -1, head_size) #.reshape(num_layers, num_heads, head_size, hidden_dim)
emb_inv = emb.T

# project function
ALNUM_CHARSET = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
def convert_to_tokens(indices, tokenizer, extended=False, extra_values_pos=None, strip=True):
    if extended:
        res = [tokenizer.convert_ids_to_tokens([idx])[0] if idx < len(tokenizer) else
               (f"[pos{idx - len(tokenizer)}]" if idx < extra_values_pos else f"[val{idx - extra_values_pos}]")
               for idx in indices]
    else:
        res = tokenizer.convert_ids_to_tokens(indices)
    if strip:
        res = list(map(lambda x: x[1:] if x[0] == 'Ġ' else "#" + x, res))
    return res
def top_tokens(v, k=100, tokenizer=None, only_alnum=False, only_ascii=True, with_values=False,
               exclude_brackets=False, extended=True, extra_values=None, only_from_list=None):
    if tokenizer is None:
        tokenizer = my_tokenizer
    v = deepcopy(v)
    ignored_indices = []
    if only_ascii:
        ignored_indices.extend([key for val, key in tokenizer.vocab.items() if not val.strip('Ġ▁').isascii()])
    if only_alnum:
        ignored_indices.extend(
            [key for val, key in tokenizer.vocab.items() if not (set(val.strip('Ġ▁[] ')) <= ALNUM_CHARSET)])
    if only_from_list:
        ignored_indices.extend(
            [key for val, key in tokenizer.vocab.items() if val.strip('Ġ▁ ').lower() not in only_from_list])
    if exclude_brackets:
        ignored_indices = set(ignored_indices).intersection(
            {key for val, key in tokenizer.vocab.items() if not (val.isascii() and val.isalnum())})
        ignored_indices = list(ignored_indices)

    ignored_indices = list(set(ignored_indices))
    v[ignored_indices] = -np.inf
    extra_values_pos = len(v)
    if extra_values is not None:
        v = torch.cat([v, extra_values])
    values, indices = torch.topk(v, k=k)
    res = convert_to_tokens(indices, tokenizer, extended=extended, extra_values_pos=extra_values_pos)
    if with_values:
        res = list(zip(res, values.cpu().numpy()))
    return res

# token list
tokens_list = set()
imdb = load_dataset('imdb')['train']['text']
max_tokens_num = None

if max_tokens_num is None:
    tokens_list = set()
    for txt in tqdm(imdb):
        tokens_list = tokens_list.union(set(tokenizer.tokenize(txt)))
else:
    tokens_list = Counter()
    for txt in tqdm(imdb):
        tokens_list.update(set(tokenizer.tokenize(txt)))
    tokens_list = map(lambda x: x[0], tokens_list.most_common(max_tokens_num))
tokens_list = set([*map(lambda x: x.strip('Ġ▁').lower(), tokens_list)])


# draw
colors = mcolors.CSS4_COLORS
names = list(colors)
names_ = ["v", "k","W_V","W_Q","W_K","W_O"]
def get_heads(n):
    if n == "v":
        return V_heads
    if n == "k":
        return K_heads
    if n == "W_V":
        return W_V_heads
    if n == "W_Q":
        return W_Q_heads
    if n == "W_K":
        return W_K_heads
    if n == "W_O":
        return W_O_heads

## draw 3D with clusters
for i in range(0,num_layers):
    fig = plt.figure(figsize=(50, 8), dpi=80)
    for name_i, name in enumerate(names_):
        ax = fig.add_subplot(3, 2, name_i+1, projection='3d')
        data = np.load(f"./results3/tsne-layer{i}-{name}.npz")
        results = data['y_pred']

        data1 = np.load(f"./results5/MiniBatchKMeans-layer{i}-{name}.npz")
        y_pred = data1['y_pred']
        cluster = data1['clusters']
        data_count1 = collections.Counter(y_pred)
        print(f"AffinityPropagation-layer{i}-{name}-{data_count1}")
        for k_i, k in enumerate(data_count1.keys()):
            mask = (y_pred == k)
            ax.scatter(results[mask, 0], results[mask, 1], results[mask, 2],c=names[(k_i+1 )% len(names)])
        ax.set_title(f"tsne-layer{i}-{name}")
        # ax.text2D(0.5, 1, f"tsne-layer{i}-{name}", transform=ax.transAxes)
        print(f"tsne-layer{i}-{name}")
    plt.show()


## project clusters
for name in names_:
    for i in range(0,num_layers):
        data1 = np.load(f"./results4/AffinityPropagation-layer{i}-{name}.npz")
        y_pred = data1['y_pred']
        cluster = data1['clusters']
        data_count1 = collections.Counter(y_pred)
        print(f"AffinityPropagation-layer{i}-{name}-{data_count1}")
        for k_i, k in enumerate(data_count1.keys()):
            heads = get_heads(name)[i]
            mask = (y_pred == k)
            mask_x = torch.from_numpy(mask).resize(heads.shape[0], 1).expand(heads.shape)
            output_x = torch.masked_select(heads, mask=mask_x).reshape(-1, heads.shape[1])[:3]
            c = torch.from_numpy(cluster[k].reshape(1,-1))
            hs = torch.concat((c, output_x), dim=0)
            if hs.shape[0] > 3 and data_count1.get(k) < 500 :
                print(f"__________________layer{i}-class{k}-num{data_count1.get(k)}_____________________")
                for num in range(hs.shape[0]):
                    list_name = 'cluster' if num == 0 else f"{name}-{num}"
                    print(f"{list_name}-{top_tokens((hs[num]) @ emb, k=30, only_from_list=tokens_list, only_alnum=False)}")
                print(f"__________________layer{i}-class{k}-num{data_count1.get(k)}_____________________")



for name in names_:
    for i in range(0,num_layers):
        data1 = np.load(f"./results4/AffinityPropagation-layer{i}-{name}.npz")
        y_pred = data1['y_pred']
        cluster = data1['clusters']
        data_count1 = collections.Counter(y_pred)
        print(f"AffinityPropagation-layer{i}-{name}-{data_count1}")

## draw 3D
colors_ = ['r','b','g','y','c','m','k','c']
for i in range(0, num_layers):
    for name in names_:
        fig = plt.figure(figsize=(20, 8), dpi=80)
        ax = fig.add_subplot(111, projection='3d')
        # ax = Axes3D(fig)
        data = np.load(f"./results_cos/tsne-layer{i}-{name}.npz")
        results = data['y_pred']
        x = results[:, 0]
        y = results[:, 1]
        z = results[:, 2]
        # plt.scatter(results[:, 0], results[:, 1], c=names[i])
        ax.scatter(x, y, z)
        ax.set_title(f"tsne-layer{i}-{name}")
        # ax.text2D(0.5, 1, f"tsne-layer{i}-{name}", transform=ax.transAxes)
        print(f"tsne-layer{i}-{name}")
        plt.show()

## draw each tsne
fig = plt.figure(figsize=(20,8),dpi=80)
# ax = Axes3D(fig)
for i in range(0,2):
    data = np.load(f"./results2/tsne-layer{i}-v.npz")
    results = data['y_pred']
    x = results[:, 0]
    y = results[:, 1]
    # z = results[:, 2]
    plt.scatter(results[:, 0], results[:, 1], c=names[i])
    # ax.scatter(x, y, z)
plt.show()

## add clusters
fig = plt.figure(figsize=(20,8),dpi=80)
# ax = Axes3D(fig)
for i in range(0,2):
    data1 = np.load(f"./results2/MeanShift-layer{i}-v.npz")['y_pred']
    data_count1 = collections.Counter(data1)
    data = np.load(f"./results2/tsne-layer{i}-v.npz")
    results = data['y_pred']
    for k_i , k in enumerate(data_count1.keys()):
        mask = data1 == k
        x = results[:, 0]
        y = results[:, 1]
        # z = results[:, 2]
        plt.scatter(results[mask, 0], results[mask, 1], c=names[k_i])
    # ax.scatter(x, y, z)
    plt.show()


plt.figure(figsize=(20,8),dpi=80)
data = np.load(f"./results/tsne-v.npz")
results = data['y_pred']
plt.scatter(results[:, 0], results[:, 1])
plt.show()



