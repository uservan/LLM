from collections import Counter
import numpy as np
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

sns.set_palette('colorblind')
cmap = sns.color_palette('colorblind')

device = 'cuda:2'
model = AutoModelForCausalLM.from_pretrained('gpt2-large').to(device)
tokenizer = AutoTokenizer.from_pretrained('gpt2-large')

### function
def dist_matmul(A, B, dist_B=False):
    if dist_B:
        return dist_matmul(B.T, A.T, dist_B=False).T
    B_module = nn.Linear(*B.shape).to(B.device)
    with torch.no_grad():
        nn.init.zeros_(B_module.bias)
        B_module.weight.set_(B.T)
    B_module = nn.DataParallel(B_module)
    res = B_module(A)
    del B_module
    return res
def get_token_freqs(sents):
    token_probs_dict = Counter()
    for s in tqdm(sents):
        for t in tokenizer.encode(s):
            token_probs_dict[t] += 1
    return token_probs_dict
def _batched_recall_metric(xs, y, k=20, k2=3, only_ascii=False):
    if k2 is None:
        k2 = k
    out = []
    xs_idx = torch.topk(xs, k=k, dim=-1).indices
    y_idx = torch.topk(y, k=k2, dim=-1).indices
    for i in range(len(y)):
        A = set(xs_idx[i].ravel().cpu().tolist())
        B = set(y_idx[i].cpu().tolist())
        if only_ascii:
            A, B = map(lambda X: {x for x in X if tokenizer.decode(x).isascii()}, (A, B))
        out.append(len(A & B) / len(B))
    return out
def _batched_sim(x, y, k=100):
    out = []
    x_idx = torch.topk(x, k=k, dim=-1).indices
    y_idx = torch.topk(y, k=k, dim=-1).indices
    for i in range(len(y)):
        A, B = set(x_idx[i].cpu().tolist()), set(y_idx[i].cpu().tolist())
        out.append(len(A & B) / len(A | B))
    return out
def _plot_comparison(tuples_by_layers, figsize=(15, 9), plots_per_row=2, legend=True):
    fig = plt.figure(figsize=figsize)
    for i, (real, fake, param_name) in enumerate(tuples_by_layers):
        plt.subplot(len(tuples_by_layers) // plots_per_row, plots_per_row, i + 1)
        plt.title(param_name)
        ax = sns.barplot(x='x', y='y', hue='hue',
                    data={'x': [*np.arange(num_layers), *np.arange(num_layers)],
                          'y': [*real, *fake],
                          'hue': (['aligned'] * num_layers) + (['random'] * num_layers),
                    }
                   )
        ax.legend_.remove()
        # plt.xlabel("layer")
    handles, labels = ax.get_legend_handles_labels()
    if legend:
        fig.legend(handles, labels, loc='lower center')


### Parameter Extraction
emb = model.get_output_embeddings().weight.data.T
print(emb)
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

W_V_heads = W_V.reshape(num_layers, hidden_dim, num_heads, head_size).permute(0, 2, 1, 3)
W_O_heads = W_O.reshape(num_layers, num_heads, head_size, hidden_dim)
W_Q_heads = W_Q.reshape(num_layers, hidden_dim, num_heads, head_size).permute(0, 2, 1, 3)
W_K_heads = W_K.reshape(num_layers, hidden_dim, num_heads, head_size).permute(0, 2, 1, 3)


sents = load_imdb()
sents = [s for s in sents if len(s) > 100]

ln_f = model.transformer.ln_f
emb_cpu = deepcopy(emb).cpu()
ln_f_cpu = deepcopy(ln_f).cpu()
emb_func = lambda x: ln_f_cpu(x) @ emb_cpu
E1 = emb

# ### Memorize Intermediate States
# mem = defaultdict(list)
# def _memorize_inp_outp(sm, only_last=None):
#     global tokenizer
#
#     def f(m, inp, outp):
#         inp = inp[0]
#         outp = outp[0]
#         if only_last:
#             inp, outp = inp[-only_last:], outp[-only_last:]
#         inp, outp = (inp.squeeze(), outp.squeeze())
#
#         if 'attn.c_proj' in sm:
#             mem[f"intermediate-{sm}"].append((inp.cpu().detach(), outp.cpu().detach()))
#             inp = inp.reshape(inp.shape[0], num_heads, head_size)
#             layer_idx = int(re.search(r'\.h\.(\d+)\.', sm).group(1))
#             inp_ = torch.empty(inp.shape[0], inp.shape[1], hidden_dim)
#             for head_idx in range(inp.shape[1]):
#                 inp_[:, head_idx] = (inp[:, head_idx].unsqueeze(1) @ W_O_heads[layer_idx][head_idx]).squeeze()
#             inp = inp_
#         mem[sm].append((inp.cpu().detach(), outp.cpu().detach()))
#
#     return f
# submodules = [
#     *[f'transformer.h.{j}.attn' for j in range(num_layers)],
#     *[f'transformer.h.{j}.attn.c_proj' for j in range(num_layers)],
#     *[f'transformer.h.{j}' for j in range(num_layers)],
#     *[f'transformer.h.{j}.mlp.c_fc' for j in range(num_layers)],
# ]
# if 'mem_hooks' in globals():
#     [h.remove() for h in mem_hooks]
# mem_hooks = []
# for sm in submodules:
#     mem_hooks.append(model.get_submodule(sm).register_forward_hook(_memorize_inp_outp(sm)))
#
# sent_stop = 1024
# num_sent_samples = 60
# sents_sample = np.random.choice(sents, size=num_sent_samples)
#
# for i in trange(num_sent_samples):
#     sampled_sent = sents_sample[i] # sents[i] #
#     sampled_sent = ' '.join(sampled_sent.split(' ')[:sent_stop])
#     model(**{k: v.to(device) if isinstance(v, torch.Tensor) else v
#                       for k, v in tokenizer(sampled_sent, return_tensors='pt', truncation=True).items()})
#
#
# ## Experiments
# max_explored_tokens = 1000
# k_ = 50
# k2 = k_
# only_ascii = False
#
# ### Hidden States and Parameters
# real_K_recall, fake_K_recall, real_V_recall, fake_V_recall = [], [], [], []
#
# for i1 in trange(num_layers):
#     sum_lens = sum([len(x[0]) for x in mem[f'transformer.h.0.mlp.c_fc']])
#     tmp1_pre = torch.topk(torch.cat([y for _, y in mem[f'transformer.h.{i1}.mlp.c_fc']])[:max_explored_tokens],
#                           k=5, dim=-1).indices
#     tmp1_pre_perm = torch.topk(
#         torch.cat([y for _, y in mem[f'transformer.h.{i1}.mlp.c_fc']])[torch.randperm(sum_lens)[:max_explored_tokens]],
#                                k=5, dim=-1).indices
#     tmp1_K, tmp1_V = K_heads[i1][tmp1_pre] @ E1, V_heads[i1][tmp1_pre] @ E1
#     tmp1_K_perm, tmp1_V_perm = K_heads[i1][tmp1_pre_perm] @ E1, V_heads[i1][tmp1_pre_perm] @ E1
#     tmp2 = emb_func(torch.cat([y for _, y in mem[f'transformer.h.{i1}']])[:max_explored_tokens])
#
#     fake_K_recall.append(np.mean(_batched_recall_metric(tmp1_K_perm, tmp2, k=k_, k2=k2, only_ascii=only_ascii)))
#     real_K_recall.append(np.mean(_batched_recall_metric(tmp1_K, tmp2, k=k_, k2=k2, only_ascii=only_ascii)))
#     fake_V_recall.append(np.mean(_batched_recall_metric(tmp1_V_perm, tmp2, k=k_, k2=k2, only_ascii=only_ascii)))
#     real_V_recall.append(np.mean(_batched_recall_metric(tmp1_V, tmp2, k=k_, k2=k2, only_ascii=only_ascii)))
#
# real_WV_recall, fake_WV_recall, real_WO_recall, fake_WO_recall = [], [], [], []
#
# for i1 in trange(num_layers):
#     tmp1_pre = torch.topk(torch.cat(
#         [x for x, _ in mem[f'intermediate-transformer.h.{i1}.attn.c_proj']])[:max_explored_tokens], k=5, dim=-1).indices
#     tmp1_x, tmp1_y = np.unravel_index(tmp1_pre.numpy(), (num_heads, head_size))
#     del tmp1_pre
#     tmp1_WV, tmp1_WO = W_V_heads[i1, tmp1_x, :, tmp1_y] @ E1, W_O_heads[i1, tmp1_x, tmp1_y] @ E1
#     tmp1 = tmp1_WV
#     tmp2 = emb_func(torch.cat([y for _, y in mem[f'transformer.h.{i1}']])[:max_explored_tokens])
#
#     fake_WV_recall.append(np.mean(_batched_recall_metric(tmp1_WV[torch.randperm(len(tmp1_WV))], tmp2, k=k_, k2=k2,
#                                                          only_ascii=only_ascii)))
#     real_WV_recall.append(np.mean(_batched_recall_metric(tmp1_WV, tmp2, k=k_, k2=k2, only_ascii=only_ascii)))
#     fake_WO_recall.append(np.mean(_batched_recall_metric(tmp1_WO[torch.randperm(len(tmp1_WO))], tmp2, k=k_, k2=k2,
#                                                          only_ascii=only_ascii)))
#     real_WO_recall.append(np.mean(_batched_recall_metric(tmp1_WO, tmp2, k=k_, k2=k2, only_ascii=only_ascii)))
#
# _plot_comparison([(real_K_recall, fake_K_recall, "$K$"),
#                   (real_V_recall, fake_V_recall, "$V$"),
#                   (real_WV_recall, fake_WV_recall, "$W_V$"),
#                   (real_WO_recall, fake_WO_recall, "$W_O$")
#                  ])
# plt.savefig("artifacts/weights_vs_states.pdf")
#
# ##Related Parameter Pairs are Connected
