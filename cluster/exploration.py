import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from tabulate import tabulate
from tqdm import tqdm, trange
from copy import deepcopy
import numpy as np
from collections import Counter
from datasets import load_dataset

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


model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
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

W_Q_heads = W_Q.reshape(num_layers, hidden_dim, num_heads, head_size).permute(0, 2, 1, 3)
W_K_heads = W_K.reshape(num_layers, hidden_dim, num_heads, head_size).permute(0, 2, 1, 3)
W_V_heads = W_V.reshape(num_layers, hidden_dim, num_heads, head_size).permute(0, 2, 1, 3)
W_O_heads = W_O.reshape(num_layers, num_heads, head_size, hidden_dim)
emb_inv = emb.T

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

i1, i2 = 23, 906
# i1, i2 = np.random.randint(num_layers), np.random.randint(d_int)

# print(i1, i2)
# print(tabulate([*zip(
#     top_tokens((K_heads[i1, i2]) @ emb, k=30, only_from_list=tokens_list, only_alnum=False),
#     top_tokens((V_heads[i1, i2]) @ emb, k=30, only_from_list=tokens_list, only_alnum=False),
#     # top_tokens((-K_heads[i1, i2]) @ emb, k=200, only_from_list=tokens_list),
#     # top_tokens((-V_heads[i1, i2]) @ emb, k=200, only_from_list=tokens_list),
# )], headers=['K', 'V', '-K', '-V']))


def approx_topk(mat, min_k=500, max_k=250_000, th0=10, max_iters=10, verbose=False):
    _get_actual_k = lambda th, th_max: torch.nonzero((mat > th) & (mat < th_max)).shape[0]
    th_max = np.inf
    left, right = 0, th0
    while True:
        # s = (mat > right) & (mat < th_max)
        actual_k = _get_actual_k(right, th_max)
        if verbose:
            print(f"one more iteration. {actual_k}")
        if actual_k <= max_k:
            break
        left, right = right, right * 2
    if min_k <= actual_k <= max_k:
        th = right
    else:
        for _ in range(max_iters):
            mid = (left + right) / 2
            actual_k = _get_actual_k(mid, th_max)
            if verbose:
                print(f"one more iteration. {actual_k}")
            if min_k <= actual_k <= max_k:
                break
            if actual_k > max_k:
                left = mid
            else:
                right = mid
        th = mid
    s = torch.nonzero((mat > th) & (mat < th_max))
    s = s.tolist()
    return torch.nonzero((mat > th) & (mat < th_max)).tolist()

def get_top_entries(tmp, all_high_pos, only_ascii=False, only_alnum=False, exclude_same=False, exclude_fuzzy=False, tokens_list=None):
    remaining_pos = all_high_pos
    if only_ascii:
        remaining_pos = [*filter(
            lambda x: (tokenizer.decode(x[0]).strip('Ġ▁').isascii() and tokenizer.decode(x[1]).strip('Ġ▁').isascii()),
            remaining_pos)]
    if only_alnum:
        remaining_pos = [*filter(
            lambda x: (tokenizer.decode(x[0]).strip('Ġ▁ ').isalnum() and tokenizer.decode(x[1]).strip('Ġ▁ ').isalnum()),
            remaining_pos)]
    if exclude_same:
        remaining_pos = [*filter(
            lambda x: tokenizer.decode(x[0]).lower().strip() != tokenizer.decode(x[1]).lower().strip(),
            remaining_pos)]
    if exclude_fuzzy:
        remaining_pos = [*filter(
            lambda x: not _fuzzy_eq(tokenizer.decode(x[0]).lower().strip(), tokenizer.decode(x[1]).lower().strip()),
            remaining_pos)]
    if tokens_list:
        remaining_pos = [*filter(
            lambda x: ((tokenizer.decode(x[0]).strip('Ġ▁').lower().strip() in tokens_list) and
                       (tokenizer.decode(x[1]).strip('Ġ▁').lower().strip() in tokens_list)),
            remaining_pos)]

    pos_val = tmp[[*zip(*remaining_pos)]]
    good_cells = [*map(lambda x: (tokenizer.decode(x[0]), tokenizer.decode(x[1])), remaining_pos)]
    good_tokens = list(map(lambda x: Counter(x).most_common(), zip(*good_cells)))
    remaining_pos_best = np.array(remaining_pos)[torch.argsort(pos_val if reverse_list else -pos_val)[:50]]
    good_cells_best = [*map(lambda x: (tokenizer.decode(x[0]), tokenizer.decode(x[1])), remaining_pos_best)]
    # good_cells[:100]
    # list(zip(good_tokens[0], good_tokens[1]))
    return good_cells_best

# i1, i2 = 23, 9
# W_V_tmp, W_O_tmp = W_V_heads[i1, i2, :], W_O_heads[i1, i2]
# tmp = (emb_inv @ (W_V_tmp @ W_O_tmp) @ emb)
# all_high_pos = approx_topk(tmp, th0=1, verbose=True) # torch.nonzero((tmp > th) & (tmp < th_max)).tolist()
# exclude_same = False
# reverse_list = False
# only_ascii = True
# only_alnum = False
# res = get_top_entries(tmp, all_high_pos, only_ascii=only_ascii, only_alnum=only_alnum,  exclude_same=exclude_same, tokens_list=None)
# print(res)

i1, i2 = 6, 2152
from sklearn.manifold import TSNE
import pandas as pd
import plotly.express as px


def _calc_df(vector, k, coef, normalized, tokenizer):
    mat = emb
    if normalized:
        mat = F.normalize(mat, dim=-1)
    dot = vector @ mat
    sol = torch.topk(dot * coef, k=k).indices  # np.argsort(dot * coef)[-k:]
    pattern = mat[:, sol].T
    scores = coef * dot[sol]
    # labels = tokenizer.batch_decode(sol)
    labels = convert_to_tokens(sol, tokenizer=tokenizer)
    X_embedded = TSNE(n_components=3,
                      learning_rate=10,
                      init='pca',
                      perplexity=3).fit_transform(pattern)

    df = pd.DataFrame(dict(x=X_embedded.T[0], y=X_embedded.T[1], z=X_embedded.T[2], label=labels, score=scores))
    return df


def plot_embedding_space(vector, is_3d=False, add_text=False, k=100, coef=1, normalized=False, tokenizer=None):
    df = _calc_df(vector, k=k, coef=coef, normalized=normalized, tokenizer=tokenizer)
    kwargs = {}
    scatter_fn = px.scatter
    if add_text:
        kwargs.update({'text': 'label'})
    if is_3d:
        scatter_fn = px.scatter_3d
        kwargs.update({'z': 'z'})
    fig = scatter_fn(
        data_frame=df,
        x='x',
        y='y',
        custom_data=["label", "score"],
        color="score", size_max=1, **kwargs)

    fig.update_traces(
        hovertemplate="<br>".join([
            "ColX: %{x}",
            "ColY: %{y}",
            "label: %{customdata[0]}",
            "score: %{customdata[1]}"
        ])
    )

    if add_text:
        fig.update_traces(textposition='middle right')
    fig.show()
plot_embedding_space(K_heads[i1][i2], tokenizer=tokenizer, normalized=False)