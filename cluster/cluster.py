from collections import Counter
import numpy as np
from sklearn.cluster import MiniBatchKMeans,AffinityPropagation,DBSCAN
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
import collections
import sklearn

sns.set_palette('colorblind')
cmap = sns.color_palette('colorblind')

device = 'cpu'#'cuda:2'
model = AutoModelForCausalLM.from_pretrained('gpt2-large').to(device)
tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
def Parametr_Extraction(model):
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
    return K_heads,V_heads,W_V_heads,W_Q_heads,W_K_heads,W_O_heads

K_heads,V_heads,W_V_heads,W_Q_heads,W_K_heads,W_O_heads = Parametr_Extraction(model)

def cluster(model,v,name):
    num_layers = model.config.n_layer
    for i in range(0,num_layers):
        data = np.array(v[i]) * 1000

        # algorithm = AffinityPropagation(random_state=100, max_iter=1000,damping=0.8)
        # algorithm.fit(data)
        # y_pred = algorithm.labels_
        # clusters = algorithm.cluster_centers_
        # data_count1 = collections.Counter(y_pred)
        # print(f"{name}-num{len(data_count1.keys())}-AffinityPropagation-layer{i}-{data_count1}")
        # np.savez(f"./results4/AffinityPropagation-layer{i}-{name}", y_pred=y_pred, clusters=clusters)

        # bandwidth = cluster.estimate_bandwidth(data, quantile=0.1, n_samples=1000)
        # algorithm = cluster.MeanShift(bandwidth=bandwidth)
        # algorithm.fit(data)
        # y_pred = algorithm.labels_
        # clusters = algorithm.cluster_centers_
        # data_count1 = collections.Counter(y_pred)
        # print(f"MeanShift-layer{i}-{data_count1}")
        # np.savez(f"./results2/MeanShift-layer{i}-v", y_pred=y_pred,clusters=clusters)
        #
        algorithm = MiniBatchKMeans(n_clusters=int(data.shape[0]/5),
                                    max_iter=1000, random_state=0,)
        algorithm.fit(data)
        y_pred = algorithm.labels_
        clusters = algorithm.cluster_centers_
        data_count1 = collections.Counter(y_pred)
        print(f"MiniBatchKMeans-layer{i}-{name}-{data_count1}")
        np.savez(f"./results5/MiniBatchKMeans-layer{i}-{name}", y_pred=y_pred, clusters=clusters)
        #

        # algorithm = DBSCAN(eps=0.3, min_samples=10)
        # algorithm.fit(data)
        # y_pred = algorithm.labels_
        # # clusters = algorithm.cluster_centers_
        # data_count1 = collections.Counter(y_pred)
        # print(f"DBSCAN-layer{i}-{data_count1}")
        # np.savez(f"./results_cos/DBSCAN-layer{i}-{name}", y_pred=y_pred)

        # from sklearn.metrics import pairwise_distances
        # distance_matrix = pairwise_distances(data, data, metric='cosine', n_jobs=-1)
        # tsne = manifold.TSNE(
        #     n_components=3,
        #     random_state=0,
        #     n_iter=1000,
        #     # init="random",
        #     metric="mahalanobis"
        # )
        # Y = tsne.fit_transform(data)
        # np.savez(f"./results_cos/tsne-layer{i}-{name}", y_pred=Y)
        # plt.scatter(Y[red, 0], Y[red, 1], c="r")

hidden_dim = model.config.n_embd
num_layers = model.config.n_layer
v = V_heads.numpy()
k = K_heads.numpy()
W_V = W_V_heads.permute(0, 1, 3, 2).reshape(num_layers,-1,hidden_dim).permute(0, 2, 1).numpy()
W_Q = W_Q_heads.permute(0, 1, 3, 2).reshape(num_layers,-1,hidden_dim).permute(0, 2, 1).numpy()
W_K = W_K_heads.permute(0, 1, 3, 2).reshape(num_layers,-1,hidden_dim).permute(0, 2, 1).numpy()
W_O = W_O_heads.reshape(num_layers,- 1, hidden_dim).numpy()
cluster(model,v,"v")
cluster(model,k,"k")
cluster(model,W_V,"W_V")
cluster(model,W_Q,"W_Q")
cluster(model,W_K,"W_K")
cluster(model,W_O,"W_O")


# data = np.array(v.reshape(-1,v.shape[-1]))
#
# tsne = manifold.TSNE(
#     n_components=2,
#     n_iter=10000,
#     perplexity=10
# )
# Y = tsne.fit_transform(data)
# np.savez(f"./results2/tsne-v", y_pred=Y)

# data = np.array([[1, 2], [1, 4], [1, 0],
#           [4, 2], [4, 0], [4, 4],
#           [4, 5], [0, 1], [2, 2],
#           [3, 2], [5, 5], [1, -1]])

# algorithm = MiniBatchKMeans(n_clusters=10000,
#                           random_state=0,
#                           n_init="auto")
# bandwidth = cluster.estimate_bandwidth(data, quantile=0.3)
# algorithm = cluster.MeanShift(bandwidth=2)
#
# algorithm.fit(data)
# y_pred = algorithm.labels_
# np.savez(f"./results/MeanShift", y_pred=y_pred)