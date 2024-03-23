import pickle
import uuid
import glob
import os
from collections import defaultdict
import torch
import numpy as np

from utils import *
import copy
import umap
import seaborn as sns
import matplotlib.pyplot as plt

datalabels = {
    "emotion": ["sadness", "joy", "love", "anger", "fear"], # , "surprise"
    "language":['', 'en', 'ru', 'sl'],
    "math":['original', 'think']
}

def draw_umap(layer_activations, y, dataset_dict, model_name, layer_id, key, model_key):
    datalabel = datalabels[key]
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(layer_activations)
    # umap.plot.points(manifold, labels=y, theme="fire")
    plt.figure(figsize=(10, 10), dpi=80)
    y_array = np.array(y)
    for i, l in enumerate(datalabel):
        index = np.nonzero(y_array == i)
        plt.scatter(embedding[index, 0], embedding[index, 1], c=sns.color_palette()[i], label=l)
    plt.grid(linestyle="--", alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="best")  # legend有自己的参数可以控制图例位置
    plt.title(f"{model_name} - layer_id:{layer_id} - {model_key} - {dataset_dict[key]}")
    dirs = f'./umap/{model_name}/{key}'
    if not os.path.exists(dirs): os.makedirs(dirs)
    plt.savefig(os.path.join(dirs, f"{model_name}-layer_id:{layer_id}-{model_key}.png"))
    # plt.show()
def draw_matrix(layer_activations, y, text, dataset_dict, model_name, layer_id, key, model_key):
    # print(layer_activations)
    index = np.argsort(layer_activations, axis=0)
    hidden_state_num = index.shape[1]
    results = np.zeros_like(index)
    scores = np.zeros((1, hidden_state_num))
    text_list = []
    label = np.array(y)
    for i in range(hidden_state_num):
        ind = index[:, i]
        results[:, i] = label[ind]
        list_ = [text[ind[j]] for j in range(ind.shape[0])]
        text_list.append(list_)
    # print(text_list)

    for i in range(1, index.shape[0]):
        score_tmp = np.where(results[i-1] == results[i], 1, 0)
        scores = scores + score_tmp
    # plt.figure(figsize=(30, 10), dpi=80)
    # ax = plt.matshow(results, cmap=plt.cm.Reds)
    # plt.colorbar(ax.colorbar, fraction=0.025)
    # plt.title(f"{model_name} - {model_key} - layer_id:{layer_id} - {dataset_dict[key]}")
    # dirs = f'./sort_pic/{model_name}/{key}'
    # if not os.path.exists(dirs): os.makedirs(dirs)
    # plt.savefig(os.path.join(dirs, f"{model_name}-{model_key}-layer_id:{layer_id}.png"))
    # plt.show()
    return scores/(index.shape[0]-1), results.T, text_list

def draw_statistic(newH,model_name,title):
    plt.figure(figsize=(30, 10), dpi=80)
    colors = sns.color_palette("hls", 4)
    x = range(hidden_state.shape[0])

    mean = torch.mean(newH, dim=1)
    plt.plot(x, mean.numpy(), label=f"mean", c=colors[0])
    std = torch.std(newH, dim=1)
    plt.plot(x, std.numpy(), label=f"std", c=colors[1])
    max_ = torch.max(newH, dim=1)
    plt.plot(x, max_.values.numpy(), label=f"max_", c=colors[2])
    min_ = torch.min(newH, dim=1)
    plt.plot(x, min_.values.numpy(), label=f"min_", c=colors[3])

    plt.grid(linestyle="--", alpha=0.5)
    plt.legend(loc="best")  # legend有自己的参数可以控制图例位置
    plt.title(f"{model_name}- {title}")
    dirs = f'./stat/{model_name}/'
    if not os.path.exists(dirs): os.makedirs(dirs)
    plt.savefig(os.path.join(dirs, f"{model_name}-{title}.png"))
    plt.show()


def draw_bar(top_text, top, pd_index,model_name, key, model_key,dataset_dict, layer_id):
    import pandas as pd
    dirs = f'./sort_bar/{model_name}/{key}/layer{layer_id}'
    if not os.path.exists(dirs): os.makedirs(dirs)
    text_pd = pd.DataFrame(top_text, columns=pd_index)
    result_pd = pd.DataFrame(top, columns=pd_index)
    text_pd.to_csv(os.path.join(dirs, f"{pd_index[0]}-{pd_index[-1]}-text.csv"))
    result_pd.to_csv(os.path.join(dirs, f"{pd_index[0]}-{pd_index[-1]}-label.csv"))

    x = range(results.shape[0])
    label_nums = len(datalabels[key])
    colors = sns.color_palette("hls", label_nums)

    for i in range(label_nums):
        plt.figure(figsize=(30, 10), dpi=80)
        sum_ = np.sum(top == i, axis=1)
        sum_index = np.argmax(sum_)
        plt.bar(x, sum_, label=datalabels[key][i], color=colors[i])
        plt.legend(loc="best")  # legend有自己的参数可以控制图例位置
        title = f"{model_name}-{model_key}-{dataset_dict[key]}-layer{layer_id}-label:{i}-{datalabels[key][i]}-max index:{sum_index}"
        plt.title(title)

        plt.savefig(os.path.join(dirs, f"{model_name}-{model_key}-{datalabels[key][i]}{i}-{pd_index[0]}-{pd_index[-1]}.png"))
        plt.show()
        print(f"label:{i}-{datalabels[key][i]}; sort:{pd_index[0]}-{pd_index[-1]}; max index:{sum_index}")
        print(f"labels: {top[sum_index]}")
        print("sentences:")
        for line in top_text[sum_index]:
            print(line)
        print("++++++++++++++++++++++++")

if __name__ == "__main__":
    dataset_dict = {
        'emotion': "dair-ai/emotion",
        "math": 'camel-ai/math',
        # 'paraphrase': 'embedding-data/WikiAnswers',
        'language': 'opus_wikipedia'
    }
    module_keys = {# 'mlp': {"text": [], "label": [], "activation": []},
                   # 'attn': {"text": [], "label": [], "activation": []},
                   'hidden_states': {"text": [], "label": [], "activation": []}}
    for key in ["language"]: # dataset_dict.keys():

        model_list = {'gpt2-xl': copy.deepcopy(module_keys),
                      # 'gpt2-large': copy.deepcopy(module_keys),
                      # 'gpt2-medium': copy.deepcopy(module_keys)
                      } # ,  'EleutherAI/gpt-j-6b',
        for model_name, module_keys in model_list.items():
            files = glob.glob(os.path.join(f'./hiddenStates/{model_name}/{key}', "*.dat"))
            for file_i, file in enumerate(files):
                if file_i > 100:
                    break
                with open(file, 'rb') as f:
                    obj = pickle.load(f)
                    for model_key, module_dict in module_keys.items():
                        module_dict["label"].append(obj.label)
                        module_dict["text"].append(obj.text)
                        hidden_state = obj.activation[model_key].cpu()
                        sentence_hidden_state = torch.mean(hidden_state, dim=1)
                        module_dict["activation"].append(sentence_hidden_state)

                        # newH = hidden_state[:,-1] #hidden_state.reshape(hidden_state.shape[0],-1)
                        # # hidden_state.reshape(hidden_state.shape[0],-1)  hidden_state[:,-1]
                        # draw_statistic(newH, model_name, "1")
                        # print(newH)

        # print(model_list)
        top_k = 10
        for model_name, module_keys in model_list.items():
            for model_key, module_dict in module_keys.items():
                activations = torch.stack(module_dict["activation"], dim=0)
                layer_nums = activations.shape[1]
                score_list = []
                for layer_id in sorted(range(layer_nums-1), reverse=True):
                    layer_activations = activations[:, layer_id, :].cpu().numpy()
                    y = module_dict["label"]
                    text = module_dict['text']
                    # draw_umap(layer_activations, y, dataset_dict, model_name, layer_id, key, model_key)
                    scores, results, text_list = draw_matrix(layer_activations, y, text, dataset_dict, model_name, layer_id, key, model_key)

                    pd_index = range(0, top_k)
                    top = results[:, :top_k]
                    top_text = [text_list[i][:top_k] for i in range(len(text_list))]
                    draw_bar(top_text, top, pd_index, model_name, key, model_key, dataset_dict, layer_id)

                    pd_index = range(results.shape[1] - top_k, results.shape[1])
                    top = results[:, -top_k:]
                    top_text = [text_list[i][-top_k:] for i in range(len(text_list))]
                    draw_bar(top_text, top, pd_index, model_name, key, model_key, dataset_dict, layer_id)

                    score_list.append(scores)
                    print(f"finish {model_name} - layer_id:{layer_id} - {model_key} - {dataset_dict[key]}")

                plt.figure(figsize=(30, 10), dpi=80)
                # scores = np.stack(score_list, axis=0)
                colors = sns.color_palette("hls", len(score_list))
                max_index = [0, 0, 0]
                for i, l in enumerate(score_list):
                    maxi = np.max(l)
                    if maxi > max_index[1]:
                        max_index = [np.argmax(l), maxi, i]
                    l = l.reshape(-1)
                    x = range(l.shape[0])
                    plt.scatter(x, l, c=colors[i])
                    # plt.plot(x, l, c=colors[i], label=f"layer{i}")
                plt.scatter(max_index[0], max_index[1], s=200, label=f"{max_index[0]}-layer{max_index[2]}", marker='^')
                plt.grid(linestyle="--", alpha=0.5)
                plt.legend(loc="best")  # legend有自己的参数可以控制图例位置
                plt.title(f"{model_name} - {model_key} - {dataset_dict[key]}")
                dirs = f'./sort_scores/{model_name}/{key}'
                if not os.path.exists(dirs): os.makedirs(dirs)
                plt.savefig(os.path.join(dirs, f"{model_name}-{model_key}.png"))
                # plt.show()


        print(f'finish {key}')