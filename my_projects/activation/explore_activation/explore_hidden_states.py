import copy
import os
import glob
import pickle

import numpy as np
import torch
import sys
sys.path.append('../utils/')
# from actiavtion.utils import utils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap

def collect_info(path, module_keys):
    files = glob.glob(os.path.join(path, "*.dat"))
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

def plot_umap(activations, y_array, datalabel, title, dir_path, show=False):
    embedding = umap.UMAP().fit_transform(activations)
    # umap.plot.points(manifold, labels=y, theme="fire")
    plt.subplots(figsize=(10, 10), dpi=80)

    colors = sns.color_palette('hls', len(datalabel))
    for i, l in enumerate(datalabel):
        index = np.nonzero(y_array == i)
        plt.scatter(embedding[index, 0], embedding[index, 1], c=colors[i], label=l)
    plt.grid(linestyle="--", alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="best")  # legend有自己的参数可以控制图例位置
    plt.title(title)
    if dir_path:
        dirs = dir_path
        if not os.path.exists(dirs): os.makedirs(dirs)
        plt.savefig(os.path.join(dirs, f"{title}_umap.png"))
    if show:
        plt.show()

def get_score_label(activations, y_array, text):
    index = np.argsort(activations, axis=0)
    hidden_state_num = index.shape[1]
    results = np.zeros_like(index)
    text_list = []
    label = y_array
    for i in range(hidden_state_num):
        ind = index[:, i]
        results[:, i] = label[ind]
        list_ = [text[ind[j]] for j in range(ind.shape[0])]
        text_list.append(list_)
    return results.T, text_list

def plot_socre_label(results, top, top_text, dirs, datalabel, pd_index, title, show=False ):
    if dirs:
        if not os.path.exists(dirs): os.makedirs(dirs)
    text_pd = pd.DataFrame(top_text, columns=pd_index)
    result_pd = pd.DataFrame(top, columns=pd_index)
    text_pd.to_csv(os.path.join(dirs, f"{pd_index[0]}-{pd_index[-1]}-text.csv"))
    result_pd.to_csv(os.path.join(dirs, f"{pd_index[0]}-{pd_index[-1]}-label.csv"))

    x = range(results.shape[0])
    label_nums = len(datalabel)
    colors = sns.color_palette("hls", label_nums)

    max_index_list = []
    for i in range(label_nums):
        plt.figure(figsize=(30, 10), dpi=200)
        sum_ = np.sum(top == i, axis=1)
        sum_index = np.argmax(sum_)
        max_index_list.append(sum_index)
        plt.bar(x, sum_, label=datalabel[i], color=colors[i])
        plt.legend(loc="best")  # legend有自己的参数可以控制图例位置
        plt.title(f'{title}_label:{i}_{datalabel[i]}_max index:{sum_index}')
        if dirs:
            plt.savefig(os.path.join(dirs, f"{title}_label:{i}_{datalabel[i]}_{pd_index[0]}_{pd_index[-1]}.png"))
        if show:
            plt.show()
        # print(
        #     f"label:{i}-{datalabels[key][i]}; sort:{pd_index[0]}-{pd_index[-1]}; max index:{sum_index}")
        # print(f"labels: {top[sum_index]}")
        # print("sentences:")
        # for line in top_text[sum_index]:
        #     print(line)
        # print("++++++++++++++++++++++++")

    return result_pd, text_pd, max_index_list

def plot_line(x, max_pd, label_list, label, layer_id, y_lim=None, title='', dirs=None, show=False):
    plt.figure(figsize=(30, 10), dpi=200)
    max_pd = max_pd.T
    label_num = max_pd.shape[1]
    colors = sns.color_palette("hls", label_num)
    for i in range(label_num):
        plt.plot(x, max_pd.iloc[:, i], label=f"{label_list[i]}-{label}-{layer_id}", c=colors[i], alpha=.5)
    plt.title(f"{title}")
    if y_lim:
        plt.ylim(y_lim)
    plt.legend(loc="best")
    if dirs:
        if not os.path.exists(dirs): os.makedirs(dirs)
        plt.savefig(os.path.join(dirs, f"{title}_scores.png"))
    if show:
        plt.show()

def plot_common_max_min(data_pd, layer_id, top_k, label_list ,title='', dirs=None ,show=False):
    label_data_pd = data_pd.groupby(['label'])
    groups = label_data_pd.indices
    colors = sns.color_palette("hls", (len(groups.keys()) + 1) * 2)
    plt.figure(figsize=(30, 8), dpi=200)
    stat = []
    for label_i, sentence_index in groups.items():
        sentence_unit = data_pd.iloc[sentence_index, :-1]
        # print(sentence_unit)
        units = sentence_unit.values
        sorts = np.argsort(units, axis=1)

        def get_count(t):
            stats_top = np.zeros((units.shape[-1]))
            for t_i in range(t.shape[0]):
                stats_top[t[t_i]] = stats_top[t[t_i]] + 1
            return stats_top / t.shape[0]

        stats_top = get_count(t=sorts[:, :top_k])
        stats_last = get_count(t=sorts[:, -top_k:])
        stat.append((label_i, label_list[label_i], stats_top, stats_last))
        pos = range(1, sorts.shape[-1] + 1)
        plt.bar(pos, stats_top, color=colors[label_i * 2], label=f"{label_list[label_i]}-layer{layer_id}-top{top_k}",
                alpha=.5)
        plt.bar(pos, -stats_last, color=colors[label_i * 2 + 1],
                label=f"{label_list[label_i]}-layer{layer_id}-last{top_k}", alpha=.5)
    plt.legend(loc="best")
    plt.title(title)
    if dirs:
        if not os.path.exists(dirs): os.makedirs(dirs)
        plt.savefig(os.path.join(dirs, f"{title}_common_max_min.png"))
    if show:
        plt.show()
    return stat

if __name__ == "__main__":


    datalabels = {
        "emotion": ["sadness", "joy", "love", "anger", "fear", "surprise"],  # , "surprise"
        "language": ['', 'en', 'ru', 'sl'],
        "math": ['original', 'think']
    }
    dataset_dict = {
        'emotion': "dair-ai_emotion",
        "math": 'camel-ai_math',
        # 'paraphrase': 'embedding-data/WikiAnswers',
        'language': 'opus_wikipedia'
    }
    module_keys_m = {  # 'mlp': {"text": [], "label": [], "activation": []},
        # 'attn': {"text": [], "label": [], "activation": []},
        'hidden_states': {"text": [], "label": [], "activation": []}}
    top_k = 10
    for key in [ "math", "emotion", "language",]:  # dataset_dict.keys():
        label_list = datalabels[key]
        model_list = {'gpt2-xl': copy.deepcopy(module_keys_m),
                      # 'gpt2-large': copy.deepcopy(module_keys_m),
                      # 'gpt2-medium': copy.deepcopy(module_keys_m)
                      }  # ,  'EleutherAI/gpt-j-6b',
        for model_name, module_keys in model_list.items():
            path = f'../get_activation/hiddenStates/{model_name}/{key}'
            collect_info(path, module_keys)
            # print(module_keys)
            for layer_id in range(22, 24):
                # layer_id = 20
                for key_ in module_keys.keys():
                    d = module_keys[key_]
                    activations = torch.stack(d['activation']).cpu().numpy()[:,layer_id,:]
                    # construct dataframe
                    unit_num = activations.shape[-1]
                    sentence_num = activations.shape[0]
                    data_pd = pd.DataFrame(activations,
                                 index=[f"sentence{i}" for i in range(sentence_num)],
                                 columns=[f"unit{i}" for i in range(unit_num)])
                    data_pd['label'] = d['label']
                    # data_pd['text'] = d['text']

                    dir_path = f'./result/activation/{key}/{model_name}/layer{layer_id}'

                    # draw umap
                    y_array = np.array(d['label'])
                    title = f"{model_name}_layer_id:{layer_id}_{dataset_dict[key]}"
                    datalabel = datalabels[key]
                    plot_umap(activations, y_array, datalabel, title, dir_path)

                    # draw score bar
                    text = d['text']
                    results, text_list = get_score_label(activations, y_array, text)
                    # title = f"{model_name}_{dataset_dict[key]}_layer{layer_id}"
                    dirs = f"{dir_path}/score_label"

                    pd_index = range(0, top_k)
                    top = results[:, :top_k]
                    top_text = [text_list[i][:top_k] for i in range(len(text_list))]
                    result_top_pd, text_top_pd, max_index_list = plot_socre_label(results, top, top_text, dirs, datalabel, pd_index, title)

                    pd_index = range(results.shape[1] - top_k, results.shape[1])
                    top = results[:, -top_k:]
                    top_text = [text_list[i][-top_k:] for i in range(len(text_list))]
                    result_last_pd, text_last_pd, max_index_list = plot_socre_label(results, top, top_text, dirs, datalabel, pd_index, title)

                    # draw score values
                    x = range(unit_num)
                    # max_pd = data_pd.groupby(['label']).max()
                    # plot_line(x, max_pd, label_list, "max", layer_id)
                    # min_pd = data_pd.groupby(['label']).min()
                    # plot_line(x, min_pd, label_list,"min", layer_id)
                    # std_pd = data_pd.groupby(['label']).std()
                    # plot_line(x, std_pd, label_list, "std", layer_id)
                    mean_pd = data_pd.groupby(['label']).mean()
                    plot_line(x, mean_pd, label_list, "mean", layer_id, None, title, dir_path, False)
                    # print(min_pd)

                    # draw common
                    stat = plot_common_max_min(data_pd, layer_id, top_k, label_list, title, dir_path)
                    standard_top, standard_last = 0.8, 0.8
                    for label_i, label_name, stat_top, stat_last in stat:
                        top_index = np.nonzero(stat_top>standard_top)
                        last_index = np.nonzero(stat_last>standard_last)
                        print(f'{label_name}:')
                        print(result_top_pd.iloc[top_index])
                        print(text_top_pd.iloc[top_index])
                        print(result_last_pd.iloc[last_index])
                        print(text_last_pd.iloc[last_index])
                    print(stat)




