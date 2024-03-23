import torch
import numpy as np
from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM,
                          GPTNeoForCausalLM, GPT2TokenizerFast,LlamaConfig)
import os
import random
from typing import *
import matplotlib.pyplot as plt
import os
import seaborn as sns
import torch.nn as nn
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer, BloomForCausalLM
from torch.utils.data import DataLoader
import sys
# print(sys.path)
import os
sys.path.append(os.path.join(sys.path[0], '../'))
import numpy as np
from utils.model_utils import load_gpt_model_and_tokenizer, make_inputs, decode_tokens, predict_from_input
from transformers import AutoConfig, AutoTokenizer
import matplotlib.pyplot as plt
import os
import seaborn as sns


# ## draw Attention
# device = 'cuda:1'
# model_name = 'EleutherAI/gpt-j-6b' #   # 'EleutherAI/gpt-j-6b' 'meta-llama/Llama-2-7b'
# model_config = AutoConfig.from_pretrained(model_name)
# n_layers = model_config.n_layer
# n_heads = model_config.n_head
# token_num = 10

# # model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device, True)
# # dataset = load_dataset('monology/pile-uncopyrighted', split='train[:1000]').filter(lambda text: len(text['text'])>token_num)  # gpt-neo
# # def process_func(examples):
# #     contents = [e + tokenizer.eos_token for e in examples["text"]]
# #     return tokenizer(contents, max_length=token_num, truncation=True)
# # tokenized_ds = dataset.map(process_func, batched=True, remove_columns=dataset.column_names)
# # dl = DataLoader(tokenized_ds, batch_size=1, collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False),
# #                 shuffle=True)
# # result_list = []
# # for batch in dl:
# #     if torch.cuda.is_available():
# #         batch = {k: v.to(torch.device(device)) for k, v in batch.items()}
# #     output_and_cache = model(**batch, output_hidden_states=True, output_attentions=True, use_cache=True)
# #     ground_attentions = torch.cat(output_and_cache.attentions, dim=0).cpu().detach()
# #     print(ground_attentions)
# #     result_list.append(ground_attentions)
# #
# # results = torch.stack(result_list, dim=0).numpy()
# # np.save('./result/attention.npy', results)

# attentions = np.load('./result/attention.npy')
# layer_id, head_id = 20, 15
# examples_num = 50
# show_attention = attentions[:examples_num,layer_id,head_id,-1, :]
# # print(np.sum(show_attention, axis=1))
# def plt_heatMap_sns(scores, save_path=None, title=None, cmap=None, y_ticks=None, x_ticks=None, show=None):
#     plt.subplots(figsize=(20, 20), dpi=200)
#     plt.rcParams['font.size'] = '10'
#     if cmap is None:
#         cmap = sns.color_palette("Reds", as_cmap=True)
#     if x_ticks and y_ticks:
#         sns.heatmap(scores, cmap=cmap,  xticklabels=x_ticks, yticklabels=y_ticks)
#     else:
#         sns.heatmap(scores, cmap=cmap)
#     if title is not None:
#         plt.title(title)
#     if save_path:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         plt.savefig(os.path.join(save_path, f'{title}.png'), bbox_inches="tight")
#     if show:
#         plt.show()
#     plt.close()


# x_ticks = [f"token{i + 1}" for i in range(token_num)]
# y_ticks = [f"examples{i + 1}" for i in range(examples_num)]
# plt_heatMap_sns(show_attention,title=f"layer{layer_id}_head{head_id}_attentions",
#                 x_ticks=x_ticks, y_ticks=y_ticks
#                 , show=True, save_path='./result/'
#                 )

from utils.evaluation_lm_eval import run_eval_harness
from gpt_neo import load_model

layers = []  # [23]
is_linear = True

device = 'cuda:1'
save_model_path = os.path.join(sys.path[0], f'./results_back/gpt_neo_{layers}_{is_linear}_{layers}')
model_name = 'EleutherAI/gpt-neo-1.3B'  # 'EleutherAI/gpt-neo-1.3B' # 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T' # 'EleutherAI/gpt-neo-125m'
model, tokenizer, _ = load_model(model_name, device=device, layers=layers, train_layers=layers, is_linear=is_linear)
# model.load_state_dict(torch.load(save_model_path+'pth'))
model = model.to(torch.device(device))

# result2 = run_eval_harness(model, tokenizer, "test_gpt_j",None, torch.device(device), 4, sink_token=None)

# model.eval()
# ipt = tokenizer("hello, I", return_tensors="pt").to(model.device)
# results = model.generate(**ipt, max_length=512, do_sample=True, eos_token_id=tokenizer.eos_token_id)[0]
# print(tokenizer.decode(results ,skip_special_tokens=True))

from transformers import pipeline
#文本生成
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device, pad_token_id=tokenizer.eos_token_id)
results= text_generator("As far as I am concerned, I will",
			   max_length=128,
			   do_sample=True)
print(results)