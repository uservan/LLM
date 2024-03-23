import torch
import numpy as np
from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM,
                          GPTNeoForCausalLM, GPT2TokenizerFast,LlamaConfig, GPTJModel, AutoConfig, GPT2LMHeadModel)
import random
from typing import *
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer, BloomForCausalLM
from torch.utils.data import DataLoader

import sys
# print(sys.path)
import os
sys.path.append(os.path.join(sys.path[0], '../'))
from utils.trace_utils import TraceDict2
from baukit import TraceDict, Trace
from utils.evaluation_lm_eval import run_eval_harness
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import yaml
from copy import deepcopy

class MyGPTJMLP(nn.Module):
    def __init__(self,config):  # in MLP: intermediate_size= 4 * embed_dim
        super().__init__()
        self.config = config

    def forward(self, hidden_states: Optional[torch.FloatTensor]) -> torch.FloatTensor:
        device = hidden_states.device
        dtype = hidden_states.dtype
        results = torch.zeros_like(hidden_states).to(device=device, dtype=dtype)
        return results

class MyGPTJAttention(nn.Module):
    def __init__(self, config, is_linear):
        super().__init__()
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.linear = nn.Sequential(nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim*2, bias=False),
                                    nn.Linear(in_features=self.embed_dim*2, out_features=self.embed_dim, bias=False))
#         from torch.nn.init import xavier_uniform_
#         xavier_uniform_(self.linear[0].weight.data)
#         xavier_uniform_(self.linear[1].weight.data)
        self.is_linear = is_linear

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor, Tuple[torch.Tensor]],
        Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]],
    ]:
        bsz, q_len, _ = hidden_states.size()
        input_dtype = hidden_states.dtype
        if input_dtype != torch.float16:
            self.linear = self.linear.to(input_dtype)
        device = hidden_states

        key = torch.ones((bsz, q_len, self.num_heads, self.head_dim), dtype=input_dtype).to(device).transpose(1, 2)
        value = torch.ones((bsz, q_len, self.num_heads, self.head_dim), dtype=input_dtype).to(device).transpose(1, 2)

        attn_weights = torch.zeros((bsz, self.num_heads, q_len, q_len), dtype=input_dtype).to(device)

        attn_output = self.linear(hidden_states)
        p = self.linear.parameters()
        if self.is_linear == 'linear_atten':
            # attn_output = self.resid_dropout(attn_output)
            pass
        if self.is_linear == 'zero_atten':
            attn_output = torch.zeros_like(attn_output, dtype=input_dtype).to(device)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

class MyGPTJBlock(nn.Module):
    def __init__(self, config, block):
        super().__init__()
        self.ln_1 = block.ln_1
        self.attn = block.attn
        self.mlp = block.mlp

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states=hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions)

class MyGPTNeoBlock(nn.Module):
    def __init__(self, config, block):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.intermediate_size if config.intermediate_size is not None else 4 * hidden_size
        self.ln_1 = block.ln_1
        self.attn = block.attn
        self.ln_2 = block.ln_2
        self.mlp = block.mlp

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = attn_output + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)

class MyGPTNeoSelfAttention(nn.Module):
    def __init__(self, config, is_linear):
        super().__init__()

        max_positions = config.max_position_embeddings
        # bias = torch.tril(torch.ones((max_positions, max_positions), dtype=bool)).view(
        #     1, 1, max_positions, max_positions
        # )

        # local causal self attention is a sliding window where each token can only attend to the previous
        # window_size tokens. This is implemented by updating the causal mask such that for each token
        # all other tokens are masked except the previous window_size tokens.
        # if attention_type == "local":
        #     bias = torch.bitwise_xor(bias, torch.tril(bias, -config.window_size))

        # self.attn_dropout = nn.Dropout(float(config.attention_dropout))
        self.resid_dropout = nn.Dropout(float(config.resid_dropout))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.linear = nn.Sequential(nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim*2, bias=False),
                                    nn.Linear(in_features=self.embed_dim*2, out_features=self.embed_dim, bias=False))
#         from torch.nn.init import xavier_uniform_
#         xavier_uniform_(self.linear[0].weight.data)
#         xavier_uniform_(self.linear[1].weight.data)
        self.is_linear = is_linear

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):

        bsz, q_len, _ = hidden_states.size()
        input_dtype = hidden_states.dtype
        if input_dtype != torch.float16:
            self.linear = self.linear.to(input_dtype)
        self.device = hidden_states.device

        key = torch.ones((bsz, q_len, self.num_heads, self.head_dim), dtype=input_dtype).to(
            self.device).transpose(1, 2)
        value = torch.ones((bsz, q_len, self.num_heads, self.head_dim), dtype=input_dtype).to(
            self.device).transpose(1, 2)

        attn_weights = torch.zeros((bsz, self.num_heads, q_len, q_len), dtype=input_dtype).to(self.device)

        attn_output = self.linear(hidden_states)
        p = self.linear.parameters()
        if self.is_linear == 'linear_atten':
            # attn_output = self.resid_dropout(attn_output)
            pass
        if self.is_linear == 'zero_atten':
            attn_output = torch.zeros_like(attn_output, dtype=input_dtype).to(self.device)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

class MyGPTNeoMLP(nn.Module):
    def __init__(self, config):  # in MLP: intermediate_size= 4 * hidden_size
        super().__init__()

    def forward(self, hidden_states):
        device = hidden_states.device
        dtype = hidden_states.dtype
        results = torch.zeros_like(hidden_states).to(device=device, dtype=dtype)
        return results

class MyLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, device, is_linear):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        # self.resid_dropout = nn.Dropout(float(config.resid_dropout))
        self.linear = nn.Sequential(nn.Linear(in_features=2048, out_features=5632, bias=False, dtype=torch.float16),
                                                      nn.Linear(in_features=5632, out_features=2048, bias=False, dtype=torch.float16))
        self.device = device
        self.is_linear =is_linear
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )


    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()
        input_dtype = hidden_states.dtype
        if input_dtype != torch.float16:
            self.linear = self.linear.to(input_dtype)


        key_states = torch.zeros((bsz, q_len, self.num_key_value_heads, self.head_dim), dtype=input_dtype).to(self.device).transpose(1, 2)
        value_states = torch.zeros((bsz, q_len, self.num_key_value_heads, self.head_dim), dtype=input_dtype).to(self.device).transpose(1, 2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.zeros((bsz, self.num_heads, q_len, q_len), dtype=input_dtype).to(self.device)

        attn_output = self.linear(hidden_states)
        if not self.is_linear:
            attn_output = torch.zeros_like(attn_output, dtype=input_dtype).to(self.device)

        return attn_output, attn_weights, past_key_value

def plot_multi_umap(embeddings, datalabel, title, save_path, show=False):
    # umap.plot.points(manifold, labels=y, theme="fire")
    plt.subplots(figsize=(10, 10), dpi=80)
    marks = [ 'o', '*', '^', '2', '4','p','h','+','D','_','v','<','w','>']
    # colors = sns.color_palette('hls', len(datalabel))
    colors = sns.color_palette('coolwarm', len(datalabel))
    # for layer_id in range(old_shape[0]):
    for e_i, embedding in enumerate(embeddings):
        for i, l in enumerate(datalabel):
            plt.scatter(embedding[:, :, i, 0].reshape(-1), embedding[:, :, i, 1].reshape(-1), color=colors[i],label=l, marker=marks[e_i])
    plt.grid(linestyle="--", alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="best")  # legend有自己的参数可以控制图例位置
    if title is not None:
        plt.title(title)
    if save_path:
        dirs = save_path
        if not os.path.exists(dirs): os.makedirs(dirs)
        plt.savefig(os.path.join(dirs, f"{title}_umap.png"))
    if show:
        plt.show()

def plot_umap(embedding, datalabel, title, save_path, show=False):
    # umap.plot.points(manifold, labels=y, theme="fire")
    plt.subplots(figsize=(10, 10), dpi=80)
    marks = [ 'o', '>', '^', '2', '4','p','h','+','D','_','v','<','w','*']
    # colors = sns.color_palette('hls', len(datalabel))
    colors = sns.color_palette('coolwarm', len(datalabel))
    # for layer_id in range(old_shape[0]):
    for i, l in enumerate(datalabel):
        plt.scatter(embedding[:, :, i, 0].reshape(-1), embedding[:, :, i, 1].reshape(-1), color=colors[i],label=l)
    plt.grid(linestyle="--", alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="best")  # legend有自己的参数可以控制图例位置
    if title is not None:
        plt.title(title)
    if save_path:
        dirs = save_path
        if not os.path.exists(dirs): os.makedirs(dirs)
        plt.savefig(os.path.join(dirs, f"{title}_umap.png"))
    if show:
        plt.show()

def plot_umap_3_dim(embedding, datalabel, title, save_path, show=False):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection = '3d')
    colors = sns.color_palette('coolwarm', len(datalabel))
    for i, l in enumerate(datalabel):
        ax.scatter(embedding[:, :, i, 0].reshape(-1), embedding[:, :, i, 1].reshape(-1), embedding[:, :, i, 2].reshape(-1), color=colors[i],label=l)
    ax.view_init(20,5) # 设置观察视角
    if title is not None:
        plt.title(title)
    if save_path:
        dirs = save_path
        if not os.path.exists(dirs): os.makedirs(dirs)
        plt.savefig(os.path.join(dirs, f"{title}_umap.png"))
    if show:
        plt.show()

def plot_dist(new_dist, save_path=None, title=None, datalabel=None, show=None):
    colors = sns.color_palette('coolwarm', len(datalabel))
    plt.figure(figsize=(20,8), dpi=150)
    x = list(range(new_dist.shape[0]))
    for i, l in enumerate(datalabel):
        plt.plot(x, new_dist[:,i].reshape(-1), color=colors[i],label=l)
        plt.scatter(x, new_dist[:,i].reshape(-1), color=colors[i])
    plt.grid(linestyle="--", alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="best")  # legend有自己的参数可以控制图例位置
    if title is not None:
        plt.title(title)
    if save_path:
        dirs = save_path
        if not os.path.exists(dirs): os.makedirs(dirs)
        plt.savefig(os.path.join(dirs, f"{title}_dist_plot.png"))
    if show:
        plt.show()

def plt_heatMap_sns(scores, save_path=None, title=None, cmap=None, y_ticks=None, x_ticks=None, show=None):
    plt.subplots(figsize=(20, 20), dpi=200)
    plt.rcParams['font.size'] = '5'
    if cmap is None:
        cmap = sns.color_palette("Reds", as_cmap=True)
    if x_ticks and y_ticks:
        sns.heatmap(scores, cmap=cmap,  xticklabels=x_ticks, yticklabels=y_ticks)
    else:
        sns.heatmap(scores, cmap=cmap)
    if title is not None:
        plt.title(title)
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'{title}.png'), bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def load_model(model_name: str, device='cuda', low_cpu_mem_usage=False, layers=[], train_layers=[], is_linear='zero_mlp',show_params=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=low_cpu_mem_usage).to(device)
    device, dtype = model.device, model.dtype

    MODEL_CONFIG = model.config


    if model_name.find('gpt2-xl') != -1:
        MODEL_CONFIG = {"n_heads": model.config.num_attention_heads,
                        "n_layers": model.config.num_hidden_layers,
                        "resid_dim": model.config.hidden_size,
                        "name_or_path": model.config.name_or_path,
                        "attn_hook_names": [f'transformer.h.{layer}.attn.attention.attn_dropout' for layer in
                                            range(model.config.num_hidden_layers)],
                        "k_q_names": [f'model.layers.{layer}.self_attn.k_proj' for layer in
                                            range(model.config.num_hidden_layers)] + 
                                    [f'model.layers.{layer}.self_attn.q_proj' for layer in
                                            range(model.config.num_hidden_layers)]
                        }
    if model_name.find('opt') != -1:
        MODEL_CONFIG = {"n_heads": model.config.num_attention_heads,
                        "n_layers": model.config.num_hidden_layers,
                        "resid_dim": model.config.hidden_size,
                        "name_or_path": model.config.name_or_path,
                        "attn_hook_names": [f'transformer.h.{layer}.attn.attention.attn_dropout' for layer in
                                            range(model.config.num_hidden_layers)],
                        "k_q_names": [f'model.layers.{layer}.self_attn.k_proj' for layer in
                                            range(model.config.num_hidden_layers)] + 
                                    [f'model.layers.{layer}.self_attn.q_proj' for layer in
                                            range(model.config.num_hidden_layers)]
                        }
    if model_name.find('mistralai') != -1:
        MODEL_CONFIG = {"n_heads": model.config.num_attention_heads,
                        "n_layers": model.config.num_hidden_layers,
                        "resid_dim": model.config.hidden_size,
                        "name_or_path": model.config.name_or_path,
                        "attn_hook_names": [f'transformer.h.{layer}.attn.attention.attn_dropout' for layer in
                                            range(model.config.num_hidden_layers)],
                        "k_q_names": [f'model.layers.{layer}.self_attn.k_proj' for layer in
                                            range(model.config.num_hidden_layers)] + 
                                    [f'model.layers.{layer}.self_attn.q_proj' for layer in
                                            range(model.config.num_hidden_layers)]
                        }

    if model_name.find('gpt-neo') != -1:
        for layer_id in layers:
            if is_linear == 'zero_mlp':
                model.transformer.h[layer_id].mlp = MyGPTNeoMLP(model.config).to(device=device, dtype = dtype)
            elif is_linear == 'zero_layer':
                model.transformer.h[layer_id].mlp = MyGPTNeoMLP(model.config).to(device=device, dtype = dtype)
                model.transformer.h[layer_id].attn.attention = MyGPTNeoSelfAttention(model.config,is_linear='zero_atten').to(device=device, dtype = dtype)
            elif is_linear == 'zero_residual':
                block = model.transformer.h[layer_id]
                model.transformer.h[layer_id] = MyGPTNeoBlock(model.config, block=block)
            else:
                model.transformer.h[layer_id].attn.attention = MyGPTNeoSelfAttention(model.config,is_linear=is_linear).to(device=device, dtype = dtype)
            # model.model.layers[layer_id].self_attn = LlamaAttentionLinear(model.config , device).to(device)

            
            # model.model.layers[layer_id].self_attn = LlamaAttentionLinear(model.config , device).to(device)

        MODEL_CONFIG = {"n_heads": model.config.num_attention_heads,
                        "n_layers": model.config.num_hidden_layers,
                        "resid_dim": model.config.hidden_size,
                        "name_or_path": model.config.name_or_path,
                        "attn_hook_names": [f'transformer.h.{layer}.attn.attention.attn_dropout' for layer in
                                            range(model.config.num_hidden_layers)],
                        "k_q_names": [f'transformer.h.{layer}.attn.attention.k_proj' for layer in
                                            range(model.config.num_hidden_layers)] + 
                                    [f'transformer.h.{layer}.attn.attention.q_proj' for layer in
                                            range(model.config.num_hidden_layers)] +
                                    [f'transformer.h.{layer}.attn.attention.v_proj' for layer in
                                            range(model.config.num_hidden_layers)],
                         "q_names":  [f'transformer.h.{layer}.attn.attention.q_proj' for layer in
                                            range(model.config.num_hidden_layers)]

                        }
        # p = model.named_parameters()
        if len(train_layers) > 0:
            for k, params in model.named_parameters():
                flag = False
                for layer_id in train_layers:
                    name = 'name'
                    if is_linear == 'linear_atten':
                        name = f'transformer.h.{layer_id}.attn.attention'
                    if is_linear == 'zero_atten':
                        name = f'transformer.h.{layer_id}.mlp'
                    if is_linear == 'zero_mlp':
                         name = f'transformer.h.{layer_id}.attn'
                    if is_linear == 'zero_layer':
                        name = f'transformer.h.{layer_id}.'
                    if k.find(name) != -1:
                        flag = True
                params.requires_grad = flag

        # return model, tokenizer, MODEL_CONFIG
    if model_name.find('TinyLlama') != -1:
        model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)#.to(device)

        for layer_id in layers:
            # model.transformer.h[layer_id].attn.attention = MyGPTNeoSelfAttention(model.config, device=device,is_linear=is_linear).to(device)
            model.model.layers[layer_id].self_attn = MyLlamaAttention(model.config, device, is_linear=is_linear).to(
                device)

        MODEL_CONFIG = {"n_heads": model.config.num_attention_heads,
                        "n_layers": model.config.num_hidden_layers,
                        "resid_dim": model.config.hidden_size,
                        "name_or_path": model.config.name_or_path,
                        "attn_hook_names": [f'model.layers.{layer}.self_attn.attn_dropout' for layer in
                                            range(model.config.num_hidden_layers)],
                        }
        if len(train_layers) > 0:
            for k, params in model.named_parameters():
                for layer_id in train_layers:
                    if k.find(str(layer_id)) == -1:
                        params.requires_grad = False

        # return model, tokenizer, MODEL_CONFIG

    if model_name.find('gpt-j-6b') !=-1:
        for layer_id in layers:
            if is_linear == 'zero_mlp':
                model.transformer.h[layer_id].mlp = MyGPTJMLP(model.config).to(device=device, dtype = dtype)
            elif is_linear == 'zero_layer':
                model.transformer.h[layer_id].mlp = MyGPTJMLP(model.config).to(device=device, dtype = dtype)
                model.transformer.h[layer_id].attn = MyGPTJAttention(model.config,is_linear='zero_atten').to(device=device, dtype = dtype)
            elif is_linear == 'zero_residual':
                block = model.transformer.h[layer_id]
                model.transformer.h[layer_id] = MyGPTJBlock(model.config, block=block).to(device=device, dtype = dtype)
            else:
                model.transformer.h[layer_id].attn = MyGPTJAttention(model.config,is_linear=is_linear).to(device=device, dtype = dtype)
            # model.model.layers[layer_id].self_attn = LlamaAttentionLinear(model.config , device).to(device)

        MODEL_CONFIG = {"n_heads": model.config.num_attention_heads,
                        "n_layers": model.config.num_hidden_layers,
                        "resid_dim": model.config.hidden_size,
                        "name_or_path": model.config.name_or_path,
                        "attn_hook_names": [f'transformer.h.{layer}.attn.attn_dropout' for layer in
                                            range(model.config.num_hidden_layers)],
                        "k_q_names": [f'transformer.h.{layer}.attn.k_proj' for layer in
                                            range(model.config.num_hidden_layers)] + 
                                    [f'transformer.h.{layer}.attn.q_proj' for layer in
                                            range(model.config.num_hidden_layers)] ,
                         "k_names": [f'transformer.h.{layer}.attn.k_proj' for layer in
                                            range(model.config.num_hidden_layers)] ,          
                         "v_names": [f'transformer.h.{layer}.attn.v_proj' for layer in
                                            range(model.config.num_hidden_layers)] ,
                        "out_proj": [f'transformer.h.{layer}.attn.out_proj' for layer in
                                            range(model.config.num_hidden_layers)],
                        "fc_out": [f'transformer.h.{layer}.mlp.fc_out' for layer in
                                            range(model.config.num_hidden_layers)],
                        "fc_in": [f'transformer.h.{layer}.mlp.fc_in' for layer in
                                            range(model.config.num_hidden_layers)],
                          "q_names":  [f'transformer.h.{layer}.attn.q_proj' for layer in
                                            range(model.config.num_hidden_layers)]
                        }
        # p = model.named_parameters()
        if len(train_layers) > 0:
            for k, params in model.named_parameters():
                flag = False
                for layer_id in train_layers:
                    name = 'name'
                    if is_linear == 'linear_atten':
                        name = f'transformer.h.{layer_id}.attn'
                    if is_linear == 'zero_atten':
                        name = f'transformer.h.{layer_id}.mlp'
                    if is_linear == 'zero_mlp':
                         name = f'transformer.h.{layer_id}.attn'
                    if k.find(name) != -1:
                        flag = True
                params.requires_grad = flag
        
    if show_params:
        for name, param in model.named_parameters():
            if param.requires_grad:
                # print(torch.isnan(param.grad).any())
                # print('name:{} param grad:{} param requires_grad:{},params:{}'.format(name, param.grad, param.requires_grad,param))
                print('name:{} param requires_grad:{}, detype:{}, device:{}'.format(name, param.requires_grad, param.dtype, param.device))

    return model, tokenizer, MODEL_CONFIG

import torchvision.models as models
import torchvision.transforms as T
class LidarResnetEncoder(nn.Module):
    """
    Resnet family to encode lidar.

    Parameters
    ----------
    params: dict
        The parameters of resnet encoder.
    """
    def __init__(self, params):
        super(LidarResnetEncoder, self).__init__()

        self.num_layers = params['num_layers']
        self.pretrained = params['pretrained']
        self.transform = T.Resize(params['size'])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if self.num_layers not in resnets:
            raise ValueError(
                "{} is not a valid number of resnet "
                "layers".format(self.num_layers))

        self.encoder = resnets[self.num_layers](self.pretrained)

    def forward(self, input_lidar):
        """
        Compute deep features from input images.
        todo: multi-scale feature support

        Parameters
        ----------
        input_lidar : torch.Tensor
            The input images have shape of (B,L,C,H,W), where L is the num of agents.
        Returns
        -------
        features: torch.Tensor
            The deep features for each image with a shape of (B,L,C,H,W)
        """
        b, l, c, h, w = input_lidar.shape
        input_lidar = input_lidar.view(b*l*c, h, w)
        
        input_lidar = self.transform(input_lidar)# Image.NEAREST
        input_lidar = torch.unsqueeze(input_lidar,dim=1)
        
        input_lidar = input_lidar.expand(-1,3,-1,-1)
        
        x = self.encoder.conv1(input_lidar)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)

        x = self.encoder.layer1(self.encoder.maxpool(x))
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        # x = self.encoder.layer4(x)

        x = x.view(b, l, -1 , x.shape[-2], x.shape[-1])

        return x

if __name__ == '__main__':
    # encoder = LidarResnetEncoder({'num_layers':18, 'pretrained':True, 'size':512})
    # sample = torch.zeros((1,5,128,496,432))
    # result = encoder(sample)

    # 第九部分 替换第一个token的hidden state为last token的hidden state, 查看attn热力图
    device_str = 'cuda:0'
    layers, train_layers, is_linear = [] , [], 'linear_atten'
    model_name = 'EleutherAI/gpt-j-6b'  # facebook/opt-13b# 'mistralai/Mistral-7B-v0.1' 'openai-community/gpt2-xl' #'EleutherAI/gpt-j-6b' # 'EleutherAI/gpt-neo-1.3B' # 'EleutherAI/gpt-neo-125m'
    model, tokenizer, MODEL_CONFIG = load_model(model_name, device=device_str, layers=layers, train_layers=train_layers,
                                                is_linear=is_linear, show_params=False)
    
    n_layers, n_heads = MODEL_CONFIG['n_layers'], MODEL_CONFIG['n_heads']
    x_ticks = [f"layer{i + 1}" for i in range(n_layers)]
    target_token = 'Apple'
    check_token_id = -1
    save_path_ = os.path.join(sys.path[0], './result/hidden_attn_heat5')
    prompt = 'The Space Needle is in downtown' # 'The Space Needle is in downtown' # 'Beats Music is owned by', 'Beats Music is owned by Apple and the Space Needle is in downtown'
    encoded_line = tokenizer.encode(prompt)
    codes = tokenizer.convert_ids_to_tokens(encoded_line)
    y_ticks = [f"head{i_head}-{c}" for i_head in range(n_heads) for i, c in enumerate(codes)]

    ## attention sink形成原因

    def try_hook_ground(model, tokenizer, model_config, prompt, check_token_id, device, change_layer_name=set(), edit_input=None, edit_output=None):
        model.eval()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with TraceDict2(model, layers=model_config['q_names']+model_config['out_proj']+model_config['fc_out']+model_config['fc_in'],
                        retain_output=True, retain_input=True, edit_input=edit_input, edit_output=edit_output) as ret:
            output_and_cache = model(**inputs, output_hidden_states=True, output_attentions=True)
            attn_output_list = [ret[q].output for q in model_config['out_proj']]
            attn_output = torch.cat(attn_output_list, dim=0).detach().cpu().numpy()
            attn_input_list = [ret[q].input for q in model_config['out_proj']]
            attn_input = torch.cat(attn_input_list, dim=0).detach().cpu().numpy()
            mlp_output_list = [ret[q].output for q in model_config['fc_out']]
            mlp_output = torch.cat(mlp_output_list, dim=0).detach().cpu().numpy()
            mlp_input_list = [ret[q].output for q in model_config['fc_in']]
            mlp_input = torch.cat(mlp_input_list, dim=0).detach().cpu().numpy()
            q_list = [ret[q].output for q in model_config['q_names']]
            past_qs = torch.cat(q_list, dim=0).detach().cpu().numpy()
            past_qs = np.transpose(np.reshape(past_qs,newshape=past_qs.shape[:-1]+(n_heads, -1)),(0,2,1,3))
        ground_attentions = torch.cat(output_and_cache.attentions, dim=0).cpu()
        hidden_state = torch.cat(output_and_cache.hidden_states, dim=0).detach().cpu().numpy()
        past_key = torch.cat([key_values[0] for key_values in output_and_cache.past_key_values], dim=0).detach().cpu().numpy()
        past_values = torch.cat([key_values[1] for key_values in output_and_cache.past_key_values], dim=0).detach().cpu().numpy()
        return attn_input ,attn_output,mlp_input,mlp_output, hidden_state, ground_attentions.detach().numpy(), past_key, past_values, past_qs
    

    def try_hook4(model, tokenizer, model_config, prompt, check_token_id, device, change_layer_name=set()):
        def modify(layer_ids, token_id, n_heads, check_token_id, device):
            def modify_output(output, layer_name, inputs):
                current_layer = int(layer_name.split(".")[2])
                # if current_layer == edit_layer:
                #     if isinstance(output, tuple):
                #         output[0][:, idx] += fv_vector.to(device)
                #         return output
                #     else:
                #         return output
                # else:
                #     return output
                if layer_name in change_layer_name:
                    output[:,0,:] = 0
                return output

            def modify_input(input, layer_name):
                # print(layer_name)
                # for layer_id in layer_ids:
                #     if str(layer_id) in layer_name.split('.'):
                # heads_range = range(n_heads)
                if layer_name in model_config['k_q_names']:
                    input[:, 0, :] = input[:, -1, :]
                # input[:, 4, :] = input[:, -1, :]
                return input

            return modify_output, modify_input
        model.eval()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        token_id, layer_ids = 0, range(3, model_config['n_layers']-1)
        modify_output, modify_input = modify(layer_ids, token_id, model_config['n_heads'],  check_token_id , device)
        with TraceDict2(model, layers=model_config['q_names']+model_config['out_proj']+model_config['fc_out']+model_config['fc_in'],
                        edit_output=modify_output, retain_output=True, retain_input=True) as ret:
            output_and_cache = model(**inputs, output_hidden_states=True, output_attentions=True)
            attn_output_list = [ret[q].output for q in model_config['out_proj']]
            attn_output = torch.cat(attn_output_list, dim=0).detach().cpu().numpy()
            attn_input_list = [ret[q].input for q in model_config['out_proj']]
            attn_input = torch.cat(attn_input_list, dim=0).detach().cpu().numpy()
            mlp_output_list = [ret[q].output for q in model_config['fc_out']]
            mlp_output = torch.cat(mlp_output_list, dim=0).detach().cpu().numpy()
            mlp_input_list = [ret[q].output for q in model_config['fc_in']]
            mlp_input = torch.cat(mlp_input_list, dim=0).detach().cpu().numpy()
            q_list = [ret[q].output for q in model_config['q_names']]
            past_qs = torch.cat(q_list, dim=0).detach().cpu().numpy()
            past_qs = np.transpose(np.reshape(past_qs,newshape=past_qs.shape[:-1]+(n_heads, -1)),(0,2,1,3))
        ground_attentions = torch.cat(output_and_cache.attentions, dim=0).cpu()
        hidden_state = torch.cat(output_and_cache.hidden_states, dim=0).detach().cpu().numpy()
        past_key = torch.cat([key_values[0] for key_values in output_and_cache.past_key_values], dim=0).detach().cpu().numpy()
        past_values = torch.cat([key_values[1] for key_values in output_and_cache.past_key_values], dim=0).detach().cpu().numpy()

        token_id, layer_ids = 0, range(3, model_config['n_layers']-1)
        modify_output, modify_input = modify(layer_ids, token_id, model_config['n_heads'],  check_token_id , device)
        with TraceDict2(model, layers=model_config['k_q_names']+model_config['out_proj']+model_config['fc_out']+model_config['fc_in'], 
                        edit_input=modify_input, retain_output=True, retain_input=True) as ret:
            output_and_cache = model(**inputs,output_attentions=True,output_hidden_states=True)
            change_attentions = torch.stack(output_and_cache.attentions, dim=0)[:,0].cpu()
            change_hidden_state = torch.cat(output_and_cache.hidden_states, dim=0).detach().cpu().numpy()
            q_list = [ret[q].output for q in model_config['q_names']]
            change_past_qs = torch.cat(q_list, dim=0).detach().cpu().numpy()
            change_past_qs = np.transpose(np.reshape(change_past_qs,newshape=change_past_qs.shape[:-1]+(n_heads, -1)),(0,2,1,3))
            change_past_key = torch.cat([key_values[0] for key_values in output_and_cache.past_key_values], dim=0).detach().cpu().numpy()
            change_past_values = torch.cat([key_values[1] for key_values in output_and_cache.past_key_values], dim=0).detach().cpu().numpy()
            change_attn_output_list = [ret[q].output for q in model_config['out_proj']]
            change_attn_output = torch.cat(change_attn_output_list, dim=0).detach().cpu().numpy()
            change_mlp_output_list = [ret[q].output for q in model_config['fc_out']]
            change_mlp_output = torch.cat(change_mlp_output_list, dim=0).detach().cpu().numpy()
            change_mlp_input_list = [ret[q].output for q in model_config['fc_in']]
            change_mlp_input = torch.cat(change_mlp_input_list, dim=0).detach().cpu().numpy()
            change_attn_input_list = [ret[q].input for q in model_config['out_proj']]
            change_attn_input = torch.cat(change_attn_input_list, dim=0).detach().cpu().numpy()
        
        return attn_input ,attn_output,mlp_input,mlp_output,change_attn_input,change_attn_output,change_mlp_input, change_mlp_output,\
            hidden_state, ground_attentions.detach().numpy(), change_hidden_state,change_attentions.detach().numpy(), past_key, past_values, past_qs,change_past_key, change_past_values, change_past_qs
    
    # attn_output,mlp_input,mlp_output, hidden_state,ground_attentions, change_hidden_state,change_attentions, past_key, past_values,past_qs,change_past_key, change_past_values, change_past_qs = try_hook4(model, tokenizer, MODEL_CONFIG, prompt,check_token_id, torch.device(device_str))
    # ground_attentions, change_attentions = ground_attentions[:, :, check_token_id, :],change_attentions[:, :, check_token_id, :]
    
    # # 绘制attn热力图
    # plt_heatMap_sns(ground_attentions.reshape(ground_attentions.shape[0], -1).T,
    #                 title="gpt2_attentions", x_ticks=x_ticks, y_ticks=y_ticks
    #                 , show=True, save_path=save_path)
    # plt_heatMap_sns(change_attentions.reshape(change_attentions.shape[0], -1).T,
    #                 title="gpt2_change_attentions", x_ticks=x_ticks, y_ticks=y_ticks
    #                 , show=True, save_path=save_path)

    
    # hidden state的热力图，折线图，数据值
    def get_hidden_dist(hidden_state):
        state_norm_list = [np.linalg.norm(hidden_state - hidden_state[:,i][:,np.newaxis,:], axis=2)[:,:,np.newaxis] for i in range(hidden_state.shape[1])]
        dist = np.concatenate(state_norm_list,axis=2)
        return dist
    def get_coff(head_states):
        cos_coeff = np.zeros((head_states.shape[0],head_states.shape[1],head_states.shape[1]))
        from sklearn.metrics.pairwise import cosine_similarity
        for layer_id_ in range(head_states.shape[0]):
            for token_id in range(head_states.shape[1]):
                other,last = head_states[layer_id_], head_states[layer_id_,token_id][np.newaxis,:]
                s = cosine_similarity(other, last).reshape(-1)
                cos_coeff[layer_id_,token_id] = s
        return cos_coeff
    
    # ## 探查是mlp导致的还是attn导致的，显示是mlp导致的
    # dist = get_hidden_dist(attn_output)
    # plt_heatMap_sns(dist.reshape(dist.shape[0], -1).T, title="normal_attn_output_without_normalize", x_ticks=x_ticks,
    #                  y_ticks=[ f"{c1}-{c2}" for c1 in codes for c2 in codes]
    #                 , show=True, save_path=save_path)
    # plot_dist(dist[:,-1],save_path=save_path, title='normal_attn_output',
    #           datalabel=[ f"{codes[-1]}-{c2}" for c2 in codes],
    #           show=True)
    
    # dist = get_hidden_dist(mlp_input)
    # plt_heatMap_sns(dist.reshape(dist.shape[0], -1).T, title="normal_mlp_input_without_normalize", x_ticks=x_ticks,
    #                  y_ticks=[ f"{c1}-{c2}" for c1 in codes for c2 in codes]
    #                 , show=True, save_path=save_path)
    # plot_dist(dist[:,-1],save_path=save_path, title='normal_mlp_input',
    #           datalabel=[ f"{codes[-1]}-{c2}" for c2 in codes],
    #           show=True)
    # dist = get_hidden_dist(mlp_output)
    # plt_heatMap_sns(dist.reshape(dist.shape[0], -1).T, title="normal_mlp_output_without_normalize", x_ticks=x_ticks,
    #                  y_ticks=[ f"{c1}-{c2}" for c1 in codes for c2 in codes]
    #                 , show=True, save_path=save_path)
    # plot_dist(dist[:,-1],save_path=save_path, title='normal_mlp_output',
    #           datalabel=[ f"{codes[-1]}-{c2}" for c2 in codes],
    #           show=True)    
    
    # ## 绘制
    # dist = get_hidden_dist(hidden_state[1:])
    # plt_heatMap_sns(dist.reshape(dist.shape[0], -1).T, title="head_dist_without_normalize", x_ticks=x_ticks,
    #                  y_ticks=[ f"{c1}-{c2}" for c1 in codes for c2 in codes]
    #                 , show=True, save_path=save_path)
    # plot_dist(dist[:,-1],save_path=save_path, title='dist',
    #           datalabel=[ f"{codes[-1]}-{c2}" for c2 in codes],
    #           show=True)
    # dist = get_hidden_dist(hidden_state[1:,1:])
    # plt_heatMap_sns(dist.reshape(dist.shape[0], -1).T, title="head_dist_without_fisrt_token", x_ticks=x_ticks,
    #                  y_ticks=[ f"{c1}-{c2}" for c1 in codes[1:] for c2 in codes[1:]]
    #                 , show=True, save_path=save_path)
    # plot_dist(dist[:,2],save_path=save_path, title='dist_without_first_token',
    #           datalabel=[ f"{codes[2]}-{c2}" for c2 in codes[1:]],
    #           show=True)
    
    # coff = get_coff(hidden_state[1:])
    # plt_heatMap_sns(coff.reshape(coff.shape[0], -1).T, title="gpt2_coff", x_ticks=x_ticks,
    #                  y_ticks=[ f"{c1}-{c2}" for c1 in codes for c2 in codes]
    #                 , show=True, save_path=save_path)
    

    def get_umap(activations, dim=2):
            old_shape = activations.shape
            embedding = umap.UMAP(n_components = dim).fit_transform(activations.reshape(-1, old_shape[-1]))
            embedding = embedding.reshape(old_shape[:-1]+(dim,))
            return embedding
    head_label =  [f'head{i}' for i in range(n_heads)]
    layer_label =  [f'layer{i}' for i in range(n_layers)]
    
    # # new_shape = hidden_state.shape[:-1] + (n_heads, -1)
    # # head_states = np.transpose(np.reshape(hidden_state,newshape=new_shape),(0,2,1,3))[1:]
    # # change_head_states = np.transpose(np.reshape(change_hidden_state,newshape=new_shape),(0,2,1,3))[1:]

    # # hidden state降为可视化
    # embedding = get_umap(hidden_state)
    # plot_umap(embedding[1:,np.newaxis,:,:], 
    #           codes,title="hidden_states_token_label",save_path=save_path)
    # plot_umap(embedding[1:,np.newaxis,1:,:], 
    #           codes[1:],title="hidden_states_token_label_without_first_token",save_path=save_path)
    # plot_umap(np.transpose(embedding[1:,np.newaxis,:,:],(2,1,0,3)), 
    #           layer_label,title="hidden_states_layer_label",save_path=save_path)
    # plot_umap(np.transpose(embedding[1:,np.newaxis,1:,:],(2,1,0,3)), 
    #           layer_label,title="hidden_states_token_layer_without_first_token",save_path=save_path)
    
    # # hidden state降为可视化
    # change_embedding = get_umap(change_hidden_state)
    # plot_umap(change_embedding[1:,np.newaxis,:,:], 
    #           codes,title="change_hidden_states_token_label",save_path=save_path)
    # plot_umap(np.transpose(change_embedding[1:,np.newaxis,:,:],(2,1,0,3)), 
    #           layer_label,title="change_hidden_states_layer_label",save_path=save_path)
    def is_not_None(a):
        return a is not None
   
    # # value降为可视化
    def draw_umap_kqv(past_values, title, save_path):
        if is_not_None(past_values):
            save_path = os.path.join(save_path,'kqv')
        
            past_values_embedding = get_umap(np.transpose(past_values,(0,2,1,3)).reshape(past_values.shape[0],past_values.shape[2], -1)[:,np.newaxis,:,:], dim=3)
            plot_umap_3_dim(past_values_embedding,codes,title=f"{title}_token_label_3d",save_path=save_path)
            
            plot_umap_3_dim(np.transpose(past_values_embedding,(2,1,0,3))[:,:,5:],layer_label[5:],title=f"{title}_layer_label_3d_less",save_path=save_path)
            plot_umap_3_dim(np.transpose(past_values_embedding,(2,1,0,3)),layer_label,title=f"{title}_layer_label_3d",save_path=save_path)
            past_values_embedding = get_umap(past_values, dim=3)
            plot_umap_3_dim(past_values_embedding, codes,title=f"head_{title}_token_label_3d",save_path=save_path)
            plot_umap_3_dim(np.transpose(past_values_embedding,(2,1,0,3)),layer_label,title=f"head_{title}_layer_label_3d",save_path=save_path)
            plot_umap_3_dim(np.transpose(past_values_embedding,(0,2,1,3)),head_label,title=f"head_{title}_head_label_3d",save_path=save_path)
        

            past_values_embedding = get_umap(np.transpose(past_values,(0,2,1,3)).reshape(past_values.shape[0],past_values.shape[2], -1)[:,np.newaxis,:,:])
            plot_umap(past_values_embedding,codes,title=f"{title}_token_label",save_path=save_path)
            plot_umap(np.transpose(past_values_embedding,(2,1,0,3)),layer_label,title=f"{title}_layer_label",save_path=save_path)
            plot_umap(np.transpose(past_values_embedding,(2,1,0,3))[:,:,5:],layer_label[5:],title=f"{title}_layer_label_less",save_path=save_path)
            past_values_embedding = get_umap(past_values)
            plot_umap(past_values_embedding, codes,title=f"head_{title}_token_label",save_path=save_path)
            plot_umap(np.transpose(past_values_embedding,(2,1,0,3)),layer_label,title=f"head_{title}_layer_label",save_path=save_path)
            plot_umap(np.transpose(past_values_embedding,(0,2,1,3)),head_label,title=f"head_{title}_head_label",save_path=save_path)
    def draw_umap_kqv_layer(past_values, title, layer_list, save_path):
        if is_not_None(past_values):
            save_path = os.path.join(save_path,'kqv_layer')
            past_values_embedding = get_umap(np.transpose(past_values,(0,2,1,3)).reshape(past_values.shape[0],past_values.shape[2], -1)[:,np.newaxis,:,:])
            # plot_umap(past_values_embedding,codes,title=f"{title}_token_label",save_path=save_path)
            for layer_id in layer_list:
                plot_umap(past_values_embedding[layer_id][np.newaxis,:,:,:],codes,title=f"layer{layer_id}_{title}_token_label",save_path=save_path)
            past_values_embedding = get_umap(past_values)
            for layer_id in layer_list:
                plot_umap(past_values_embedding[layer_id][np.newaxis,:,:,:], codes,title=f"layer{layer_id}_head_{title}_token_label",save_path=save_path)
            
            past_values_embedding = get_umap(np.transpose(past_values,(0,2,1,3)).reshape(past_values.shape[0],past_values.shape[2], -1)[:,np.newaxis,:,:], 3)
            for layer_id in layer_list:
                plot_umap_3_dim(past_values_embedding[layer_id][np.newaxis,:,:,:],codes,title=f"layer{layer_id}_{title}_token_label_3d",save_path=save_path)
            past_values_embedding = get_umap(past_values, 3)
            for layer_id in layer_list:
                plot_umap_3_dim(past_values_embedding[layer_id][np.newaxis,:,:,:], codes,title=f"layer{layer_id}_head_{title}_token_label_3d",save_path=save_path)
           
            # plot_umap(np.transpose(past_values_embedding,(2,1,0,3)),layer_label,title=f"{title}_layer_label",save_path=save_path)
        
            # plot_umap(past_values_embedding, codes,title=f"head_{title}_token_label",save_path=save_path)
            # plot_umap(np.transpose(past_values_embedding,(2,1,0,3)),layer_label,title=f"head_{title}_layer_label",save_path=save_path)
            # plot_umap(np.transpose(past_values_embedding,(0,2,1,3)),head_label,title=f"head_{title}_head_label",save_path=save_path)
    def draw_multi_key_q(layer_id, save_path,past_key,past_qs):
        if is_not_None(past_key) and is_not_None(past_qs):
            save_path = os.path.join(save_path,'key_q')
            key_q = np.vstack((past_key[np.newaxis,:],past_qs[np.newaxis,:]))
            key_q_embedding = get_umap(key_q[:,layer_id])
            plot_multi_umap((key_q_embedding[0][np.newaxis,:],key_q_embedding[1][np.newaxis,:]),codes,title=f"key_q_token_label_layer{layer_id}",save_path=save_path)
            plot_multi_umap((np.transpose(key_q_embedding[0][np.newaxis,:],(0,2,1,3)),
                            np.transpose(key_q_embedding[1][np.newaxis,:],(0,2,1,3))),
                            head_label,title=f"key_q_head_label_layer{layer_id}",save_path=save_path)
            for head_id in range(n_heads):
                plot_multi_umap((key_q_embedding[0,head_id ][np.newaxis,np.newaxis,:],
                                    key_q_embedding[1,head_id][np.newaxis,np.newaxis,:]),codes,
                                    title=f"key_q_token_label_layer{layer_id}_head{head_id}",save_path=save_path)
    
    # layer_id = 24
    # draw_multi_key_q(layer_id, save_path)   
        
    # layer_list = list(range(27))
    # draw_umap_kqv_layer(past_key,'past_key',layer_list , save_path)
    # draw_umap_kqv_layer(past_values,'past_values', layer_list, save_path)
    # draw_umap_kqv_layer(past_qs,'past_qs',layer_list, save_path)
    
    # draw_umap_kqv(past_values, 'past_values', save_path)
    # draw_umap_kqv(past_key, 'past_key', save_path)
    # draw_umap_kqv(past_qs, 'past_qs', save_path)
    
    def draw_dist(past_key,past_values,past_qs,attn_input,attn_output, mlp_input, mlp_output,
                  hidden_state, ground_attentions,change_name, layer_id,layer_list,save_path):
        save_path = os.path.join(save_path,change_name)

        def draw_hidden_embedding(hidden_state, name, dim=2):
            if is_not_None(hidden_state):
                embedding = get_umap(hidden_state, dim)
                if dim ==3 :
                    plot_umap_3_dim(embedding[:,np.newaxis,:,:], 
                            codes,title=f"{name}_token_label_{dim}d",save_path=save_path)
                    plot_umap_3_dim(embedding[:,np.newaxis,1:,:], 
                            codes[1:],title=f"{name}_token_label_without_first_token_{dim}d",save_path=save_path)
                    plot_umap_3_dim(np.transpose(embedding[:,np.newaxis,:,:],(2,1,0,3)), 
                            layer_label,title=f"{name}_layer_label_{dim}d",save_path=save_path)
                    plot_umap_3_dim(np.transpose(embedding[:,np.newaxis,1:,:],(2,1,0,3)), 
                            layer_label,title=f"{name}_token_layer_without_first_token_{dim}d",save_path=save_path)
                    plot_umap(embedding[:,np.newaxis,:,[0,1]], 
                            codes,title=f"{name}_token_label_3d_x_y",save_path=save_path)
                    plot_umap(embedding[:,np.newaxis,:,[0,2]], 
                            codes,title=f"{name}_token_label_3d_x_z",save_path=save_path)
                    plot_umap(embedding[:,np.newaxis,:,[1,2]], 
                            codes,title=f"{name}_token_label_3d_y_z",save_path=save_path)
                else:
                    plot_umap(embedding[:,np.newaxis,:,:], 
                            codes,title=f"{name}_token_label",save_path=save_path)
                    plot_umap(embedding[:,np.newaxis,1:,:], 
                            codes[1:],title=f"{name}_token_label_without_first_token",save_path=save_path)
                    plot_umap(np.transpose(embedding[:,np.newaxis,:,:],(2,1,0,3)), 
                            layer_label,title=f"{name}_layer_label",save_path=save_path)
                    plot_umap(np.transpose(embedding[:,np.newaxis,1:,:],(2,1,0,3)), 
                                layer_label,title=f"{name}_token_layer_without_first_token",save_path=save_path)
        
        if is_not_None(attn_input):
            dist = get_hidden_dist(attn_input)
            plt_heatMap_sns(dist.reshape(dist.shape[0], -1).T, title=f"attn_input_without_normalize", x_ticks=x_ticks,
                            y_ticks=[ f"{c1}-{c2}" for c1 in codes for c2 in codes]
                            , show=True, save_path=save_path)
            plot_dist(dist[:,-1],save_path=save_path, title=f'attn_input',
                    datalabel=[ f"{codes[-1]}-{c2}" for c2 in codes],
                    show=True)

        if is_not_None(attn_output):
            dist = get_hidden_dist(attn_output)
            plt_heatMap_sns(dist.reshape(dist.shape[0], -1).T, title=f"attn_output_without_normalize", x_ticks=x_ticks,
                            y_ticks=[ f"{c1}-{c2}" for c1 in codes for c2 in codes]
                            , show=True, save_path=save_path)
            plot_dist(dist[:,-1],save_path=save_path, title=f'attn_output',
                    datalabel=[ f"{codes[-1]}-{c2}" for c2 in codes],
                    show=True)
        if is_not_None(mlp_input):
            dist = get_hidden_dist(mlp_input)
            plt_heatMap_sns(dist.reshape(dist.shape[0], -1).T, title=f"mlp_input_without_normalize", x_ticks=x_ticks,
                            y_ticks=[ f"{c1}-{c2}" for c1 in codes for c2 in codes]
                            , show=True, save_path=save_path)
            plot_dist(dist[:,-1],save_path=save_path, title=f'mlp_input',
                    datalabel=[ f"{codes[-1]}-{c2}" for c2 in codes],
                    show=True)
        if is_not_None(mlp_output):
            dist = get_hidden_dist(mlp_output)
            plt_heatMap_sns(dist.reshape(dist.shape[0], -1).T, title=f"mlp_output_without_normalize", x_ticks=x_ticks,
                            y_ticks=[ f"{c1}-{c2}" for c1 in codes for c2 in codes]
                            , show=True, save_path=save_path)
            plot_dist(dist[:,-1],save_path=save_path, title=f'mlp_output',
                    datalabel=[ f"{codes[-1]}-{c2}" for c2 in codes],
                    show=True)
        if is_not_None(ground_attentions):
            plt_heatMap_sns(ground_attentions.reshape(ground_attentions.shape[0], -1).T,
                            title=f"{change_name}_attentions", x_ticks=x_ticks, y_ticks=y_ticks
                            , show=True, save_path=save_path)
        if is_not_None(hidden_state):
            dist = get_hidden_dist(hidden_state[1:])
            plot_dist(dist[:,-1],save_path=save_path, title=f"{change_name}_dist",
                    datalabel=[ f"{codes[-1]}-{c2}" for c2 in codes],
                    show=True)
            plot_dist(dist[:,-1,1:],save_path=save_path, title=f"{change_name}_dist_without_first_token",
                    datalabel=[ f"{codes[-1]}-{c2}" for c2 in codes[1:]],
                    show=True)
        
        draw_umap_kqv(past_values, 'past_values', save_path)
        draw_umap_kqv(past_key, 'past_key', save_path)
        draw_umap_kqv(past_qs, 'past_qs', save_path)


        draw_hidden_embedding(hidden_state[1:],'hidden_states')
        draw_hidden_embedding(attn_input, 'attn_input')
        draw_hidden_embedding(attn_output, 'attn_output')
        draw_hidden_embedding(mlp_input, 'mlp_input')
        draw_hidden_embedding(mlp_output, 'mlp_output')
        draw_hidden_embedding(hidden_state[1:],'hidden_states', 3)
        draw_hidden_embedding(attn_input, 'attn_input', 3)
        draw_hidden_embedding(attn_output, 'attn_output', 3)
        draw_hidden_embedding(mlp_input, 'mlp_input', 3)
        draw_hidden_embedding(mlp_output, 'mlp_output', 3)

       
        draw_multi_key_q(layer_id, save_path,past_key,past_qs)   
            
        draw_umap_kqv_layer(past_key,'past_key',layer_list , save_path)
        draw_umap_kqv_layer(past_values,'past_values', layer_list, save_path)
        draw_umap_kqv_layer(past_qs,'past_qs',layer_list, save_path)
        
        
    
    ## 第一部分
    # data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    # from counterfact import CounterFactDataset
    # dataset = CounterFactDataset(os.path.join(sys.path[0], './data/'))

    prompt = '' #'The Space Needle is in downtown' # 'The Space Needle is in downtown' # 'Beats Music is owned by', 'Beats Music is owned by Apple and the Space Needle is in downtown'
    prompts = [
        # 'Beats Music is owned by',
        'The Space Needle is in downtown'
        ]
    # for prompt in prompts: #'neighborhood_prompts']:
    #     encoded_line = tokenizer.encode(prompt)
    #     codes = tokenizer.convert_ids_to_tokens(encoded_line)
    #     y_ticks = [f"head{i_head}-{c}" for i_head in range(n_heads) for i, c in enumerate(codes)]
    #     save_path = os.path.join(save_path_, str(len(codes)))
    #     layer_id,layer_list =  10, list(range(0, 27, 3))

    #     change_layer_name, change_name = {},'ground_truth'  # ,'transformer.h.0.mlp.fc_out' 'transformer.h.1.attn.out_proj', 'out_proj' # 
    #     attn_input ,attn_output,mlp_input,mlp_output, hidden_state, ground_attentions, past_key, past_values, past_qs = try_hook_ground(model, tokenizer, MODEL_CONFIG, prompt,check_token_id, torch.device(device_str))
    #     ground_attentions = ground_attentions[:, :, check_token_id, :]
    #     draw_dist(past_key,past_values,past_qs,attn_input,attn_output, mlp_input, mlp_output,hidden_state, ground_attentions,change_name, layer_id,layer_list,save_path)
        
    #     # 探查是layer 1 的mlp还是attn导致的layer2第一个token突然变化
    #     change_layer_name, change_name= {'transformer.h.1.mlp.fc_out'},'fc_out'  # ,'transformer.h.0.mlp.fc_out' 'transformer.h.1.attn.out_proj', 'out_proj' # 
    #     # change_layer_name, change_name = {},'ground_truth'  # ,'transformer.h.0.mlp.fc_out' 'transformer.h.1.attn.out_proj', 'out_proj' # 
    #     attn_input,attn_output,mlp_input,mlp_output,change_attn_input, change_attn_output,change_mlp_input, change_mlp_output,\
    #         hidden_state,ground_attentions, change_hidden_state,change_attentions, past_key, past_values,past_qs,change_past_key, \
    #             change_past_values, change_past_qs = try_hook4(model, tokenizer, MODEL_CONFIG, prompt,check_token_id, torch.device(device_str), change_layer_name)
    #     ground_attentions, change_attentions = ground_attentions[:, :, check_token_id, :],change_attentions[:, :, check_token_id, :]
    #     draw_dist(past_key,past_values,past_qs,attn_input,attn_output, mlp_input, mlp_output,hidden_state, ground_attentions,change_name, layer_id,layer_list,save_path)
    #     draw_dist(change_past_key,change_past_values,change_past_qs, change_attn_input,change_attn_output, change_mlp_input, change_mlp_output,change_hidden_state, change_attentions, 'change',layer_id,layer_list, save_path)

    prompts = [
        'Beats Music is owned by',
        'The Space Needle is in downtown',
        'A wiki is a form of online hypertext publication that is collaboratively edited',
        'A wiki is a form of online hypertext publication that is collaboratively edited and managed by its own audience directly through a web',
        'Wikis can also make WYSIWYG editing available to users, usually through a JavaScript control that translates graphically entered formatting instructions into the corresponding',
        'Wikipedia became the most famous wiki site, launched in January 2001 and entering the top ten most popular websites in 2007. In the early 2000s, wikis were increasingly adopted in enterprise as collaborative software. Common uses included project communication, intranets, and documentation, initially for technical ',
        'Wikis can also be created on a "wiki farm", where the server-side software is implemented by the wiki farm owner. Some wiki farms can also make private, password-protected wikis. Free wiki farms generally contain advertising on every'
        'Wikis are enabled by wiki software, otherwise known as wiki engines. A wiki engine, being a form of a content management system, differs from other web-based systems such as blog software or static site generators, in that the content is created without any defined owner or leader, and wikis have little inherent structure, allowing structure to emerge according to the needs of the users. Wiki engines usually allow content to be written using a simplified markup language and sometimes edited with the help of a rich-text editor. There are dozens of different wiki'
    ]
    prompts = [
        'Beats Music is owned by',
        'The Space Needle is in downtown'
        ,'The concept of artificial intelligence (AI) has captured the imagination of scientists, engineers, and enthusiasts'
        ,'The concept of artificial intelligence (AI) has captured the imagination of scientists, engineers, and enthusiasts alike for decades. AI refers to the simulation of human intelligence in machines, enabling them to perform tasks that typically require human cognition, such as learning, problem-solving, and decision-making. Over the years, AI has evolved from simple rule-based systems to complex neural networks capable of remarkable feats, including natural'
        ,'The concept of artificial intelligence (AI) has captured the imagination of scientists, engineers, and enthusiasts alike for decades. AI refers to the simulation of human intelligence in machines, enabling them to perform tasks that typically require human cognition, such as learning, problem-solving, and decision-making. Over the years, AI has evolved from simple rule-based systems to complex neural networks capable of remarkable feats, including natural language processing, image recognition, and autonomous driving. One of the most significant advancements in AI has been the development of deep learning, a subfield of machine learning that uses artificial neural networks with many layers to learn from vast amounts of'
        ,'The concept of artificial intelligence (AI) has captured the imagination of scientists, engineers, and enthusiasts alike for decades. AI refers to the simulation of human intelligence in machines, enabling them to perform tasks that typically require human cognition, such as learning, problem-solving, and decision-making. Over the years, AI has evolved from simple rule-based systems to complex neural networks capable of remarkable feats, including natural language processing, image recognition, and autonomous driving. One of the most significant advancements in AI has been the development of deep learning, a subfield of machine learning that uses artificial neural networks with many layers to learn from vast amounts of data. Deep learning algorithms have demonstrated unprecedented accuracy in various domains, revolutionizing industries such as healthcare, finance, and transportation. However, despite its tremendous potential, AI also raises ethical concerns and challenges. Issues related to bias in algorithms, data privacy, job displacement, and the existential risk posed by superintelligent'
        ,'The concept of artificial intelligence (AI) has captured the imagination of scientists, engineers, and enthusiasts alike for decades. AI refers to the simulation of human intelligence in machines, enabling them to perform tasks that typically require human cognition, such as learning, problem-solving, and decision-making. Over the years, AI has evolved from simple rule-based systems to complex neural networks capable of remarkable feats, including natural language processing, image recognition, and autonomous driving. One of the most significant advancements in AI has been the development of deep learning, a subfield of machine learning that uses artificial neural networks with many layers to learn from vast amounts of data. Deep learning algorithms have demonstrated unprecedented accuracy in various domains, revolutionizing industries such as healthcare, finance, and transportation. However, despite its tremendous potential, AI also raises ethical concerns and challenges. Issues related to bias in algorithms, data privacy, job displacement, and the existential risk posed by superintelligent AI have sparked debates worldwide. As we continue to push the boundaries of AI research and development, it is essential to consider the societal implications and ensure that AI technologies are deployed responsibly and ethically. By addressing these challenges collaboratively, we can harness the power of AI to create a more equitable, sustainable, and prosperous future for'
        ]
    # result_dict = list()
    # for prompt in prompts: #'neighborhood_prompts']:
    #     encoded_line = tokenizer.encode(prompt)
    #     codes = tokenizer.convert_ids_to_tokens(encoded_line)
    #     y_ticks = [f"head{i_head}-{c}" for i_head in range(n_heads) for i, c in enumerate(codes)]
    #     save_path = os.path.join(save_path_, str(len(codes)))
    #     layer_id,layer_list =  10, list(range(0, 27, 8))
    #     change_layer_name, change_name = {},'ground_truth'  # ,'transformer.h.0.mlp.fc_out' 'transformer.h.1.attn.out_proj', 'out_proj' # 
    #     attn_input ,attn_output,mlp_input,mlp_output, hidden_state, ground_attentions, past_key, past_values, past_qs = try_hook_ground(model, tokenizer, MODEL_CONFIG, prompt,check_token_id, torch.device(device_str))
        
    #     ground_attentions_ = ground_attentions.reshape(-1, ground_attentions.shape[-1])
    #     ground_attentions_mean = np.mean(ground_attentions_, axis=0)
    #     sort_list = np.argsort(ground_attentions_mean)[-3:]
    #     result = []
    #     for index in sort_list:
    #         result.append([int(index), codes[index], float(ground_attentions_mean[index])])
    #     result_dict.append({
    #         'length':len(codes),
    #         'text':prompt,
    #         'code':result,
    #     })
    #     ground_attentions = ground_attentions[:, :, check_token_id, :]
    #     if len(codes) < 200:
    #         draw_dist(past_key,past_values,past_qs,attn_input,attn_output, mlp_input, mlp_output,hidden_state, ground_attentions,change_name, layer_id,layer_list,save_path)
    # yaml.dump(result_dict ,open(f"./results/more_range_prompt2.yml", "w"))
        
    # 第二部分：探查交换attn后的效果
    # def change_attn():
    #     import yaml
    #     change_token_ids = range(1,5)
    #     eval_tasks = ["hellaswag", "piqa", "winogrande",  "mathqa", ]
    #     dict_acc_all = dict()
    #     def modify(change_token_id=1):
    #             def modify_output(output, layer_name, inputs):
    #                 current_layer = int(layer_name.split(".")[2])
    #                 # if current_layer == edit_layer:
    #                 #     if isinstance(output, tuple):
    #                 #         output[0][:, idx] += fv_vector.to(device)
    #                 #         return output
    #                 #     else:
    #                 #         return output
    #                 # else:
    #                 #     return output
    #                 return output

    #             def modify_input(input, layer_name):
    #                 # print(layer_name)
    #                 # for layer_id in layer_ids:
    #                 #     if str(layer_id) in layer_name.split('.'):
    #                 # heads_range = range(n_heads)
    #                 # for i in range(1, input.shape[2]):
    #                 tmp = input[:, :, 1:, change_token_id]
    #                 input[:, :, 1:, change_token_id] = input[:, :, 1:, 0]
    #                 input[:, :, 1:, 0] = tmp
    #                 # input[:, 4, :] = input[:, -1, :]
    #                 return input

    #             return modify_output, modify_input
    #     for change_token_id in change_token_ids:
    #         modify_output, modify_input = modify(change_token_id)
    #         with TraceDict2(model, layers=MODEL_CONFIG['attn_hook_names'], edit_input=modify_input,
    #                             edit_output=modify_output, retain_output=False) as ret:
    #             result2 = run_eval_harness(model, tokenizer, model_name ,eval_tasks, torch.device(device_str), 4, sink_token=None)
    #             dict_acc = dict()
    #             print(f'change to token {change_token_id}_________')
    #             for (key) in result2['results'].keys():
    #                 print(key, result2['results'][key]['acc'])
    #                 dict_acc[key] = float(result2['results'][key]['acc'])
    #         dict_acc_all[change_token_id] = dict_acc
    #         yaml.dump(dict_acc_all ,open(f"./results/change_attn.yml", "w"))
    #     print(dict_acc_all)
    # change_attn()
   
    # # 第三部分：测试替换第一个token或者fc_out的效果
    # model_config = MODEL_CONFIG
    # change_layer_name, change_name= {'transformer.h.1.mlp.fc_out'},'fc_out'  # ,'transformer.h.0.mlp.fc_out' 'transformer.h.1.attn.out_proj', 'out_proj' # 
    # eval_tasks = ["hellaswag", "piqa", "winogrande",  "mathqa", ]
    # def modify():
    #     def modify_output(output, layer_name, inputs):
    #         current_layer = int(layer_name.split(".")[2])
    #         # if current_layer == edit_layer:
    #         #     if isinstance(output, tuple):
    #         #         output[0][:, idx] += fv_vector.to(device)
    #         #         return output
    #         #     else:
    #         #         return output
    #         # else:
    #         #     return output
    #         if layer_name in change_layer_name:
    #             output[:,0,:] = 0
    #         return output

    #     def modify_input(input, layer_name):
    #         # print(layer_name)
    #         # for layer_id in layer_ids:
    #         #     if str(layer_id) in layer_name.split('.'):
    #         # heads_range = range(n_heads)
    #         if layer_name in model_config['k_q_names']:
    #             input[:, 0, :] = input[:, -1, :]
    #         # input[:, 4, :] = input[:, -1, :]
    #         return input

    #     return modify_output, modify_input
    # modify_output, modify_input = modify()
    # dict_acc_all = dict()
    # with TraceDict2(model, layers=model_config['k_q_names']+model_config['fc_out'], 
    #                     edit_input=modify_input, retain_output=False, retain_input=False) as ret:
    #     result2 = run_eval_harness(model, tokenizer, model_name ,eval_tasks, torch.device(device_str), 4, sink_token=None)
    #     dict_acc = dict()
    #     for (key) in result2['results'].keys():
    #         print(key, result2['results'][key]['acc'])
    #         dict_acc[key] = float(result2['results'][key]['acc'])
    # dict_acc_all['change to last token'] = dict_acc
    # with TraceDict2(model, layers=model_config['k_q_names']+model_config['fc_out'], 
    #                     edit_output=modify_output, retain_output=False, retain_input=False) as ret:
    #     result2 = run_eval_harness(model, tokenizer, model_name ,eval_tasks, torch.device(device_str), 4, sink_token=None)
    #     dict_acc = dict()
    #     for (key) in result2['results'].keys():
    #         print(key, result2['results'][key]['acc'])
    #         dict_acc[key] = float(result2['results'][key]['acc'])
    # dict_acc_all['fc_out'] = dict_acc
    # yaml.dump(dict_acc_all ,open(f"./results/small_attn.yml", "w"))
        
    ## 第四部分，找出选sink token的原则：
    # data = load_dataset('gsm8k', 'main', split='test')
    # check_token_id = -1
    # model_config = MODEL_CONFIG
    # change_layer_name, change_name= {'transformer.h.1.mlp.fc_out'},'fc_out'  # ,'transformer.h.0.mlp.fc_out' 'transformer.h.1.attn.out_proj', 'out_proj' # 
    # def modify():
    #     def modify_output(output, layer_name, inputs):
    #         current_layer = int(layer_name.split(".")[2])
    #         # if current_layer == edit_layer:
    #         #     if isinstance(output, tuple):
    #         #         output[0][:, idx] += fv_vector.to(device)
    #         #         return output
    #         #     else:
    #         #         return output
    #         # else:
    #         #     return output
    #         if layer_name in change_layer_name:
    #             output[:,0,:] = 0
    #         return output

    #     def modify_input(input, layer_name):
    #         # print(layer_name)
    #         # for layer_id in layer_ids:
    #         #     if str(layer_id) in layer_name.split('.'):
    #         # heads_range = range(n_heads)
    #         if layer_name in model_config['k_q_names']:
    #             input[:, 0, :] = input[:, -1, :]
    #         # input[:, 4, :] = input[:, -1, :]
    #         return input

    #     return modify_output, modify_input
    # modify_output, modify_input = modify()
    # model.eval()
    # result_dict = list()
    # for text in data[:100]['question']:
    #     inputs = tokenizer(text, return_tensors="pt").to(device_str)
    #     encoded_line = tokenizer.encode(text)
    #     codes = tokenizer.convert_ids_to_tokens(encoded_line)
    #     def get_max_code(ground_attentions):
    #         ground_attentions = ground_attentions.reshape(-1, ground_attentions.shape[-1])
    #         ground_attentions_mean = np.mean(ground_attentions, axis=0)
    #         sort_list = np.argsort(ground_attentions_mean)[-3:]
    #         index, max_num = np.argmax(ground_attentions_mean),np.max(ground_attentions_mean)
    #         result = []
    #         for index in sort_list:
    #             result.append([int(index), codes[index], float(ground_attentions_mean[index])])
    #         return result
    #     output_and_cache = model(**inputs, output_hidden_states=True, output_attentions=True)
    #     ground_attentions = torch.cat(output_and_cache.attentions, dim=0).cpu().detach().numpy()[:, :, check_token_id, :]
    #     code =get_max_code(ground_attentions)
    #     # print(ground_attentions)
    #     with TraceDict2(model, layers=model_config['k_q_names']+model_config['fc_out'], 
    #                     edit_input=modify_input, retain_output=False, retain_input=False) as ret:
    #         output_and_cache = model(**inputs, output_hidden_states=True, output_attentions=True)
    #         ground_attentions = torch.cat(output_and_cache.attentions, dim=0).cpu().detach().numpy()[:, :, check_token_id, :]
    #         ground_attentions = ground_attentions.reshape(-1, ground_attentions.shape[-1])
    #     code_change =get_max_code(ground_attentions)
    #     result_dict.append({
    #         'text':text,
    #         'code':code,
    #         'code_change':code_change,
    #     })
    #     yaml.dump(result_dict ,open(f"./results/sink_token_more.yml", "w"))
    
    # ## 第五部分， see the function of attn/mlp
    # prompt = 'The concept of artificial intelligence (AI) has captured the imagination of scientists, engineers, and enthusiasts alike for decades. AI refers to the simulation of human intelligence in machines, enabling them to perform tasks that typically require human cognition, such as learning, problem-solving, and decision-making. Over the years, AI has evolved from simple rule-based systems to complex neural networks capable of remarkable feats, including natural'
    # # 'Beats Music is owned by'# Apple and the Space Needle is in downtown'
    # # prompt = "Apple was founded as Apple Computer Company on April 1, 1976, to produce and market Steve Wozniak's Apple I personal computer. The company was incorporated by Wozniak and Steve Jobs in 1977. Its second computer, the Apple II, became a best seller as one of the first mass-produced microcomputers. Apple introduced the Lisa in 1983 and the Macintosh in 1984, as some of the first computers to use a graphical user interface and a mouse. By 1985, the company's internal problems included the high cost of its products and power struggles between executives. That year Jobs left Apple to form"
    # encoded_line = tokenizer.encode(prompt)
    # codes = tokenizer.convert_ids_to_tokens(encoded_line)
    # y_ticks = [f"head{i_head}-{c}" for i_head in range(n_heads) for i, c in enumerate(codes)]

    # model_config = MODEL_CONFIG
    # model.eval()
    # inputs = tokenizer(prompt, return_tensors="pt").to(device_str)
    # check_token_id = -1
    # save_path = save_path_
    
    # def try_hook(layer_list):
    #     def modify():
    #         def modify_output(output, layer_name, inputs):
    #             current_layer = int(layer_name.split(".")[2])
    #             # if current_layer == edit_layer:
    #             #     if isinstance(output, tuple):
    #             #         output[0][:, idx] += fv_vector.to(device)
    #             #         return output
    #             #     else:
    #             #         return output
    #             # else:
    #             #     return output
    #             if layer_name in layer_list:
    #                 output[:] = 0
    #             return output
    #         def modify_input(input, layer_name):
    #             # print(layer_name)
    #             # for layer_id in layer_ids:
    #             #     if str(layer_id) in layer_name.split('.'):
    #             # heads_range = range(n_heads)
    #             if layer_name in model_config['k_q_names']:
    #                 input[:, 0, :] = input[:, -1, :]
    #             # input[:, 4, :] = input[:, -1, :]
    #             return input
    #         return modify_output, modify_input
    #     modify_output, modify_input = modify()
    #     with TraceDict2(model, layers=model_config['q_names']+model_config['out_proj']+model_config['fc_out']+model_config['fc_in'],
    #                     edit_output=modify_output, retain_output=True, retain_input=True) as ret:
    #         output_and_cache = model(**inputs, output_hidden_states=True, output_attentions=True)
    #         attn_output_list = [ret[q].output for q in model_config['out_proj']]
    #         attn_output = torch.cat(attn_output_list, dim=0).detach().cpu().numpy()
    #         attn_input_list = [ret[q].input for q in model_config['out_proj']]
    #         attn_input = torch.cat(attn_input_list, dim=0).detach().cpu().numpy()
    #         mlp_output_list = [ret[q].output for q in model_config['fc_out']]
    #         mlp_output = torch.cat(mlp_output_list, dim=0).detach().cpu().numpy()
    #         mlp_input_list = [ret[q].output for q in model_config['fc_in']]
    #         mlp_input = torch.cat(mlp_input_list, dim=0).detach().cpu().numpy()
    #         q_list = [ret[q].output for q in model_config['q_names']]
    #         past_qs = torch.cat(q_list, dim=0).detach().cpu().numpy()
    #         past_qs = np.transpose(np.reshape(past_qs,newshape=past_qs.shape[:-1]+(n_heads, -1)),(0,2,1,3))
    #     ground_attentions = torch.cat(output_and_cache.attentions, dim=0).cpu()
    #     hidden_state = torch.cat(output_and_cache.hidden_states, dim=0).detach().cpu().numpy()
    #     past_key = torch.cat([key_values[0] for key_values in output_and_cache.past_key_values], dim=0).detach().cpu().numpy()
    #     past_values = torch.cat([key_values[1] for key_values in output_and_cache.past_key_values], dim=0).detach().cpu().numpy()
    #     return attn_input,attn_output,mlp_input,mlp_output,hidden_state,ground_attentions.detach().numpy(),past_key, past_values, past_qs
    #     # return attn_input ,attn_output,mlp_input,mlp_output,change_attn_input,change_attn_output,change_mlp_input, change_mlp_output,\
    #     #     hidden_state, ground_attentions.detach().numpy(), change_hidden_state,change_attentions.detach().numpy(), past_key, past_values, past_qs,change_past_key, change_past_values, change_past_qs
   
    # change_name = 'zero_attn'
    # attn_input,attn_output,mlp_input,mlp_output,hidden_state,ground_attentions,past_key, past_values, past_qs = try_hook(model_config['out_proj'])
    # ground_attentions = ground_attentions[:, :, check_token_id, :]
    # layer_id,layer_list =  10, list(range(0, 27, 3))
    # draw_dist(past_key,past_values,past_qs,attn_input,attn_output, mlp_input, mlp_output,hidden_state, ground_attentions,change_name, layer_id,layer_list,save_path)
    
    # change_name = 'zero_mlp'
    # attn_input,attn_output,mlp_input,mlp_output,hidden_state,ground_attentions,past_key, past_values, past_qs = try_hook(model_config['fc_out'])
    # def draw_other_token_attn(save_path, ground_attentions,change_name):
    #     save_path = os.path.join(save_path,change_name)
    #     for i in range(3, len(codes)-1):
    #         ground_attentions_ = ground_attentions[:, :, i, :]
    #         plt_heatMap_sns(ground_attentions_.reshape(ground_attentions_.shape[0], -1).T,
    #                             title=f"{change_name}_attentions_token{i}", x_ticks=x_ticks, y_ticks=y_ticks
    #                             , show=True, save_path=save_path)
    # # draw_other_token_attn(save_path, ground_attentions,change_name)
    # ground_attentions = ground_attentions[:, :, check_token_id, :]
    # layer_id,layer_list =  10, list(range(0, 27, 3))
    # draw_dist(past_key,past_values,past_qs,attn_input,attn_output, mlp_input, mlp_output,hidden_state, ground_attentions,change_name, layer_id,layer_list,save_path)
    
    ## 第六部分， 查看不同model size在不同step下的情况
    # # 使用别人已经训练好的模型
    # from transformers import GPTNeoXForCausalLM, AutoTokenizer
    # import pickle
    # model_size_list = ['1.4b' , '2.8b','6.9b'] # '1b','1.4b' , '2.8b','70m', '160m','410m', '6.9b'
    # device = 'cuda:3'
    # prompt = 'The Space Needle is in downtown' # 'Beats Music is owned by', 'The Space Needle is in downtown'
    # target_token = 'Apple'
    # eval_tasks = [
    #         "hellaswag",
    #         "piqa",
    #         "winogrande",
    #         "mathqa",
    #     ]
    # save_path = os.path.join(sys.path[0], './results/pythia_results.dat')
    # if os.path.exists(save_path):
    #     with open(save_path, 'rb') as f:
    #         results_dict = pickle.load(f)
    # else : results_dict = {}
    # for model_size in model_size_list:
    #     model_name = f'EleutherAI/pythia-{model_size}'
    #     for revision in range(0, 143000, 1000):
    #         result_key = f'{model_name}_step{revision}'
    #         print(result_key)
    #         if result_key not in results_dict.keys():
    #             step_result = dict()
    #             cache_dir = os.path.join(sys.path[0],f"./model_cache/{model_name}/step{revision}")
    #             model = GPTNeoXForCausalLM.from_pretrained(model_name,revision=f"step{revision}",cache_dir=cache_dir).to(device)
    #             model_config = model.config
    #             tokenizer = AutoTokenizer.from_pretrained(model_name,revision=f"step{revision}",cache_dir=cache_dir)
    #             x_ticks = [f"layer{i + 1}" for i in range(model_config.num_hidden_layers)]
    #             encoded_line = tokenizer.encode(prompt)
    #             codes = tokenizer.convert_ids_to_tokens(encoded_line)
    #             y_ticks = [f"head{i_head}-{c}" for i_head in range(model_config.num_attention_heads) for i, c in enumerate(codes)]
    #             with torch.inference_mode():
    #                 inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    #                 output_and_cache = model(**inputs, output_hidden_states=True, output_attentions=True)
    #                 # 绘制热力图
    #                 ground_attentions = torch.cat(output_and_cache.attentions, dim=0).detach().cpu().numpy()
    #                 ground_attentions = ground_attentions[:, :, -1, :]
    #                 step_result['ground_attentions'] = ground_attentions
    #                 step_result['x_ticks'] = x_ticks
    #                 step_result['y_ticks'] = y_ticks
    #                 # 测试准确率
    #                 result2 = run_eval_harness(model, tokenizer, model_name ,eval_tasks, torch.device(device), 4, sink_token=None)
    #                 for (key) in result2['results'].keys():
    #                     print(key, result2['results'][key]['acc'])
    #                     step_result[key] = result2['results'][key]['acc']
    #             results_dict[result_key] = step_result
    #             with open(save_path, 'wb') as f:
    #                 pickle.dump(results_dict, f)
    #             del model
    
    # 绘制
    # def plot_dist(new_dist, save_path=None, title=None, x=None, datalabel=None, show=None):
    #     colors = sns.color_palette('coolwarm', len(datalabel))
    #     plt.figure(figsize=(20,8), dpi=150)
    #     if not is_not_None(x):
    #         x = list(range(new_dist.shape[0]))
    #     for i, l in enumerate(datalabel):
    #         plt.plot(x, new_dist[:,i].reshape(-1), color=colors[i],label=l)
    #         plt.scatter(x, new_dist[:,i].reshape(-1), color=colors[i])
    #     plt.grid(linestyle="--", alpha=0.5)
    #     plt.xlabel("x")
    #     plt.ylabel("y")
    #     plt.legend(loc="best")  # legend有自己的参数可以控制图例位置
    #     if title is not None:
    #         plt.title(title)
    #     if save_path:
    #         dirs = save_path
    #         if not os.path.exists(dirs): os.makedirs(dirs)
    #         plt.savefig(os.path.join(dirs, f"{title}_dist_plot.png"))
    #     if show:
    #         plt.show() 
    # import pickle
    # eval_tasks = [
    #         "hellaswag",
    #         "piqa",
    #         "winogrande",
    #         "mathqa",
    #     ]
    # model_size_list = reversed(['1b','410m','70m','160m', '1.4b', '14m', '2.8b', '6.9b']) # '1b', '2.8b','70m', '160m','410m', '6.9b'
    # save_path = os.path.join(sys.path[0], './results/pythia_results.dat')
    # if os.path.exists(save_path):
    #     with open(save_path, 'rb') as f:
    #         results_dict = pickle.load(f)
    # else : results_dict = {}
    # save_path = os.path.join(sys.path[0], './result/step_attn')
    # for model_size in model_size_list:
    #     model_name = f'EleutherAI/pythia-{model_size}'
    #     dist ,x = [], []
    #     revisions = [0] + [int(2**i) for i in range(0, 10)]  + list(range(1000, 143000, 1000))
    #     for revision in revisions:
    #         result_key = f'{model_name}_step{revision}'
    #         r = results_dict.get(result_key, None)
    #         if r is not None:
    #             attn = np.mean(r['ground_attentions'][:,:,0])
    #             model_data = np.hstack([attn]+[r[task] for task in eval_tasks])
    #             dist.append(model_data)
    #             x.append(revision)
    #     dist = np.vstack(dist)
    #     print(dist)
    #     plot_dist(dist,save_path=os.path.join(sys.path[0], './result/hidden_attn_heat6'), title=f'pythia-{model_size}',
    #               x = x, datalabel=['ground_attentions']+eval_tasks,show=True)

    ## 第7部分，可视化first token的value
    data = load_dataset('gsm8k', 'main', split='test')
    check_token_id = -1
    model_config = MODEL_CONFIG
    layer_id,layer_list =  10, list(range(0, 27, 5))
     
    change_layer_name, change_name= {'transformer.h.1.mlp.fc_out'},'fc_out'  # ,'transformer.h.0.mlp.fc_out' 'transformer.h.1.attn.out_proj', 'out_proj' # 
    def modify():
        def modify_output(output, layer_name, inputs):
            current_layer = int(layer_name.split(".")[2])
            # if current_layer == edit_layer:
            #     if isinstance(output, tuple):
            #         output[0][:, idx] += fv_vector.to(device)
            #         return output
            #     else:
            #         return output
            # else:
            #     return output
            if layer_name in change_layer_name:
                output[:,0,:] = 0
            return output

        def modify_input(input, layer_name):
            # print(layer_name)
            # for layer_id in layer_ids:
            #     if str(layer_id) in layer_name.split('.'):
            # heads_range = range(n_heads)
            if layer_name in model_config['k_q_names']:
                input[:, 0, :] = input[:, -1, :]
            # input[:, 4, :] = input[:, -1, :]
            return input
        return modify_output, modify_input
    modify_output, modify_input = modify()
    save_path = os.path.join(sys.path[0], './result/hidden_attn_heat7_2')
    model.eval()
    length = 50
    emb = model.get_output_embeddings().weight.data.T.detach().cpu().numpy()

    def draw_dist_heat_map(past_v, v_type, title, save_path):
        save_path = os.path.join(save_path, v_type, title)
        past_v = np.transpose(past_v,(0,2,1,3)).reshape(past_v.shape[0],past_v.shape[2], -1)
        for layer_id in layer_list:
            token_v = past_v[layer_id]
            dist_list, dist_n_list = [], []
            for token_id in range(token_v.shape[0]):
                dist = np.linalg.norm(token_v - token_v[token_id][np.newaxis,:], axis=1)
                dist_list.append(dist)
                dist_normalize = dist/np.sum(dist)
                dist_n_list.append(dist_normalize)
            dist , dist_normalize = np.vstack(dist_list), np.vstack(dist_n_list)
            plt_heatMap_sns(dist, title=f"{title}_layer{layer_id}_without_normalize", x_ticks=codes, y_ticks=codes
                            , show=True, save_path=save_path)
            plt_heatMap_sns(dist_normalize,title=f"{title}_layer{layer_id}_normalize", x_ticks=codes, y_ticks=codes
                            , show=True, save_path=save_path)

    def get_max_code(ground_attentions,codes):
            ground_attentions = ground_attentions.reshape(-1, ground_attentions.shape[-1])
            ground_attentions_mean = np.mean(ground_attentions, axis=0)
            sort_list = np.argsort(ground_attentions_mean)[-3:]
            index, max_num = np.argmax(ground_attentions_mean),np.max(ground_attentions_mean)
            result = []
            for index in sort_list:
                result.append([int(index), codes[index], float(ground_attentions_mean[index])])
            return result
    
    def get_max_attn_token(modify_input=None, second_token=[-1], sort_index=-1):
        attn_input_list,attn_output_list, hidden_state_list, values_list, codes =[], [], [],[],[]
        key_list, q_list = [], []
        for text in data[:length]['question']:
            inputs = tokenizer(text, return_tensors="pt").to(device_str)
            encoded_line = tokenizer.encode(text)
            codes_ = tokenizer.convert_ids_to_tokens(encoded_line)

            attn_input ,attn_output,mlp_input,mlp_output, hidden_state, ground_attentions, past_key, past_values, past_qs = try_hook_ground(model, tokenizer, MODEL_CONFIG, text,check_token_id, torch.device(device_str),edit_input=modify_input)
            code_change =get_max_code(ground_attentions,codes_)
            max_index = code_change[sort_index][0]
            hidden_state_list.append(hidden_state[:,max_index][:,np.newaxis,:])
            attn_input_list.append(attn_input[:,max_index][:,np.newaxis,:])
            attn_output_list.append(attn_output[:,max_index][:,np.newaxis,:])
            values_list.append(past_values[:,:,max_index][:,:,np.newaxis,:])
            key_list.append(past_key[:,:,max_index][:,:,np.newaxis,:])
            q_list.append(past_qs[:,:,max_index][:,:,np.newaxis,:])
            codes.append(codes_[max_index]+f'_{max_index}')
            if len(hidden_state_list) in second_token:
                token_id = 3
                hidden_state_list.append(hidden_state[:,token_id][:,np.newaxis,:])
                attn_input_list.append(attn_input[:,token_id][:,np.newaxis,:])
                attn_output_list.append(attn_output[:,token_id][:,np.newaxis,:])
                values_list.append(past_values[:,:,token_id][:,:,np.newaxis,:])
                key_list.append(past_key[:,:,token_id][:,:,np.newaxis,:])
                q_list.append(past_qs[:,:,token_id][:,:,np.newaxis,:])
                codes.append('second '+codes_[token_id])
        return attn_input_list,attn_output_list, hidden_state_list, values_list,key_list,q_list, codes
    
    def convert_to_tokens(indices, tokenizer, extended=False, extra_values_pos=None, strip=True):
        if extended:
            res = [tokenizer.convert_ids_to_tokens([idx])[0] if idx < len(tokenizer) else 
                (f"[pos{idx-len(tokenizer)}]" if idx < extra_values_pos else f"[val{idx-extra_values_pos}]") 
                for idx in indices]
        else:
            res = tokenizer.convert_ids_to_tokens(indices)
        if strip:
            res = list(map(lambda x: x[1:] if x[0] == 'Ġ' else "#" + x, res))
        return res
    def top_tokens(v, k=100, tokenizer=None, only_alnum=False, only_ascii=True, with_values=False, 
               exclude_brackets=False, extended=True, extra_values=None, only_from_list=None):
        v = torch.tensor(deepcopy(v))
        ignored_indices = []
        if only_ascii:
            ignored_indices.extend([key for val, key in tokenizer.vocab.items() if not val.strip('Ġ▁').isascii()])
        if only_alnum: 
            ignored_indices.extend([key for val, key in tokenizer.vocab.items() if not (set(val.strip('Ġ▁[] ')) <= ALNUM_CHARSET)])
        if only_from_list:
            ignored_indices.extend([key for val, key in tokenizer.vocab.items() if val.strip('Ġ▁ ').lower() not in only_from_list])
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
    def get_convert_result(past_values, codes, k=30):
        past_v = np.transpose(past_values,(0,2,1,3)).reshape(past_values.shape[0],past_values.shape[2], -1)
        convert_result = dict()
        for layer_id in layer_list:
            layer_result = dict()
            for code_i in range(len(codes)):
                res = top_tokens(past_v[layer_id][code_i] @ emb, tokenizer=tokenizer, k=k, only_alnum=False)
                layer_result[codes[code_i]] = res
            convert_result[layer_id] = layer_result
        return convert_result
    path = '/scr1/wyang107/projects/LLM/my_projects/actiavtion/pretrain_activation/results/convert'

    # v_type = 'first_token'
    # results = get_max_attn_token()
    # attn_input_list,attn_output_list, hidden_state_list, values_list,key_list,q_list, codes = results
    # attn_input ,attn_output,hidden_state,past_values, past_key, past_qs =  np.concatenate(attn_input_list, axis=1), np.concatenate(attn_output_list, axis=1) ,\
    #     np.concatenate(hidden_state_list, axis=1), np.concatenate(values_list, axis=2), np.concatenate(key_list, axis=2), np.concatenate(q_list, axis=2)
    # for kqv_type, past_v in zip(['past_values','past_key'],[past_values,past_key]):
    #     convert_result = get_convert_result(past_v, codes, k=5)
    #     yaml.dump(convert_result ,open(f"{path}/{v_type}_{kqv_type}_convert_result.yml", "w"))
    #     draw_dist_heat_map(past_v, v_type, f'{kqv_type}_dist', save_path)     
    # # draw_dist(past_key,past_values,past_qs,attn_input,attn_output, None, None,hidden_state, None,'first_token', layer_id,layer_list,save_path)
    
    # v_type = 'first_token_with'
    # results2 = get_max_attn_token(second_token=[1,3,5])
    # attn_input_list,attn_output_list, hidden_state_list, values_list,key_list,q_list, codes = results2
    # attn_input ,attn_output,hidden_state,past_values, past_key, past_qs =  np.concatenate(attn_input_list, axis=1), np.concatenate(attn_output_list, axis=1) ,\
    #     np.concatenate(hidden_state_list, axis=1), np.concatenate(values_list, axis=2), np.concatenate(key_list, axis=2), np.concatenate(q_list, axis=2)
    # draw_dist_heat_map(past_values, 'first_token_with', 'past_values_dist', save_path)
    # for kqv_type, past_v in zip(['past_values','past_key'],[past_values,past_key]):
    #     convert_result = get_convert_result(past_v, codes, k=5)
    #     yaml.dump(convert_result ,open(f"{path}/{v_type}_{kqv_type}_convert_result.yml", "w"))
    #     draw_dist_heat_map(past_v, v_type, f'{kqv_type}_dist', save_path)  
    # # draw_dist(past_key,past_values,past_qs,attn_input,attn_output, None, None,hidden_state, None,'first_token_with', layer_id,layer_list,save_path)
    
    # v_type = 'change_token'
    # results3 = get_max_attn_token(modify_input=modify_input)
    # attn_input_list,attn_output_list, hidden_state_list, values_list,key_list,q_list, codes = results3
    # attn_input ,attn_output,hidden_state,past_values, past_key, past_qs =  np.concatenate(attn_input_list, axis=1), np.concatenate(attn_output_list, axis=1) ,\
    #     np.concatenate(hidden_state_list, axis=1), np.concatenate(values_list, axis=2), np.concatenate(key_list, axis=2), np.concatenate(q_list, axis=2)
    # # draw_dist(past_key,past_values,past_qs,attn_input,attn_output, None, None,hidden_state, None,'change_token', layer_id,layer_list,save_path)
    # for kqv_type, past_v in zip(['past_values','past_key'],[past_values,past_key]):
    #     convert_result = get_convert_result(past_v, codes, k=5)
    #     yaml.dump(convert_result ,open(f"{path}/{v_type}_{kqv_type}_convert_result.yml", "w"))
    #     draw_dist_heat_map(past_v, v_type, f'{kqv_type}_dist', save_path)  

    # v_type = 'change_token_with'
    # results4 = get_max_attn_token(modify_input=modify_input, second_token=[1,3,5])
    # attn_input_list,attn_output_list, hidden_state_list, values_list,key_list,q_list, codes = results4
    # attn_input ,attn_output,hidden_state,past_values, past_key, past_qs =  np.concatenate(attn_input_list, axis=1), np.concatenate(attn_output_list, axis=1) ,\
    #     np.concatenate(hidden_state_list, axis=1), np.concatenate(values_list, axis=2), np.concatenate(key_list, axis=2), np.concatenate(q_list, axis=2)
    # # draw_dist(past_key,past_values,past_qs,attn_input,attn_output, None, None,hidden_state, None,'change_token_with', layer_id,layer_list,save_path)
    # for kqv_type, past_v in zip(['past_values','past_key'],[past_values,past_key]):
    #     convert_result = get_convert_result(past_v, codes, k=5)
    #     yaml.dump(convert_result ,open(f"{path}/{v_type}_{kqv_type}_convert_result.yml", "w"))
    #     draw_dist_heat_map(past_v, v_type, f'{kqv_type}_dist', save_path)  

    # result5, result6 = [],[]
    # for i in range(len(results)):
    #     result5.append(results[i]+results3[i])
    #     result6.append(results2[i]+results3[i])
    
    # v_type = 'fuse_token'
    # attn_input_list,attn_output_list, hidden_state_list, values_list,key_list,q_list, codes = tuple(result5)
    # attn_input ,attn_output,hidden_state,past_values, past_key, past_qs =  np.concatenate(attn_input_list, axis=1), np.concatenate(attn_output_list, axis=1) ,\
    #     np.concatenate(hidden_state_list, axis=1), np.concatenate(values_list, axis=2), np.concatenate(key_list, axis=2), np.concatenate(q_list, axis=2)
    # # draw_dist(past_key,past_values,past_qs,attn_input,attn_output, None, None,hidden_state, None,'fuse_token', layer_id,layer_list,save_path)
    # for kqv_type, past_v in zip(['past_values','past_key'],[past_values,past_key]):
    #     convert_result = get_convert_result(past_v, codes, k=5)
    #     yaml.dump(convert_result ,open(f"{path}/{v_type}_{kqv_type}_convert_result.yml", "w"))
    #     draw_dist_heat_map(past_v, v_type, f'{kqv_type}_dist', save_path)  

    # v_type = 'fuse_token_with'
    # attn_input_list,attn_output_list, hidden_state_list, values_list,key_list,q_list, codes = tuple(result6)
    # attn_input ,attn_output,hidden_state,past_v, past_key, past_qs =  np.concatenate(attn_input_list, axis=1), np.concatenate(attn_output_list, axis=1) ,\
    #     np.concatenate(hidden_state_list, axis=1), np.concatenate(values_list, axis=2), np.concatenate(key_list, axis=2), np.concatenate(q_list, axis=2)
    # # draw_dist(past_key,past_values,past_qs,attn_input,attn_output, None, None,hidden_state, None,'fuse_token_with', layer_id,layer_list,save_path)
    # for kqv_type, past_v in zip(['past_values','past_key'],[past_values,past_key]):
    #     convert_result = get_convert_result(past_v, codes, k=5)
    #     yaml.dump(convert_result ,open(f"{path}/{v_type}_{kqv_type}_convert_result.yml", "w"))
    #     draw_dist_heat_map(past_v, v_type, f'{kqv_type}_dist', save_path)  

    # change values with the first token
    length, change_layer,sort_index= 30, 4, -1#7
    attn_input_list,attn_output_list, hidden_state_list, values_list,key_list,q_list, codes = get_max_attn_token(sort_index=sort_index)
    choose_replace = 'v'# 'k', 'q', 'v'，'k_q'
    def change_attn(change_list):
        eval_tasks = ["hellaswag", "piqa", "winogrande",  "mathqa", ]
        dict_acc_all = dict()
        def modify(choose_token_id=None):
            def modify_output(output, layer_name, inputs):
                current_layer = int(layer_name.split(".")[2])
                if current_layer > change_layer:
                    if is_not_None(choose_token_id): choose = choose_token_id
                    else: choose = random.randint(0, length-1)
                    output[:, 0] = torch.from_numpy(change_list[choose][current_layer].reshape(1,-1)).to(device=output.device)
                return output

            def modify_input(input, layer_name):
                # print(layer_name)
                # for layer_id in layer_ids:
                #     if str(layer_id) in layer_name.split('.'):
                # heads_range = range(n_heads)
                # for i in range(1, input.shape[2]):
                # tmp = input[:, :, 1:, change_token_id]
                # input[:, :, 1:, change_token_id] = input[:, :, 1:, 0]
                # input[:, :, 1:, 0] = tmp
                # input[:, 4, :] = input[:, -1, :]
                return input

            return modify_output, modify_input
        chooses = [None] + list(range(0, length, 5)) # 
        for i in chooses:
            modify_output, modify_input = modify(i)
            with TraceDict(model, layers=MODEL_CONFIG[f'{choose_replace}_names'], edit_output=modify_output, retain_output=False) as ret:
                result2 = run_eval_harness(model, tokenizer, model_name ,eval_tasks, torch.device(device_str), 4, sink_token=None)
                dict_acc = dict()
                for (key) in result2['results'].keys():
                    print(key, result2['results'][key]['acc'])
                    dict_acc[key] = float(result2['results'][key]['acc'])
            if is_not_None(i): dict_acc_all[codes[i]] = dict_acc
            else:  dict_acc_all['random'] = dict_acc
            yaml.dump(dict_acc_all ,open(f"./results/change/change_token_{choose_replace}_from{change_layer}_with{sort_index}.yml", "w"))
    # change_attn(values_list)
    

