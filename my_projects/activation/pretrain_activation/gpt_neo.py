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
from utils.evaluation_lm_eval import run_eval_harness
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap

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

def plot_umap(activations, datalabel, title, save_path, show=False):
    old_shape = activations.shape
    embedding = umap.UMAP().fit_transform(activations.reshape(-1, old_shape[-1]))
    embedding = embedding.reshape(old_shape[:-1]+(2,))
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
                        "attn_hook_names": [f'transformer.h.{layer}.attn.attention.attn_dropout' for layer in
                                            range(model.config.num_hidden_layers)],
                        "k_q_names": [f'transformer.h.{layer}.attn.k_proj' for layer in
                                            range(model.config.num_hidden_layers)] + 
                                    [f'transformer.h.{layer}.attn.q_proj' for layer in
                                            range(model.config.num_hidden_layers)] +
                                    [f'transformer.h.{layer}.attn.v_proj' for layer in
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

def draw_attention(model, tokenizer, MODEL_CONFIG, device, prompt, check_token_id, save_path, title):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_and_cache = model(**inputs, output_hidden_states=True, output_attentions=True)
    ground_attentions = torch.cat(output_and_cache.attentions, dim=0).detach().cpu().numpy()
    ground_attentions = ground_attentions[:, :, check_token_id, :]

    x_ticks = [f"layer{i + 1}" for i in range(MODEL_CONFIG['n_layers'])]
    encoded_line = tokenizer.encode(prompt)
    codes = tokenizer.convert_ids_to_tokens(encoded_line)
    y_ticks = [f"head{i_head}-{c}" for i_head in range(MODEL_CONFIG['n_heads']) for i, c in enumerate(codes)]
    plt_heatMap_sns(ground_attentions.reshape(ground_attentions.shape[0], -1).T,
                    title=title, x_ticks=x_ticks, y_ticks=y_ticks
                    , show=True, save_path=save_path)

def generate(model, tokenizer, device, prompt):
    from transformers import pipeline
    #文本生成
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device, pad_token_id=tokenizer.eos_token_id)
    results= text_generator(prompt, max_length=50, do_sample=False)
    return results[0]['generated_text']

def get_generate(layers,train_layers,is_linear, prompt):
    device = 'cuda:1'
    save_model_path = f'./results/gpt_neo_{layers}_{is_linear}_{train_layers}'
    print("=====================================================")
    print(save_model_path)
    model_name = 'EleutherAI/gpt-neo-1.3B'   # 'EleutherAI/gpt-neo-1.3B' # 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T' # 'EleutherAI/gpt-neo-125m'
    model, tokenizer, _ = load_model(model_name, device=device, layers=layers, train_layers=train_layers, is_linear=is_linear)
    model = model.to(torch.device(device))
    for name, param in model.named_parameters():
        if param.requires_grad:
            # print(torch.isnan(param.grad).any())
            # print('name:{} param grad:{} param requires_grad:{},params:{}'.format(name, param.grad, param.requires_grad,param))
            print('name:{} param requires_grad:{}:{}'.format(name, param.requires_grad,param))
    print("-----------------------------------original_gpt_neo-----------------------------------")
    print(generate(model, tokenizer, device, prompt))
    print("-----------------------------------original_gpt_neo-----------------------------------")
    model.load_state_dict(torch.load(save_model_path+'pth'))
    model = model.to(torch.device(device))
    for name, param in model.named_parameters():
        if param.requires_grad:
            # print(torch.isnan(param.grad).any())
            # print('name:{} param grad:{} param requires_grad:{},params:{}'.format(name, param.grad, param.requires_grad,param))
            print('name:{} param requires_grad:{}:{}'.format(name, param.requires_grad, param))
    print("-----------------------------------train_gpt_neo-----------------------------------")
    print(generate(model, tokenizer, device, prompt))
    print("-----------------------------------train_gpt_neo-----------------------------------")
    print('\n')

if __name__ == "__main__":
    ## 第一部分
    # layers = list(range(18,23))  # [23]
    # train_layers = list(range(24))
    # for i in layers:
    #     train_layers.remove(i)
    # is_linear = 'zero_layer'
    # save_model_path = f'./results/gpt_neo_{layers}_{is_linear}_{train_layers}'

    # device_str = 'cuda:1'

    # torch.cuda.empty_cache()
    # model_name = 'EleutherAI/gpt-neo-1.3B'  # 'EleutherAI/gpt-neo-1.3B' # 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T' # 'EleutherAI/gpt-neo-125m'
    # model, tokenizer, MODEL_CONFIG = load_model(model_name, device=device_str, layers=layers, train_layers=train_layers,
    #                                             is_linear=is_linear)

    # # dataset = load_dataset('cerebras/SlimPajama-627B',split='train[:100]') # TinyLlama
    # dataset = load_dataset('monology/pile-uncopyrighted', split='train[:2%]')  # gpt-neo

    # def process_func(examples):
    #     contents = [e + tokenizer.eos_token for e in examples["text"]]
    #     result = tokenizer(contents, max_length=256, truncation=True)
    #     return result
    # tokenized_ds = dataset.map(process_func, batched=True, remove_columns=dataset.column_names)
    # # print(len(tokenized_ds['input_ids'][2]))
    # dl = DataLoader(tokenized_ds, batch_size=8, collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False), shuffle=True)
    # # print(next(enumerate(dl)))

    # from torch.optim import AdamW
    # from tensorboardX import SummaryWriter
    # writer = SummaryWriter(save_model_path)
    # optimizer = AdamW(model.parameters(), lr=2e-7,  weight_decay=0.01) # eps=1e-5,
    # # for name, param in model.named_parameters():
    # #     # if param.requires_grad:
    # #         # print(torch.isnan(param.grad).any())
    # #         # print('name:{} param grad:{} param requires_grad:{},params:{}'.format(name, param.grad, param.requires_grad,param))
    # #     print('name:{} param requires_grad:{}, detype:{}, device:{}'.format(name, param.requires_grad, param.dtype, param.device))

    # def train(trainloader, epoch=1, log_step=100):
    #     global_step = 0
    #     lowest_loss = 0
    #     save_step = 100000
    #     device, dtype = model.device, model.dtype
    #     for ep in range(epoch):
    #         model.train()
    #         for batch in trainloader:
    #             # inp = batch['input_ids'][0]
    #             # print(inp)
    #             # print(tokenizer.decode(inp))
    #             # tokenizer(tokenizer.decode(inp))
    #             if torch.cuda.is_available():
    #                 batch = {k: v.to(device = device) for k, v in batch.items()}
    #             # with torch.autograd.detect_anomaly():
    #             optimizer.zero_grad()
    #             output = model(**batch)
    #             # for name, param in model.named_parameters():
    #                 # if param.requires_grad:
    #                 #     # print(torch.isnan(param.grad).any())
    #                 #     # print('name:{} param grad:{} param requires_grad:{},params:{}'.format(name, param.grad, param.requires_grad,param))
    #                 #     print('name:{} param requires_grad:{}'.format(name, param.requires_grad))
    #             loss = output.loss
    #             loss.backward()
    #             # torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
    #             optimizer.step()
                
    #             if global_step % log_step == 0:
    #                 print(f"ep: {ep}, global_step: {global_step}, loss: {output.loss.item()}")
    #                 if lowest_loss > output.loss.item():
    #                     lowest_loss = output.loss.item()
    #                     torch.save(model.state_dict(), save_model_path+'pth')
    #                 if  global_step % save_step == 0: 
    #                     torch.save(model.state_dict(), save_model_path+'pth')
    #                 writer.add_scalar('loss', output.loss.item(), global_step=global_step)
    #             global_step += 1
    # train(dl,log_step=100)

    ## 第二部分
    # model1, _, _ = load_model(model_name, device=device_str, layers=layers, train_layers=train_layers,
    #                                             is_linear=is_linear)
    # model1.load_state_dict(torch.load(save_model_path+'.pth'))

    # args = TrainingArguments(
    #     output_dir=save_model_path,
    #     per_device_train_batch_size=4,
    #     gradient_accumulation_steps=8,
    #     logging_steps=1,
    #     # save_strategy='epoch',
    #     # evaluation_strategy='epoch',
    #     # save_total_limit=2,
    #     learning_rate=2e-13,
    #     # weight_decay=0.01,
    #     # num_train_epochs=3,
    #     # load_best_model_at_end=True,
    # )
    # trainer = Trainer(
    #     args=args,
    #     model=model,
    #     train_dataset=tokenized_ds,
    #     data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    # )
    # trainer.train()

    # device = torch.device(device_str)
    # prompt = 'Beats Music is owned by'  # 'Beats Music is owned by', 'The Space Needle is in downtown'
    # target_token = 'Apple'
    # save_path = './result/gpt_neo/'
    # title = 'attentions_gpt_neo'
    # check_token_id = -1
    #
    # draw_attention(model, tokenizer, MODEL_CONFIG, device, prompt, check_token_id, save_path, title)

    ## 第三部分 测试文本生成
    # prompt = "hello, please tell me some skills about passing exams."
    # is_linear = 'linear_atten'
    # for layers in [[23], [22], [21]]:
    #     train_layers = layers
    #     get_generate(layers, train_layers, is_linear,prompt)


    ## 第四部分 测试准确率
    # eval_tasks = [
    #         "hellaswag",
    #         "piqa",
    #         "winogrande",
    #         "mathqa",
    #     ]
    # device = 'cuda:1'
    # model_name = 'EleutherAI/gpt-neo-1.3B'
    # config = AutoConfig.from_pretrained(model_name)
    # is_linear = 'zero_residual'
    # csv_path = f'./csv/gpt_j_{is_linear}.csv'
    # print(csv_path)
    # df = pd.DataFrame(data=None, index=None, columns=eval_tasks)
    # for layer_id in range(config.num_layers): #n_layer num_layers
    #     layers = [layer_id]  # [23]
    #     train_layers = layers
    #     save_model_path = f'./results/gpt_j_{layers}_{is_linear}_{train_layers}'
    #     # 'EleutherAI/gpt-neo-1.3B' # 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T' # 'EleutherAI/gpt-neo-125m' 'EleutherAI/gpt-j-6b'
    #     model, tokenizer, config = load_model(model_name, device=device, layers=layers, train_layers=train_layers, is_linear=is_linear)
    #     # model.load_state_dict(torch.load(save_model_path+'pth'))
    #     model = model.to(torch.device(device))
    #     result2 = run_eval_harness(model, tokenizer, save_model_path ,eval_tasks, torch.device(device), 4, sink_token=None)
    #     print(save_model_path)
    #     # generate(model, tokenizer, device, 'Hello, nice to meet you')
    #     acc_list = []
    #     for (key) in result2['results'].keys():
    #         print(key, result2['results'][key]['acc'])
    #         acc_list.append(result2['results'][key]['acc'])
    #     df.loc[f'layer{layer_id}'] = acc_list
    # print(df)
    # df.T.to_csv(path_or_buf=os.path.join(sys.path[0], csv_path), columns=df.T.columns)

    ## 第五部分 绘制热力图
    # device_str = 'cuda:2'
    # layers, train_layers, is_linear = [] , [], 'linear_atten'
    # model_name = 'openai-community/gpt2-xl' #'EleutherAI/gpt-j-6b' # 'EleutherAI/gpt-neo-1.3B' # 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T' # 'EleutherAI/gpt-neo-125m'
    # model, tokenizer, MODEL_CONFIG = load_model(model_name, device=device_str, layers=layers, train_layers=train_layers,
    #                                             is_linear=is_linear)
    # model_config = AutoConfig.from_pretrained(model_name)
    # n_layers = model_config.n_layer
    # n_heads = model_config.n_head
    # check_token_id = -1

    # prompt = '\nBeats Music is owned by' # 'Beats Music is owned by', 'The Space Needle is in downtown'
    # target_token = 'Apple'
    # x_ticks = [f"layer{i + 1}" for i in range(n_layers)]
    # save_path = os.path.join(sys.path[0], './result')
    # encoded_line = tokenizer.encode(prompt)
    # codes = tokenizer.convert_ids_to_tokens(encoded_line)

    # model.eval()
    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # output_and_cache = model(**inputs, output_hidden_states=True, output_attentions=True)
    # # 绘制热力图
    # ground_attentions = torch.cat(output_and_cache.attentions, dim=0).detach().cpu().numpy()
    # ground_attentions = ground_attentions[:, :, check_token_id, :]
    # y_ticks = [f"head{i_head}-{c}" for i_head in range(n_heads) for i, c in enumerate(codes)]
    # plt_heatMap_sns(ground_attentions.reshape(ground_attentions.shape[0], -1).T,
    #                 title="gpt2_ground_attentions", x_ticks=x_ticks, y_ticks=y_ticks
    #                 , show=True, save_path=save_path)
    # # 除去第一个token的attend之后，再归一化
    # sum_last = (1-ground_attentions[:,:,0])[:,:,np.newaxis]
    # ground_attentions = ground_attentions[:,:,1:]
    # ground_attentions_sum = ground_attentions / sum_last
    # y_ticks = [f"head{i_head}-{c}" for i_head in range(n_heads) for i, c in enumerate(codes[1:])]
    # plt_heatMap_sns(ground_attentions_sum.reshape(ground_attentions_sum.shape[0], -1).T,
    #                 title="gpt2_ground_attentions2", x_ticks=x_ticks, y_ticks=y_ticks
    #                 , show=True, save_path=save_path)
    
    ## 第六部分 获取hidden_state
    # device_str = 'cuda:2'
    # layers, train_layers, is_linear = [] , [], 'linear_atten'
    # model_name = 'EleutherAI/gpt-j-6b' # 'EleutherAI/gpt-neo-1.3B' # 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T' # 'EleutherAI/gpt-neo-125m'
    # model, tokenizer, MODEL_CONFIG = load_model(model_name, device=device_str, layers=layers, train_layers=train_layers,
    #                                             is_linear=is_linear)
    # model_config = AutoConfig.from_pretrained(model_name)
    # n_layers = model_config.n_layer
    # n_heads = model_config.n_head
    # check_token_id = -1

    # prompt = 'The Space Needle is in downtown' # 'Beats Music is owned by', 'The Space Needle is in downtown'
    # target_token = 'Apple'
    # save_path = os.path.join(sys.path[0], './result')
    # encoded_line = tokenizer.encode(prompt)
    # codes = tokenizer.convert_ids_to_tokens(encoded_line)

    # model.eval()
    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # output_and_cache = model(**inputs, output_hidden_states=True, output_attentions=True)   
    # hidden_state = torch.cat(output_and_cache.hidden_states, dim=0).detach().cpu().numpy()

    # # 查看每一个token与其他token的l2距离
    # print(hidden_state)
    # x_ticks = [f"layer{i + 1}" for i in range(n_layers)]
    # y_ticks = [f"{c}" for i, c in enumerate(codes)]
    # for token_id in range(len(codes)):
    #     last_token_hidden_state = hidden_state[:,token_id,:][:,np.newaxis,:]
    #     dist = np.linalg.norm(last_token_hidden_state - hidden_state, axis=2)
    #     print(f"difference2:\n{dist}")
    #     plt_heatMap_sns(dist,title=f"the_{token_id}_token_dist", x_ticks=y_ticks, y_ticks=x_ticks, show=True, save_path=save_path)
    
    # # todo 查看每个head内，最后一个token对于其他token的距离

    # # 获取其他token与第一个token之间的距离
    # def get_first_token(prompt, model, token_id=0):
    #     model.eval()
    #     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    #     output_and_cache = model(**inputs, output_hidden_states=True, output_attentions=True)   
    #     hidden_state = torch.cat(output_and_cache.hidden_states, dim=0).detach().cpu().numpy()
    #     return hidden_state[:,token_id,:]

    # hidden_state = get_first_token(prompt=prompt, model=model)
    # for i, c in enumerate(codes):
    #     if i == 0: continue
    #     print(f"__________token_id:{i},{c}________________")
    #     hidden_state2 = get_first_token(prompt=prompt, model=model, token_id=i)
    #     dist = np.linalg.norm(hidden_state - hidden_state2, axis=1)
    #     print(f"difference:\n{dist}")
    #     if i != len(codes):
    #         hidden_state3 = get_first_token(prompt=prompt, model=model, token_id=i+1)
    #         dist = np.linalg.norm(hidden_state3 - hidden_state2, axis=1)
    #         print(f"difference2:\n{dist}")

    # # for suffix in ['\n', 'So ', 'Ask ', 'Answer ']:
    # #     print(f"+++++++++++++++++++++++++++++{suffix}+++++++++++++++++++++++")
    # #     for i, c in enumerate(codes):
    # #         print(f"__________token_id:{i},{c}________________")
    # #         hidden_state = get_first_token(prompt=prompt, model=model, token_id=i)
    # #         hidden_state2 = get_first_token(prompt=suffix+prompt, model=model, token_id=i)
    # #         dist = np.linalg.norm(hidden_state - hidden_state2, axis=1)
    # #         print(f"same_token_id:\n{dist}")
    # #         hidden_state2 = get_first_token(prompt=suffix+prompt, model=model, token_id=i+1)
    # #         dist = np.linalg.norm(hidden_state - hidden_state2, axis=1)
    # #         print(f"same_token_meaning:\n{dist}")
    
    
    # # for suffix in ['\n', 'The ', 'My ', 'Ask ', 'Answer ']:
    # #     print(f"___________________________{suffix}________________________")
    # #     hidden_state2 = get_first_token(prompt=suffix+prompt, model=model)
    # #     dist = np.linalg.norm(hidden_state - hidden_state2, axis=1)
    # #     print(dist)
    # #     hidden_state3 = get_first_token(prompt=suffix+prompt, model=model, token_id=1)
    # #     dist2 = np.linalg.norm(hidden_state - hidden_state3, axis=1)
    # #     print(dist2)
    # #     hidden_state4 = get_first_token(prompt=prompt, model=model, token_id=1)
    # #     dist2 = np.linalg.norm(hidden_state4 - hidden_state3, axis=1)
    # #     print(dist2)

    # from counterfact import CounterFactDataset
    # dataset = CounterFactDataset(os.path.join(sys.path[0], './data/'))
    # for data in dataset[:10]:
    #     for prompt in data['neighborhood_prompts'][0:]: #'neighborhood_prompts']:
    #         target = data['requested_rewrite']['target_true']['str']
    #         hidden_state = get_first_token(prompt=prompt, model=model)
    #         hidden_state2 = get_first_token(prompt="\n"+prompt, model=model)
    #         hidden_state3 = get_first_token(prompt="Ask "+prompt, model=model)
    #         dist1 = np.linalg.norm(hidden_state - hidden_state2, axis=1)
    #         dist2 = np.linalg.norm(hidden_state - hidden_state3, axis=1)
    #         print(dist1)
    #         print(dist2)

    # 第七部分，重新训练一个网络，其中记录attention热力图和准确率
    # # layers = list(range(18,23))  # [23]
    # # train_layers = list(range(24))
    # # for i in layers:
    # #     train_layers.remove(i)
    # # is_linear = 'zero_layer'
    # # save_model_path = f'./results/gpt_neo_{layers}_{is_linear}_{train_layers}'
    # model_name = 'openai-community/gpt2-xl'  # 'EleutherAI/gpt-neo-1.3B' # 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T' # 'EleutherAI/gpt-neo-125m'
    # # model, tokenizer, MODEL_CONFIG = load_model(model_name, device=device_str, layers=layers, train_layers=train_layers,
    # #                                             is_linear=is_linear)
    
    # start, end = 2,4
    # last_start, last_end, last_global_step = 0, 2, 300000
    # dir = os.path.join(sys.path[0], "./results/gpt2")
    # os.makedirs(dir, exist_ok=True)
    # max_length = 256

    # eval_tasks = [
    #         "hellaswag",
    #         "piqa",
    #         "winogrande",
    #         "mathqa",
    #     ]

    # torch.cuda.empty_cache()
    # device_str = 'cuda:2'
    # model_config = AutoConfig.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.pad_token = tokenizer.eos_token
    # model = GPT2LMHeadModel(model_config).to(torch.device(device_str))

    # prompt = 'Beats Music is owned by' # 'Beats Music is owned by', 'The Space Needle is in downtown'
    # target_token = 'Apple'
    # x_ticks = [f"layer{i + 1}" for i in range(model_config.n_layer)]
    # save_pic_path = os.path.join(dir, 'atten_pic')
    # encoded_line = tokenizer.encode(prompt)
    # codes = tokenizer.convert_ids_to_tokens(encoded_line)
    # y_ticks = [f"head{i_head}-{c}" for i_head in range(model_config.n_head) for i, c in enumerate(codes)]
    
    # if last_start is not None:
    #     model.load_state_dict(torch.load(os.path.join(dir,'model', f'gpt_2_start{last_start}_end{last_end}_global_step{last_global_step}')+'.pth'))
    # # dataset = load_dataset('cerebras/SlimPajama-627B',split='train[:100]') # TinyLlama
    # dataset = load_dataset('monology/pile-uncopyrighted', split=f'train[{start}%:{end}%]')  # gpt-neo

    # def process_func(examples):
    #     contents = [e + tokenizer.eos_token for e in examples["text"]]
    #     result = tokenizer(contents, max_length=max_length, truncation=True)
    #     return result
    # tokenized_ds = dataset.map(process_func, batched=True, remove_columns=dataset.column_names)
    # # print(len(tokenized_ds['input_ids'][2]))
    # dl = DataLoader(tokenized_ds, batch_size=6, collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False), shuffle=True)
    # # print(next(enumerate(dl)))

    # from torch.optim import AdamW
    # from tensorboardX import SummaryWriter
    # os.makedirs(os.path.join(dir, 'log'), exist_ok=True)
    # writer = SummaryWriter(os.path.join(dir, 'log'))
    # optimizer = AdamW(model.parameters(), lr=1e-7,  weight_decay=0.01) # eps=1e-5,
    # # for name, param in model.named_parameters():
    # #     # if param.requires_grad:
    # #         # print(torch.isnan(param.grad).any())
    # #         # print('name:{} param grad:{} param requires_grad:{},params:{}'.format(name, param.grad, param.requires_grad,param))
    # #     print('name:{} param requires_grad:{}, detype:{}, device:{}'.format(name, param.requires_grad, param.dtype, param.device))

    # def train(trainloader, epoch=1, log_step=10000):
    #     save_step = 200000
    #     global_step = last_global_step+1
    #     lowest_loss = 0
    #     device, dtype = model.device, model.dtype
    #     for ep in range(epoch):
    #         for batch in trainloader:
    #             model.train()
    #             if torch.cuda.is_available():
    #                 batch = {k: v.to(device = device) for k, v in batch.items()}
    #             # with torch.autograd.detect_anomaly():
    #             optimizer.zero_grad()
    #             output = model(**batch)
    #             # for name, param in model.named_parameters():
    #                 # if param.requires_grad:
    #                 #     # print(torch.isnan(param.grad).any())
    #                 #     # print('name:{} param grad:{} param requires_grad:{},params:{}'.format(name, param.grad, param.requires_grad,param))
    #                 #     print('name:{} param requires_grad:{}'.format(name, param.requires_grad))
    #             loss = output.loss
    #             loss.backward()
    #             # torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
    #             optimizer.step()
                
    #             save_model_path =os.path.join(dir, 'model',f'gpt_2_start{start}_end{end}_global_step{global_step}')
    #             os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    #             if global_step % log_step == 0:
    #                 print(f"ep: {ep}, global_step: {global_step}, loss: {output.loss.item()}")
    #                 # if lowest_loss > output.loss.item():
    #                 #     lowest_loss = output.loss.item()
    #                 #     torch.save(model.state_dict(), save_model_path+'.pth')
    #                 with torch.inference_mode():
    #                     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    #                     output_and_cache = model(**inputs, output_hidden_states=True, output_attentions=True)
    #                     # 绘制热力图
    #                     ground_attentions = torch.cat(output_and_cache.attentions, dim=0).detach().cpu().numpy()
    #                     ground_attentions = ground_attentions[:, :, -1, :]
    #                     plt_heatMap_sns(ground_attentions.reshape(ground_attentions.shape[0], -1).T,
    #                                     title=f"gpt2_attentions_global_step{global_step}", x_ticks=x_ticks, y_ticks=y_ticks
    #                                     , show=True, save_path=save_pic_path)
    #                     atten_score = np.mean(ground_attentions[:,:,0])
    #                     writer.add_scalar('atten_score', atten_score, global_step=global_step)
    #                     # 测试准确率
    #                     result2 = run_eval_harness(model, tokenizer, save_model_path ,eval_tasks, torch.device(device), 4, sink_token=None)
    #                     for (key) in result2['results'].keys():
    #                         print(key, result2['results'][key]['acc'])
    #                         writer.add_scalar(key, result2['results'][key]['acc'], global_step=global_step)
    #             if  global_step % save_step == 0: 
    #                 torch.save(model.state_dict(), save_model_path+'.pth')
    #             writer.add_scalar('loss', output.loss.item(), global_step=global_step)                     
    #             global_step += 1
    # train(dl)

    # # 使用别人已经训练好的模型
    from transformers import GPTNeoXForCausalLM, AutoTokenizer
    import pickle
    model_size_list = ['14m', '2.8b','70m','1b','160m','410m', '2.8b','6.9b',] # , , '6.9b','1.4b',
    device = 'cuda:3'
    prompt = 'The Space Needle is in downtown' # 'Beats Music is owned by', 'The Space Needle is in downtown'
    target_token = 'Apple'
    eval_tasks = [
            "hellaswag",
            "piqa",
            "winogrande",
            "mathqa",
        ]
    save_path = os.path.join(sys.path[0], './results/pythia_results.dat')
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            results_dict = pickle.load(f)
    else : results_dict = {}
    for model_size in model_size_list:
        model_name = f'EleutherAI/pythia-{model_size}'
        for revision in [int(2**i) for i in range(0, 10)] + list(range(0, 143000, 5000)):
            result_key = f'{model_name}_step{revision}'
            print(result_key)
            if result_key not in results_dict.keys():
                step_result = dict()
                cache_dir = os.path.join(sys.path[0],f"./model_cache/{model_name}/step{revision}")
                model = GPTNeoXForCausalLM.from_pretrained(model_name,revision=f"step{revision}",cache_dir=cache_dir).to(device)
                model_config = model.config
                tokenizer = AutoTokenizer.from_pretrained(model_name,revision=f"step{revision}",cache_dir=cache_dir)
                x_ticks = [f"layer{i + 1}" for i in range(model_config.num_hidden_layers)]
                encoded_line = tokenizer.encode(prompt)
                codes = tokenizer.convert_ids_to_tokens(encoded_line)
                y_ticks = [f"head{i_head}-{c}" for i_head in range(model_config.num_attention_heads) for i, c in enumerate(codes)]
                with torch.inference_mode():
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    output_and_cache = model(**inputs, output_hidden_states=True, output_attentions=True)
                    # 绘制热力图
                    ground_attentions = torch.cat(output_and_cache.attentions, dim=0).detach().cpu().numpy()
                    ground_attentions = ground_attentions[:, :, -1, :]
                    step_result['ground_attentions'] = ground_attentions
                    step_result['x_ticks'] = x_ticks
                    step_result['y_ticks'] = y_ticks
                    # 测试准确率
                    result2 = run_eval_harness(model, tokenizer, model_name ,eval_tasks, torch.device(device), 4, sink_token=None)
                    for (key) in result2['results'].keys():
                        print(key, result2['results'][key]['acc'])
                        step_result[key] = result2['results'][key]['acc']
                results_dict[result_key] = step_result
                with open(save_path, 'wb') as f:
                    pickle.dump(results_dict, f)
                del model
                        

    # 绘制
    # import pickle
    # eval_tasks = [
    #         "hellaswag",
    #         "piqa",
    #         "winogrande",
    #         "mathqa",
    #     ]
    # model_size_list = reversed(['1b','410m','70m','160m']) # '1b', '2.8b','70m', '160m','410m', '6.9b'
    # save_path = os.path.join(sys.path[0], './results/pythia_results.dat')
    # if os.path.exists(save_path):
    #     with open(save_path, 'rb') as f:
    #         results_dict = pickle.load(f)
    # else : results_dict = {}
    # save_path = os.path.join(sys.path[0], './result/step_attn')
    # for model_size in model_size_list:
    #     model_name = f'EleutherAI/pythia-{model_size}'
    #     dist = []
    #     for revision in range(0, 143000, 1000):
    #         result_key = f'{model_name}_step{revision}'
    #         r = results_dict.get(result_key, None)
    #         if r is not None:
    #             attn = np.mean(r['ground_attentions'][:,:,0])
    #             model_data = np.hstack([attn]+[r[task] for task in eval_tasks])
    #             dist.append(model_data)
    #     dist = np.vstack(dist)
    #     print(dist)
    #     plot_dist(dist,save_path=save_path, title=f'pythia-{model_size}',datalabel=['ground_attentions']+eval_tasks,show=True)


    ## 第八部分，对hidden state进行head分开，计算记录与attention热力图 deprecated
    # device_str = 'cuda:1'
    # layers, train_layers, is_linear = [] , [], 'linear_atten'
    # model_name = 'EleutherAI/gpt-neo-1.3B' # 'openai-community/gpt2-xl' #'EleutherAI/gpt-j-6b' # 'EleutherAI/gpt-neo-1.3B' # 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T' # 'EleutherAI/gpt-neo-125m'
    # model, tokenizer, MODEL_CONFIG = load_model(model_name, device=device_str, layers=layers, train_layers=train_layers,
    #                                             is_linear=is_linear)
    # model_config = AutoConfig.from_pretrained(model_name)
    # n_layers = model_config.num_layers
    # n_heads = model_config.num_heads
    # check_token_id = -1

    # prompt = 'Beats Music is owned by'# 'Beats Music is owned by', 'The Space Needle is in downtown'
    # target_token = 'Apple'
    # x_ticks = [f"layer{i + 1}" for i in range(n_layers)]
    # save_path = os.path.join(sys.path[0], './result')
    # encoded_line = tokenizer.encode(prompt)
    # codes = tokenizer.convert_ids_to_tokens(encoded_line)
    # y_ticks = [f"head{i_head}-{c}" for i_head in range(n_heads) for i, c in enumerate(codes)]

    # model.eval()
    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # output_and_cache = model(**inputs, output_hidden_states=True, output_attentions=True)
    # # 绘制热力图
    # ground_attentions = torch.cat(output_and_cache.attentions, dim=0).detach().cpu().numpy()
    # hidden_state = torch.cat(output_and_cache.hidden_states, dim=0).detach().cpu().numpy()

    # new_shape = hidden_state.shape[:-1] + (n_heads, -1)
    # head_states = np.transpose(np.reshape(hidden_state,newshape=new_shape),(0,2,1,3))[1:]
    
    # cos_coeff = np.zeros(head_states.shape[:-1])
    # from sklearn.metrics.pairwise import cosine_similarity
    # for layer_id_ in range(head_states.shape[0]):
    #     for head_id in range(head_states.shape[1]):
    #         other,last = head_states[layer_id_, head_id], head_states[layer_id_, head_id, -1][np.newaxis, :]
    #         s = cosine_similarity(other, last).reshape(-1)
    #         cos_coeff[layer_id_,head_id,-1] = 0
    #         cos_coeff[layer_id_, head_id] = s + 1
            
    # # new_shape = hidden_state.size()[:-1] + (num_heads, attn_head_size)
    # # tensor = tensor.view(new_shape)
    # plt_heatMap_sns(cos_coeff.reshape(cos_coeff.shape[0], -1).T,
    #                 title="gpt2_cos_coeff", x_ticks=x_ticks, y_ticks=y_ticks
    #                 , show=True, save_path=save_path)
    
    # dist = np.linalg.norm(head_states - head_states[:,:,-1][:,:,np.newaxis,:], axis=3) 
    # dist = dist/np.sum(dist, axis=2)[:,:,np.newaxis]
    # # new_shape = hidden_state.size()[:-1] + (num_heads, attn_head_size)
    # # tensor = tensor.view(new_shape)
    # plt_heatMap_sns(dist.reshape(dist.shape[0], -1).T,
    #                 title="gpt2_dist", x_ticks=x_ticks, y_ticks=y_ticks
    #                 , show=True, save_path=save_path)


    # ground_attentions = ground_attentions[:, :, check_token_id, :]
    # plt_heatMap_sns(ground_attentions.reshape(ground_attentions.shape[0], -1).T,
    #                 title="gpt2_attentions", x_ticks=x_ticks, y_ticks=y_ticks
    #                 , show=True, save_path=save_path)

    # # 第九部分 替换第一个token的hidden state为last token的hidden state, 查看attn热力图
    # device_str = 'cuda:1'
    # layers, train_layers, is_linear = [] , [], 'linear_atten'
    # model_name = 'EleutherAI/gpt-j-6b'  # facebook/opt-13b# 'mistralai/Mistral-7B-v0.1' 'openai-community/gpt2-xl' #'EleutherAI/gpt-j-6b' # 'EleutherAI/gpt-neo-1.3B' # 'EleutherAI/gpt-neo-125m'
    # model, tokenizer, MODEL_CONFIG = load_model(model_name, device=device_str, layers=layers, train_layers=train_layers,
    #                                             is_linear=is_linear, show_params=False)
    
    # n_layers, n_heads = MODEL_CONFIG['n_layers'], MODEL_CONFIG['n_heads']
    # prompt = 'Beats Music is owned by' # 'The Space Needle is in downtown' # 'Beats Music is owned by', 'Beats Music is owned by Apple and the Space Needle is in downtown'
    # target_token = 'Apple'
    # x_ticks = [f"layer{i + 1}" for i in range(n_layers)]
    # save_path = os.path.join(sys.path[0], './result/hidden_attn_heat')
    # encoded_line = tokenizer.encode(prompt)
    # codes = tokenizer.convert_ids_to_tokens(encoded_line)
    # y_ticks = [f"head{i_head}-{c}" for i_head in range(n_heads) for i, c in enumerate(codes)]
    # check_token_id = -1

    # def try_hook4(model, tokenizer, model_config, prompt, check_token_id, device):

    #     def modify(layer_ids, token_id, n_heads, check_token_id, device):
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
    #             return output

    #         def modify_input(input, layer_name):
    #             # print(layer_name)
    #             # for layer_id in layer_ids:
    #             #     if str(layer_id) in layer_name.split('.'):
    #             # heads_range = range(n_heads)
    #             input[:, 0, :] = input[:, 3, :]
    #             # input[:, 4, :] = input[:, -1, :]
    #             return input

    #         return modify_output, modify_input
    #     model.eval()
    #     inputs = tokenizer(prompt, return_tensors="pt").to(device)
    #     with TraceDict2(model, layers=model_config['q_names'], retain_output=True) as ret:
    #         output_and_cache = model(**inputs, output_hidden_states=True, output_attentions=True)
    #         q_list = [ret[q].output for q in model_config['q_names']]
    #         past_qs = torch.cat(q_list, dim=0).detach().cpu().numpy()
    #         past_qs = np.transpose(np.reshape(past_qs,newshape=past_qs.shape[:-1]+(n_heads, -1)),(0,2,1,3))
    #     ground_attentions = torch.cat(output_and_cache.attentions, dim=0).cpu()
    #     hidden_state = torch.cat(output_and_cache.hidden_states, dim=0).detach().cpu().numpy()
    #     past_key = torch.cat([key_values[0] for key_values in output_and_cache.past_key_values], dim=0).detach().cpu().numpy()
    #     past_values = torch.cat([key_values[1] for key_values in output_and_cache.past_key_values], dim=0).detach().cpu().numpy()

    #     token_id, layer_ids = 0, range(3, model_config['n_layers']-1)
    #     modify_output, modify_input = modify(layer_ids, token_id, model_config['n_heads'],  check_token_id , device)
    #     with TraceDict2(model, layers=model_config['k_q_names'], edit_input=modify_input,
    #                     edit_output=modify_output, retain_output=True) as ret:
    #         output_and_cache = model(**inputs,output_attentions=True,output_hidden_states=True)
    #         change_attentions = torch.stack(output_and_cache.attentions, dim=0)[:,0].cpu()
    #         change_hidden_state = torch.cat(output_and_cache.hidden_states, dim=0).detach().cpu().numpy()
    #         q_list = [ret[q].output for q in model_config['q_names']]
    #         change_past_qs = torch.cat(q_list, dim=0).detach().cpu().numpy()
    #         change_past_qs = np.transpose(np.reshape(change_past_qs,newshape=change_past_qs.shape[:-1]+(n_heads, -1)),(0,2,1,3))
    #         change_past_key = torch.cat([key_values[0] for key_values in output_and_cache.past_key_values], dim=0).detach().cpu().numpy()
    #         change_past_values = torch.cat([key_values[1] for key_values in output_and_cache.past_key_values], dim=0).detach().cpu().numpy()
    #     return hidden_state, ground_attentions.detach().numpy(), change_hidden_state,change_attentions.detach().numpy(), past_key, past_values, past_qs,change_past_key, change_past_values, change_past_qs
    # hidden_state,ground_attentions, change_hidden_state,change_attentions, past_key, past_values,past_qs,change_past_key, change_past_values, change_past_qs = try_hook4(model, tokenizer, MODEL_CONFIG, prompt,check_token_id, torch.device(device_str))
    # ground_attentions, change_attentions = ground_attentions[:, :, check_token_id, :],change_attentions[:, :, check_token_id, :]
    # plt_heatMap_sns(ground_attentions.reshape(ground_attentions.shape[0], -1).T,
    #                 title="gpt2_attentions", x_ticks=x_ticks, y_ticks=y_ticks
    #                 , show=True, save_path=save_path)
    # plt_heatMap_sns(change_attentions.reshape(change_attentions.shape[0], -1).T,
    #                 title="gpt2_change_attentions", x_ticks=x_ticks, y_ticks=y_ticks
    #                 , show=True, save_path=save_path)

    # new_shape = hidden_state.shape[:-1] + (n_heads, -1)

    # # hidden state的热力图，折线图，数据值
    # head_states = np.transpose(np.reshape(hidden_state,newshape=new_shape),(0,2,1,3))[1:]
    # dist = np.linalg.norm(head_states - head_states[:,:,-1][:,:,np.newaxis,:], axis=3)
    # plt_heatMap_sns(dist.reshape(dist.shape[0], -1).T,
    #                 title="gpt2_head_dist_without_normalize", x_ticks=x_ticks, y_ticks=y_ticks
    #                 , show=True, save_path=save_path)
    # dist_normalize = dist/np.sum(dist, axis=2)[:,:,np.newaxis]
    # # dist[:, :, (0,)] = 0
    # # new_shape = hidden_state.size()[:-1] + (num_heads, attn_head_size)
    # # tensor = tensor.view(new_shape)
    # plt_heatMap_sns(dist_normalize.reshape(dist_normalize.shape[0], -1).T,
    #                 title="gpt2_head_dis", x_ticks=x_ticks, y_ticks=y_ticks
    #                 , show=True, save_path=save_path)
    # dist2 = np.linalg.norm(hidden_state - hidden_state[:,-1][:, np.newaxis,:], axis=2)[1:,np.newaxis,]
    # new_dist = np.concatenate((dist,dist2), axis=1)[:,:,0]
    # plot_dist(new_dist,save_path=save_path, title='dist',
    #           datalabel=[f"head{i_head}" for i_head in range(n_heads)] + ['layer'],
    #           show=True)

    # cos_coeff = np.zeros(head_states.shape[:-1])
    # from sklearn.metrics.pairwise import cosine_similarity
    # for layer_id_ in range(head_states.shape[0]):
    #     for head_id in range(head_states.shape[1]):
    #         other,last = head_states[layer_id_, head_id], head_states[layer_id_, head_id, -1][np.newaxis, :]
    #         s = cosine_similarity(other, last).reshape(-1)
    #         cos_coeff[layer_id_,head_id,-1] = 0
    #         cos_coeff[layer_id_, head_id] = s + 1
    # plt_heatMap_sns(cos_coeff.reshape(cos_coeff.shape[0], -1).T,
    #                 title="gpt2_head_cos_coeff", x_ticks=x_ticks, y_ticks=y_ticks
    #                 , show=True, save_path=save_path)
    
    
    # change_head_states = np.transpose(np.reshape(change_hidden_state,newshape=new_shape),(0,2,1,3))[1:]
    #  # hidden state的热力图，折线图，数据值
    # change_dist = np.linalg.norm(change_head_states - change_head_states[:,:,-1][:,:,np.newaxis,:], axis=3)
    # plt_heatMap_sns(change_dist.reshape(change_dist.shape[0], -1).T,
    #                 title="gpt2_change_head_dist_without_normalize", x_ticks=x_ticks, y_ticks=y_ticks
    #                 , show=True, save_path=save_path)
    # dist_normalize = change_dist/np.sum(change_dist, axis=2)[:,:,np.newaxis]
    # # dist[:, :, (0,)] = 0
    # # new_shape = hidden_state.size()[:-1] + (num_heads, attn_head_size)
    # # tensor = tensor.view(new_shape)
    # plt_heatMap_sns(dist_normalize.reshape(dist_normalize.shape[0], -1).T,
    #                 title="gpt2_change_head_dist", x_ticks=x_ticks, y_ticks=y_ticks
    #                 , show=True, save_path=save_path)
    # change_dist2 = np.linalg.norm(change_hidden_state - change_hidden_state[:,-1][:, np.newaxis,:], axis=2)[1:,np.newaxis,]
    # new_dist = np.concatenate((change_dist,change_dist2), axis=1)[:,:,4]
    # plot_dist(new_dist,save_path=save_path, title='dist_change_is_token',
    #           datalabel=[f"head{i_head}" for i_head in range(n_heads)] + ['layer'],
    #           show=True)
    # new_dist = np.concatenate((change_dist,change_dist2), axis=1)[:,:,0]
    # plot_dist(new_dist,save_path=save_path, title='dist_change_first_token',
    #           datalabel=[f"head{i_head}" for i_head in range(n_heads)] + ['layer'],
    #           show=True)

    # # def draw_past_key_values(past_values, title=''):
    # #     past_values_dist = np.linalg.norm(past_values - past_values[:,:,0,:][:,:,np.newaxis,:], axis=3) 
    # #     past_values_dist = past_values_dist/np.sum(past_values_dist, axis=2)[:,:,np.newaxis]
    # #     plt_heatMap_sns(past_values_dist.reshape(past_values_dist.shape[0], -1).T,
    # #                     title=f"gpt2_dist_between_first_token_{title}", x_ticks=x_ticks, y_ticks=y_ticks
    # #                     , show=True, save_path=save_path)
        
    # #     cos_coeff = np.zeros(past_values.shape[:-1])
    # #     from sklearn.metrics.pairwise import cosine_similarity
    # #     for layer_id_ in range(past_values.shape[0]):
    # #         for head_id in range(past_values.shape[1]):
    # #             other,last = past_values[layer_id_, head_id], past_values[layer_id_, head_id, 0][np.newaxis, :]
    # #             s = cosine_similarity(other, last).reshape(-1)
    # #             # cos_coeff[layer_id_,head_id,-1] = 0
    # #             cos_coeff[layer_id_, head_id] = s + 1
    # #     plt_heatMap_sns(cos_coeff.reshape(cos_coeff.shape[0], -1).T,
    # #                     title=f"gpt2_coeff_between_first_token_{title}", x_ticks=x_ticks, y_ticks=y_ticks
    # #                     , show=True, save_path=save_path)
    # # draw_past_key_values(past_values, 'values')
    # # draw_past_key_values(past_key, 'keys')

    # # hidden state降为可视化
    # head_label =  [f'head{i}' for i in range(n_heads)]
    # layer_label =  [f'layer{i}' for i in range(n_layers)]
    # plot_umap(head_states, codes,title="head_hidden_states_token_label",save_path=save_path)
    # plot_umap(np.transpose(head_states,(0,2,1,3)),head_label,title="head_hidden_states_head_label",save_path=save_path)
    # plot_umap(np.transpose(head_states,(2,1,0,3)),layer_label,title="head_hidden_states_layer_label",save_path=save_path)
    # plot_umap(hidden_state[1:,np.newaxis,:,:], 
    #           codes,title="hidden_states_token_label",save_path=save_path)
    # plot_umap(np.transpose(hidden_state[1:,np.newaxis,:,:],(2,1,0,3)), 
    #           layer_label,title="hidden_states_layer_label",save_path=save_path)
     
    # plot_umap(head_states[:,:,1:,:], codes[1:],title="head_hidden_states_without_toke0",save_path=save_path)
    # plot_umap(hidden_state[1:,np.newaxis,1:,:], 
    #           codes[1:],title="hidden_states_without_toke0",save_path=save_path)
    
    # # hidden state降为可视化
    # plot_umap(change_head_states, codes,title="change_head_hidden_states_token_label",save_path=save_path)
    # plot_umap(np.transpose(change_head_states,(0,2,1,3)),head_label,title="change_head_hidden_states_head_label",save_path=save_path)
    # plot_umap(np.transpose(change_head_states,(2,1,0,3)),layer_label,title="change_head_hidden_states_layer_label",save_path=save_path)
    # plot_umap(change_hidden_state[1:,np.newaxis,:,:], 
    #           codes,title="change_hidden_states_token_label",save_path=save_path)
    # plot_umap(np.transpose(change_hidden_state[1:,np.newaxis,:,:],(2,1,0,3)), 
    #           layer_label,title="change_hidden_states_layer_label",save_path=save_path)
     
    # plot_umap(change_head_states[:,:,1:,:], codes[1:],title="change_head_hidden_states_without_toke0",save_path=save_path)
    # plot_umap(change_hidden_state[1:,np.newaxis,1:,:], 
    #           codes[1:],title="change_hidden_states_without_toke0",save_path=save_path)
    
    # # value降为可视化
    # plot_umap(past_values, codes,title="head_past_values_token_label",save_path=save_path)
    # plot_umap(np.transpose(past_values,(0,2,1,3)).reshape(past_values.shape[0],past_values.shape[2], -1)[:,np.newaxis,:,:],
    #            codes,title="past_values_token_label",save_path=save_path)
    # plot_umap(past_values[:,:,1:,:], codes[1:],title="head_past_values_without_toke0_token_label",save_path=save_path)
    # plot_umap(np.transpose(past_values,(0,2,1,3)).reshape(past_values.shape[0],past_values.shape[2], -1)[:,np.newaxis,1:,:],
    #            codes[1:],title="past_values_without_toke0_token_label",save_path=save_path)

    # # key降为可视化
    # plot_umap(past_key, codes,title="head_past_key_token_label",save_path=save_path)
    # plot_umap(np.transpose(past_key,(0,2,1,3)).reshape(past_key.shape[0],past_key.shape[2], -1)[:,np.newaxis,:,:],
    #            codes,title="past_key_token_label",save_path=save_path)
    # plot_umap(past_key[:,:,1:,:], codes[1:],title="head_past_key_without_toke0_token_label",save_path=save_path)
    # plot_umap(np.transpose(past_key,(0,2,1,3)).reshape(past_key.shape[0],past_key.shape[2], -1)[:,np.newaxis,1:,:],
    #            codes[1:],title="past_key_without_toke0_token_label",save_path=save_path)

    # # q降为可视化
    # plot_umap(past_qs, codes,title="head_past_qs_token_label",save_path=save_path)
    # plot_umap(np.transpose(past_qs,(0,2,1,3)).reshape(past_qs.shape[0],past_qs.shape[2], -1)[:,np.newaxis,:,:],
    #            codes,title="past_qs_token_label",save_path=save_path)
    # plot_umap(past_qs[:,:,1:,:], codes[1:],title="head_past_qs_without_toke0_token_label",save_path=save_path)
    # plot_umap(np.transpose(past_qs,(0,2,1,3)).reshape(past_qs.shape[0],past_qs.shape[2], -1)[:,np.newaxis,1:,:],
    #            codes[1:],title="past_qs_without_toke0_token_label",save_path=save_path)


    # 第十部分， 可视化token和、\n+token的token的hidden state
    # 获取数据
    # device_str = 'cuda:0'
    # layers, train_layers, is_linear = [] , [], 'linear_atten'
    # model_name = 'EleutherAI/gpt-j-6b'  # facebook/opt-13b# 'mistralai/Mistral-7B-v0.1' 'openai-community/gpt2-xl' #'EleutherAI/gpt-j-6b' # 'EleutherAI/gpt-neo-1.3B' # 'EleutherAI/gpt-neo-125m'
    # model, tokenizer, MODEL_CONFIG = load_model(model_name, device=device_str, layers=layers, train_layers=train_layers,
    #                                             is_linear=is_linear, show_params=False)
    
    # n_layers, n_heads = MODEL_CONFIG['n_layers'], MODEL_CONFIG['n_heads']
    # prompt = 'The Space Needle is in downtown' # 'The Space Needle is in downtown' # 'Beats Music is owned by', 'Beats Music is owned by Apple and the Space Needle is in downtown'
    # # print(tokenizer.vocab)
    # tokens = set()
    # import pickle
    # from counterfact import CounterFactDataset
    # dataset = CounterFactDataset(os.path.join(sys.path[0], './data/'))
    # for data in dataset[:5000]:
    #     for prompt in data['neighborhood_prompts'][0:]: #'neighborhood_prompts']:
    #       tokens.update(prompt.split(' ')) 
    # save_path = os.path.join(sys.path[0], './results/single_token_results.dat')
    # if os.path.exists(save_path):
    #     with open(save_path, 'rb') as f:
    #         results_dict = pickle.load(f)
    # else : results_dict = {}
    # for t in tokens:
    #     inputs = tokenizer(t, return_tensors="pt").to(device_str)
    #     if len(t) ==0 or inputs['input_ids'].shape[1] != 1: continue
    #     if t in results_dict.keys() and len(results_dict[t]) == 6: continue
    #     output_and_cache = model(**inputs, output_hidden_states=True, output_attentions=True)
    #     ground_attentions = torch.cat(output_and_cache.attentions, dim=0).cpu()
    #     hidden_state = torch.cat(output_and_cache.hidden_states, dim=0).detach().cpu().numpy()
    #     past_key = torch.cat([key_values[0] for key_values in output_and_cache.past_key_values], dim=0).detach().cpu().numpy()
    #     past_values = torch.cat([key_values[1] for key_values in output_and_cache.past_key_values], dim=0).detach().cpu().numpy()


    #     inputs = tokenizer("\n"+t, return_tensors="pt").to(device_str)
    #     output_and_cache = model(**inputs, output_hidden_states=True, output_attentions=True)
    #     ground_attentions2 = torch.cat(output_and_cache.attentions, dim=0).cpu()
    #     hidden_state2 = torch.cat(output_and_cache.hidden_states, dim=0).detach().cpu().numpy()
    #     past_key2 = torch.cat([key_values[0] for key_values in output_and_cache.past_key_values], dim=0).detach().cpu().numpy()
    #     past_values2 = torch.cat([key_values[1] for key_values in output_and_cache.past_key_values], dim=0).detach().cpu().numpy()

    #     results_dict[t] = (hidden_state,hidden_state2,past_key,past_values,past_key2,past_values2)
    #     with open(save_path, 'wb') as f:
    #         pickle.dump(results_dict, f)

    # 绘制umap降维
    # import pickle
    # save_path = os.path.join(sys.path[0], './results/single_token_results.dat')
    # if os.path.exists(save_path):
    #     with open(save_path, 'rb') as f:
    #         results_dict = pickle.load(f)
    # else : results_dict = {}
    # count = 0
    # for key, value in results_dict.items():
    #     if len(value) != 6 : count+=1
    # print(count)
    # save_path = os.path.join(sys.path[0], './result/token_between_first_contextual')
    # layer_ids = range(1,29)
    # result_state = None
    # for layer_id in layer_ids:
    #     hidden_state_list,hidden_state_list2 = [],[]
    #     for key, value in results_dict.items():
    #         hidden_state = value[0][layer_id, 0, :]
    #         hidden_state2 = value[1][layer_id, 1, :]
    #         hidden_state_list.append(hidden_state)
    #         hidden_state_list2.append(hidden_state2)
    #     hidden_state = np.vstack(hidden_state_list)[:,np.newaxis, np.newaxis,:]
    #     hidden_state2 = np.vstack(hidden_state_list2)[:, np.newaxis, np.newaxis,:]
    #     result_state = np.concatenate((hidden_state,hidden_state2), axis=2)
    #     plot_umap(result_state, [f'first token in layer{layer_id}', f'contextual tokenin layer{layer_id}'],
    #     title=f"token_between_first_contextual_layer{layer_id}",save_path=save_path)
    
