import torch
import numpy as np
import pandas as pd
torch.set_grad_enabled(False)
import sys
sys.path.append('../')
from utils.model_utils import load_gpt_model_and_tokenizer, make_inputs, decode_tokens, predict_from_input
from utils.trace_utils import TraceDict2
import matplotlib.pyplot as plt
import os
import seaborn as sns
from transformers import AutoConfig, AutoTokenizer
from counterfact import CounterFactDataset
import torch
from utils.evaluation_lm_eval import run_eval_harness

## try hook
def try_hook():
    model_name = 'EleutherAI/gpt-j-6b' # 'EleutherAI/gpt-j-6b'
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name)

    samples = 2
    prompt = 'When Mary and John went to the store, John gave a drink to'
    target_token = 'Mary'
    inp = make_inputs(tokenizer, [prompt] * (samples + 1))
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(model, inp)]
    [answer] = decode_tokens(tokenizer, [answer_t])


    def add_function_vector(edit_layer, fv_vector, device, idx=-1):
        """
        Adds a vector to the output of a specified layer in the model

        Parameters:
        edit_layer: the layer to perform the FV intervention
        fv_vector: the function vector to add as an intervention
        device: device of the model (cuda gpu or cpu)
        idx: the token index to add the function vector at

        Returns:
        add_act: a fuction specifying how to add a function vector to a layer's output hidden state
        """
        def add_act(output, layer_name):
            current_layer = int(layer_name.split(".")[2])
            if current_layer == edit_layer:
                if isinstance(output, tuple):
                    output[0][:, idx] += fv_vector.to(device)
                    return output
                else:
                    return output
            else:
                return output

        return add_act
    def add_act(output, layer_name):
        current_layer = int(layer_name.split(".")[2])
        # if current_layer == edit_layer:
        #     if isinstance(output, tuple):
        #         output[0][:, idx] += fv_vector.to(device)
        #         return output
        #     else:
        #         return output
        # else:
        #     return output
        return output
    n_heads = model_config['n_heads']
    def modify_input(input, layer_name):
        print(layer_name)
        inp = input.reshape(input.size()[:2] + (n_heads,) + (-1,))
        return input
    with TraceDict2(model, layers=model_config['attn_hook_names'], edit_input=modify_input, edit_output=add_act) as ret:
        outputs_exp = model(**inp).logits[:, -1, :]
        probs = torch.softmax(outputs_exp, dim=1).mean(dim=0)[answer_t]
    print(ret)
def try_hook2(model_name, prompt, check_token_id, device):

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device, True)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_and_cache = model(**inputs, output_hidden_states=True, output_attentions=True, use_cache=True)
    ground_attentions = torch.cat(output_and_cache.attentions, dim=0).cpu()
    # the final answer token is:
    soft = torch.softmax(output_and_cache.logits[0, :, :], dim=1)
    result_pos = soft.max(dim=-1).indices.cpu().numpy()
    result_prob = soft.max(dim=-1).values.cpu().numpy()
    result_token = tokenizer.decode(result_pos.tolist())
    n_tokens = soft.shape[0]

    def modify(layer_id, token_id, n_heads, check_token_id, device):
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
            return output
        def modify_input(input, layer_name):
            # print(layer_name)
            if str(layer_id) in layer_name.split('.'):
                heads_range = range(n_heads)
                input[heads_range, heads_range, check_token_id, token_id] = input[heads_range, heads_range, -1, token_id] * 0
            return input
        return modify_output, modify_input


    atten_importance = np.mean(np.zeros(ground_attentions.shape), axis=-2)
    inputs_batch = tokenizer([prompt] * model_config['n_heads'], return_tensors="pt").to(device)
    for layer_id in range(model_config['n_layers']):
        for token_id in range(n_tokens):
            modify_output, modify_input = modify(layer_id, token_id, model_config['n_heads'], check_token_id, device)
            with TraceDict2(model, layers=[model_config['layer_hook_names'][layer_id]], edit_input=modify_input, edit_output=modify_output, retain_output=False) as ret:
                outputs_exp = model(**inputs_batch).logits
                # probs = torch.softmax(outputs_exp, dim=1).mean(dim=0)
                scores = torch.softmax(outputs_exp[:, :, :], dim=-1)
                s = scores[:, check_token_id, result_pos[check_token_id]]
                mask = torch.max(scores,dim=-1).indices[:,check_token_id] != result_pos[check_token_id]
                s[mask] = 0
                s = s.cpu().numpy()

                best_score = result_prob[check_token_id]
                prob = (s- best_score) / best_score
                atten_importance[layer_id, :, token_id] = prob
    return ground_attentions[:,:,check_token_id,:].numpy(), atten_importance


def try_hook3(model, tokenizer, model_config, prompt, check_token_id, device):

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
            return output

        def modify_input(input, layer_name):
            # print(layer_name)
            # for layer_id in layer_ids:
            #     if str(layer_id) in layer_name.split('.'):
            # heads_range = range(n_heads)
            input[:, :, 1:, token_id] = 0
            # sum_input = torch.unsqueeze(torch.sum(input, dim=-1), dim=-1)
            # sum_input[:,:,0,:] = 1
            # input = input / sum_input
            return input

        return modify_output, modify_input
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_and_cache = model(**inputs, output_hidden_states=True, output_attentions=True)
    ground_attentions = torch.cat(output_and_cache.attentions, dim=0).cpu()
    # attention_before_softmax = []
    # for (k,v) in output_and_cache.past_key_values:
    #     # attn_weights = torch.matmul(query, key.transpose(-1, -2))
    #     attention_before_softmax.append(torch.matmul(k, torch.permute(v,(0,1,3,2))))

    # the final answer token is:
    soft = torch.softmax(output_and_cache.logits[0, :, :], dim=1)
    result_pos = soft.max(dim=-1).indices.cpu().numpy()
    result_prob = soft.max(dim=-1).values.cpu().numpy()
    result_token = tokenizer.decode(result_pos.tolist())
    n_tokens = soft.shape[0]



    inputs_batch = tokenizer(['\n'+prompt] * 1, return_tensors="pt").to(device)
    # print(inputs_batch.input_ids[0], inputs.input_ids[0])
    # print(tokenizer.decode(inputs_batch.input_ids[0].tolist()), tokenizer.decode(inputs.input_ids[0].tolist()))
    token_id, layer_ids = 0, range(3, model_config['n_layers']-1)
    modify_output, modify_input = modify(layer_ids, token_id, model_config['n_heads'],  check_token_id , device)
    with TraceDict2(model, layers=model_config['layer_hook_names'], edit_input=modify_input,
                    edit_output=modify_output, retain_output=False) as ret:
        output_and_cache = model(**inputs_batch,output_attentions=True)
        change_attentions = torch.stack(output_and_cache.attentions, dim=0)[:,0].cpu()
        outputs_exp = output_and_cache.logits
        scores = torch.softmax(outputs_exp[:, :, :], dim=-1)[0]
        original_token_pos = result_pos[check_token_id]
        original_token_score = result_prob[check_token_id]
        new_result_original_token_score = scores[check_token_id, original_token_pos]
        new_result_token_pos = torch.argmax(scores[check_token_id])
        new_result_token_score = scores[check_token_id,new_result_token_pos]
        result = (tokenizer.decode([original_token_pos]),tokenizer.decode([new_result_token_pos]))
    return (ground_attentions.numpy(), change_attentions.numpy(),
            (original_token_pos, original_token_score, new_result_original_token_score,
             new_result_token_pos.item(), new_result_token_score.item(), result))

def try_hook4(model, tokenizer, model_config, prompt, check_token_id, device):

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
            return output

        def modify_input(input, layer_name):
            # print(layer_name)
            # for layer_id in layer_ids:
            #     if str(layer_id) in layer_name.split('.'):
            # heads_range = range(n_heads)
            input[:, :, 1:, token_id] = 0
            sum_input = torch.unsqueeze(torch.sum(input, dim=-1), dim=-1)
            sum_input[:,:,0,:] = 1
            input = input / sum_input
            return input

        return modify_output, modify_input
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_and_cache = model(**inputs, output_hidden_states=True, output_attentions=True)
    ground_attentions = torch.cat(output_and_cache.attentions, dim=0).cpu()
    # attention_before_softmax = []
    # for (k,v) in output_and_cache.past_key_values:
    #     # attn_weights = torch.matmul(query, key.transpose(-1, -2))
    #     attention_before_softmax.append(torch.matmul(k, torch.permute(v,(0,1,3,2))))

    # the final answer token is:
    soft = torch.softmax(output_and_cache.logits[0, :, :], dim=1)
    result_pos = soft.max(dim=-1).indices.cpu().numpy()
    result_prob = soft.max(dim=-1).values.cpu().numpy()
    result_token = tokenizer.decode(result_pos.tolist())
    n_tokens = soft.shape[0]



    inputs_batch = tokenizer(['\n'+prompt] * 1, return_tensors="pt").to(device)
    # print(inputs_batch.input_ids[0], inputs.input_ids[0])
    # print(tokenizer.decode(inputs_batch.input_ids[0].tolist()), tokenizer.decode(inputs.input_ids[0].tolist()))
    token_id, layer_ids = 0, range(3, model_config['n_layers']-1)
    modify_output, modify_input = modify(layer_ids, token_id, model_config['n_heads'],  check_token_id , device)
    with TraceDict2(model, layers=model_config['layer_hook_names'], edit_input=modify_input,
                    edit_output=modify_output, retain_output=False) as ret:
        output_and_cache = model(**inputs_batch,output_attentions=True)
        change_attentions = torch.stack(output_and_cache.attentions, dim=0)[:,0].cpu()
        outputs_exp = output_and_cache.logits
        scores = torch.softmax(outputs_exp[:, :, :], dim=-1)[0]
        original_token_pos = result_pos[check_token_id]
        original_token_score = result_prob[check_token_id]
        new_result_original_token_score = scores[check_token_id, original_token_pos]
        new_result_token_pos = torch.argmax(scores[check_token_id])
        new_result_token_score = scores[check_token_id,new_result_token_pos]
        result = (tokenizer.decode([original_token_pos]),tokenizer.decode([new_result_token_pos]))
    return (ground_attentions.numpy(), change_attentions.numpy(),
            (original_token_pos, original_token_score, new_result_original_token_score,
             new_result_token_pos.item(), new_result_token_score.item(), result))


def mask_head_with_api(model_name, prompt, check_token_id, device):

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device, True)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_and_cache = model(**inputs, output_hidden_states=True, output_attentions=True, use_cache=True)
    ground_attentions = output_and_cache.attentions
    # the final answer token is:
    soft = torch.softmax(output_and_cache.logits[0, :, :], dim=1)
    result_pos = soft.max(dim=-1).indices.cpu().numpy()
    result_prob = soft.max(dim=-1).values.cpu().numpy()
    result_token = tokenizer.decode(result_pos.tolist())

    # key, value = output_and_cache.past_key_values[0]
    # kv = list(output_and_cache.past_key_values)
    # kv[0] = (key*0.1, value*0.1)
    # atten = torch.matmul(key,value.permute(0,1,3,2))
    # output_and_cache2 = model(**inputs, past_key_values=tuple(kv), output_hidden_states=True, output_attentions=True)
    # print( atten == ground_attentions[0])

    heads_importance = np.zeros((model_config['n_layers'], model_config['n_heads'],inputs.input_ids.shape[-1]))
    for layer_id in range(model_config['n_layers']):
        for head_id in range(model_config['n_heads']):
            head_mask = np.ones((model_config['n_layers'], model_config['n_heads']))
            head_mask[layer_id, head_id] = 0
            output_and_cache = model(**inputs, use_cache=True,head_mask=torch.from_numpy(head_mask).to(device))
            s = torch.softmax(output_and_cache.logits[0, :, :], dim=1)
            prob = - (result_prob - np.array([s[i, p].item() for i, p in enumerate(result_pos.tolist())]))/result_prob
            heads_importance[layer_id, head_id] = prob
            # outputs = dict()
            # outputs['past_key_values'] = output_and_cache.past_key_values  # the "attentnion memory"
            # outputs['attentions'] = output_and_cache.attentions
            # outputs['hidden_states'] = torch.cat(output_and_cache.hidden_states, dim=0)[1:].cpu()
            # check = torch.cat(ground_attentions, dim=0) == torch.cat(output_and_cache.attentions, dim=0)
            # ground_heads = torch.cat(ground_attentions, dim=0)[:,:, -1,:].cpu()
            # test_ = torch.sum(ground_heads, dim=-1)
            # heads = torch.cat(output_and_cache.attentions, dim=0)[:,:, -1,:].cpu()
            # test = torch.sum(heads, dim=-1)
            # heads_show = torch.mean(heads, dim=1)
            # print(prob)
    return heads_importance[:, :, check_token_id].T

def plt_heatMap_sns(scores, save_path=None, title=None, cmap=None, y_ticks=None, x_ticks=None, show=None):
    plt.subplots(figsize=(20, 20), dpi=200)
    plt.rcParams['font.size'] = '10'
    if cmap is None:
        cmap = sns.color_palette("Reds", as_cmap=True)
    if x_ticks and y_ticks:
        sns.heatmap(scores, cmap=cmap,  xticklabels=x_ticks, yticklabels=y_ticks)
    else:
        sns.heatmap(scores, cmap=cmap)
    if title is not None:
        plt.title(title)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(os.path.join(save_path, f'{title}.png'), bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
def plt_heatMap(scores, kind=None, save_path=None, title=None):
    fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
    h = ax.pcolor(scores,cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds"}[kind])
    ax.invert_yaxis()
    ax.set_yticks([0.5 + i for i in range(len(scores))])
    ax.set_xticks([0.5 + i for i in range(0, scores.shape[1] - 6, 5)])
    ax.set_xticklabels(list(range(0, scores.shape[1] - 6, 5)))
    # ax.set_yticklabels(labels)
    ax.set_title(title)
    cb = plt.colorbar(h)
    if title is not None:
        ax.set_title(title)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(os.path.join(save_path, title), bbox_inches="tight")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    check_token_id = -1
    device = 'cuda:3'
    model_name = 'EleutherAI/gpt-j-6b' #   # 'EleutherAI/gpt-j-6b' 'meta-llama/Llama-2-7b'
    model_config = AutoConfig.from_pretrained(model_name)
    n_layers = model_config.n_layer
    n_heads = model_config.n_head

    prompt = 'Beats Music is owned by' # 'Beats Music is owned by', 'The Space Needle is in downtown'
    target_token = 'Apple'
    save_path = './result/atten_importance/'
    x_ticks = [f"layer{i + 1}" for i in range(n_layers)]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # CounterFactDataset验证效果
    # dataset = CounterFactDataset('./data/')
    # model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device, True)
    # n,greater_num,less_num = 0,0,0
    # for data in dataset:
    #     for p in data['attribute_prompts']: #'neighborhood_prompts']:
    #         # 去掉attention sink的影响
    #         ground_attentions, change_attentions,(original_token_pos, original_token_score, new_result_original_token_score,new_result_token_pos, new_result_token_score, result) \
    #             = try_hook3(model, tokenizer, model_config, p, check_token_id, device)
    #         ground_attentions, change_attentions = ground_attentions[:, :, check_token_id, :], change_attentions[:,
    #                                                                                            :, check_token_id,
    #                                                                                            1:]
    #         encoded_line = tokenizer.encode(p)
    #         codes = tokenizer.convert_ids_to_tokens(encoded_line)
    #         y_ticks = [f"head{i_head}-{c}" for i_head in range(n_heads) for i, c in enumerate(codes)]
    #         plt_heatMap_sns(ground_attentions.reshape(ground_attentions.shape[0], -1).T,
    #                         title="ground_attentions", x_ticks=x_ticks, y_ticks=y_ticks
    #                         , show=True, save_path=save_path)
    #         if result[0] != result[1]:
    #             print(f"=====prompt:{p}, (original, changed):{result}, original score:{original_token_score}, changed score:{new_result_token_score}")
    #         else:
    #             if new_result_token_score > original_token_score: greater_num = greater_num+1
    #             else : less_num +=1
    #         n += 1
    # print(f"score is higher: {greater_num}/{n}, score is lower: {less_num}/{n}")


    encoded_line = tokenizer.encode(prompt)
    codes = tokenizer.convert_ids_to_tokens(encoded_line)

    # 查看原始attention 和去除attention sink之后的效果
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device, True)
    ground_attentions, change_attentions, (
    original_token_pos, original_token_score, new_result_original_token_score, new_result_token_pos,
    new_result_token_score, result) \
        = try_hook4(model, tokenizer, model_config, prompt, check_token_id, device)
    ground_attentions,change_attentions = ground_attentions[:, :, check_token_id, :],change_attentions[:, :, check_token_id, 1:]
    y_ticks = [f"head{i_head}-{c}" for i_head in range(n_heads) for i, c in enumerate(codes)]
    plt_heatMap_sns(ground_attentions.reshape(ground_attentions.shape[0], -1).T,
                    title="ground_attentions", x_ticks=x_ticks, y_ticks=y_ticks
                    , show=True, save_path=save_path)
    zero_attention = np.zeros_like(ground_attentions)
    zero_attention[4:,:, 1:] = ground_attentions[4:,:, 1:]
    plt_heatMap_sns(zero_attention.reshape(zero_attention.shape[0], -1).T,
                    title="ground_attentions_segment", x_ticks=x_ticks, y_ticks=y_ticks
                    , show=True, save_path=save_path)
    plt_heatMap_sns(change_attentions.reshape(change_attentions.shape[0], -1).T,
                    title="change_attentions", x_ticks=x_ticks, y_ticks=y_ticks
                    , show=True, save_path=save_path)
    zero_attention = np.zeros_like(change_attentions)
    zero_attention[4:] = change_attentions[4:]
    plt_heatMap_sns(zero_attention.reshape(zero_attention.shape[0], -1).T,
                    title="change_attentions_segment", x_ticks=x_ticks, y_ticks=y_ticks
                    , show=True, save_path=save_path)

    # 获取softmax之后的attention 以及 每个attention的影响程度
    ground_attentions, atten_importance = try_hook2(model_name, prompt, check_token_id, device)
    # print(ground_attentions, atten_importance)
    plt_heatMap_sns(ground_attentions.reshape(ground_attentions.shape[0], -1).T,
                    title="ground_attentions", x_ticks=x_ticks, y_ticks=y_ticks
                    ,show=True, save_path=save_path)
    ground_attentions2 = np.where(ground_attentions>=(1/len(codes)), 1, 0)
    plt_heatMap_sns(ground_attentions2.reshape(ground_attentions2.shape[0], -1).T,
                    title="ground_attentions2", x_ticks=x_ticks, y_ticks=y_ticks
                    , show=True, save_path=save_path)
    plt_heatMap_sns(atten_importance.reshape(atten_importance.shape[0], -1).T,
                    title="atten_importance",save_path=save_path,
                    cmap=sns.color_palette("coolwarm", as_cmap=True),
                    x_ticks=x_ticks, y_ticks=y_ticks
                    ,show=True)

    # 直接使用api， 检查每个head对于整句话的影响
    heads_importance = mask_head_with_api(model_name, prompt, -1, device)
    plt_heatMap_sns(heads_importance, title="heads_importance", save_path=save_path,
                    cmap=sns.color_palette("coolwarm", as_cmap=True),
                    x_ticks=x_ticks, y_ticks=[f'head{i}' for i in range(n_heads)],
                    show=True)