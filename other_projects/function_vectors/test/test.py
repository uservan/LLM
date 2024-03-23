import os, re, json
import torch, numpy as np

import sys
sys.path.append('..')
torch.set_grad_enabled(False)

from src.utils.extract_utils import get_mean_head_activations, compute_universal_function_vector
from src.utils.intervention_utils import fv_intervention_natural_text, function_vector_intervention
from src.utils.model_utils import load_gpt_model_and_tokenizer
from src.utils.prompt_utils import load_dataset, word_pairs_to_prompt_data, create_prompt
from src.utils.eval_utils import decode_to_vocab, sentence_eval

model_name = 'EleutherAI/gpt-j-6b'
model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name)
EDIT_LAYER = 9

dataset = load_dataset('antonym', seed=0)
mean_activations = get_mean_head_activations(dataset, model, model_config, tokenizer)

FV, top_heads = compute_universal_function_vector(mean_activations, model, model_config, n_top_heads=10)


# Sample ICL example pairs, and a test word
dataset = load_dataset('antonym')
word_pairs = dataset['train'][:5]
test_pair = dataset['test'][21]

prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=test_pair, prepend_bos_token=True)
sentence = create_prompt(prompt_data)
print("ICL prompt:\n", repr(sentence), '\n\n')

shuffled_prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=test_pair, prepend_bos_token=True, shuffle_labels=True)
shuffled_sentence = create_prompt(shuffled_prompt_data)
print("Shuffled ICL Prompt:\n", repr(shuffled_sentence), '\n\n')

zeroshot_prompt_data = word_pairs_to_prompt_data({'input':[], 'output':[]}, query_target_pair=test_pair, prepend_bos_token=True, shuffle_labels=True)
zeroshot_sentence = create_prompt(zeroshot_prompt_data)
print("Zero-Shot Prompt:\n", repr(zeroshot_sentence))


# Intervention on the zero-shot prompt
clean_logits, interv_logits = function_vector_intervention(zeroshot_sentence, [test_pair['output']], EDIT_LAYER, FV, model, model_config, tokenizer)

print("Input Sentence:", repr(zeroshot_sentence), '\n')
print(f"Input Query: {repr(test_pair['input'])}, Target: {repr(test_pair['output'])}\n")
print("Zero-Shot Top K Vocab Probs:\n", decode_to_vocab(clean_logits, tokenizer, k=5), '\n')
print("Zero-Shot+FV Vocab Top K Vocab Probs:\n", decode_to_vocab(interv_logits, tokenizer, k=5))

