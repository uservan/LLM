# general imports
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import utils

device = torch.device('cpu') # if torch.cuda.is_available() else torch.device('cpu')

# example 1
model_name = 'gpt2-medium' #
line = 'When Mary and John went to the store, John gave a drink to'
target_token = ' Mary' # notice the token includes the space before it


model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.requires_grad_(False)
model_config = model.config

# collect the hidden states before and after each of those layers (modules)
hs_collector = utils.wrap_model(model, layers_to_check = [
    '.mlp', '.mlp.c_fc', '.mlp.c_proj',
    '.attn', '.attn.c_attn', '.attn.c_proj',
    '.ln_1', '.ln_2', '',])  # '' stands for wrapping transformer.h.<layer_index> in gpt2

# add extra functions to the model (like logit lens adjust to the model decoding matrix)
model_aux = utils.model_extra(model=model, device='cpu')

tokenizer = AutoTokenizer.from_pretrained(model_name)
try:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"  # not blocking, just to prevent warnings and faster tokenization
except:
    pass
encoded_line = tokenizer.encode(line.rstrip(), return_tensors='pt').to(device)
output_and_cache = model(encoded_line, output_hidden_states=True, output_attentions=True, use_cache=True)

hs_collector['past_key_values'] = output_and_cache.past_key_values  # the "attentnion memory"
hs_collector['attentions'] = output_and_cache.attentions

# the final answer token is:
model_answer = tokenizer.decode(output_and_cache.logits[0, -1, :].argmax().item())
print(f'model_answer: "{model_answer}"')
