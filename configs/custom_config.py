# custom_config.py
from transformers import LlamaConfig

def get_custom_config():
    custom_config = LlamaConfig.from_pretrained("namannn/short-llama2-7b")
    custom_config.num_hidden_layers = 23  # Adjust this to match your pruned model
    return custom_config