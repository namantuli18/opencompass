from opencompass.models import HuggingFaceCausalLM
from mmengine.config import read_base
from transformers import LlamaConfig

with read_base():
    from opencompass.configs.datasets.demo.demo_gsm8k_chat_gen import gsm8k_datasets
    from opencompass.configs.datasets.demo.demo_math_chat_gen import math_datasets

datasets = gsm8k_datasets

def get_custom_config():
    custom_config = LlamaConfig.from_pretrained("namannn/short-llama2-7b")
    custom_config.num_hidden_layers = 23  # Adjust this to match your pruned model
    return custom_config

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='short-llama',
        path='namannn/short-llama2-7b',
        tokenizer_path='namannn/short-llama2-7b',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
            load_in_8bit=False,  # Set to True if using 8-bit quantization
            revision='main',  # Specify the correct revision/branch
            config=get_custom_config(),  # Add this line to use the custom config
        ),
        max_seq_len=1024,
        batch_size=4,
        run_cfg=dict(num_gpus=1),
    )
]

summarizer = dict(
    type='GroupedBarSummarizer',
    text_cfg=dict(
        task_abbr='boolq',
        metric='Accuracy',
    ),
)