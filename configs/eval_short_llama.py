from opencompass.models import HuggingFaceCausalLM
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.demo.demo_gsm8k_chat_gen import gsm8k_datasets
    from opencompass.configs.datasets.demo.demo_math_chat_gen import math_datasets

datasets = gsm8k_datasets
models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='short-llama',
        path='namannn/short-llama',
        tokenizer_path='namannn/short-llama',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
            load_in_8bit=False,  # Set to True if using 8-bit quantization
            revision='main',  # Specify the correct revision/branch
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