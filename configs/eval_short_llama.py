from opencompass.models import HuggingFaceCausalLM
from mmengine.config import read_base
from custom_config import get_custom_config

with read_base():
    from opencompass.configs.datasets.demo.demo_gsm8k_chat_gen import gsm8k_datasets

datasets = gsm8k_datasets

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='short-llama',
        path='meta-llama/Llama-2-7b-hf',
        tokenizer_path='meta-llama/Llama-2-7b-hf',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
            load_in_8bit=False,
            revision='main',
            # config=dict(
            #     type='custom_config.get_custom_config',
            # ),
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