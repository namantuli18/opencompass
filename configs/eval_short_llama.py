from opencompass.models import HuggingFaceCausalLM
from opencompass.datasets import BoolQDataset, boolq_postprocess
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.collections.base_medium_llama import piqa_datasets, siqa_datasets
    from opencompass.configs.models.llama.llama2_7b import models


datasets = [*piqa_datasets, *siqa_datasets]
models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='short-llama',
        path='namannn/short-llama',
        tokenizer_path='namannn/short-llama',
        model_kwargs=dict(device_map='auto'),
        max_seq_len=2048,
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