from opencompass.models import HuggingFaceCausalLM

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

datasets = ['boolq']

from opencompass.summarizers import GroupedBarSummarizer
summarizer = dict(type=GroupedBarSummarizer)