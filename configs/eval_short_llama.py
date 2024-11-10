from opencompass.models import HuggingFaceCausalLM
from opencompass.datasets import BoolQDataset, boolq_postprocess

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

datasets = [
    dict(
        abbr='boolq',
        type=BoolQDataset,
        path='./data/boolq/dev.jsonl',
        reader_cfg=dict(
            input_columns=['passage', 'question'],
            output_column='label',
        ),
        infer_cfg=dict(
            prompt_template='{passage}\nQuestion: {question}\nAnswer:',
            retriever=dict(type='ZeroRetriever'),
        ),
        eval_cfg=dict(
            evaluator=dict(type='AccEvaluator'),
            pred_postprocessor=dict(type='BoolQPostProcessor'),
        ),
    )
]

summarizer = dict(
    type='GroupedBarSummarizer',
    text_cfg=dict(
        task_abbr='boolq',
        metric='Accuracy',
    ),
)