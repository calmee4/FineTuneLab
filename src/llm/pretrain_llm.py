# -*- coding: utf-8 -*-
"""训练语言模型脚本，翻译自 Hugging Face 官方 notebook。"""

# 可选：登录 Hugging Face，便于推送模型或访问受限资源。
from huggingface_hub import notebook_login

notebook_login()

# 确认安装的 transformers 版本，确保功能兼容。
import transformers

print(transformers.__version__)

# ======================== 数据准备 ========================

from datasets import load_dataset, ClassLabel
import random
import pandas as pd
from IPython.display import display, HTML

# 加载 Wikitext-2 数据集，可根据需要替换为自定义语料。
datasets = load_dataset("wikitext", "wikitext-2-raw-v1")


def show_random_elements(dataset, num_examples=10):
    """随机展示样本，便于快速了解语料结构。"""
    assert num_examples <= len(dataset), "请求的样本数量超过数据集大小"
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))


# 需要时可调用函数预览训练语料。
show_random_elements(datasets["train"])

# ======================== 因果语言模型 (CLM) ========================

model_checkpoint = "gpt2"
tokenizer_checkpoint = "sgugger/gpt2-like-tokenizer"

from transformers import AutoTokenizer

# 与模型保持一致的分词器，确保词表完全匹配。
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)


def tokenize_function(examples):
    """将文本转换为模型输入 ID。"""
    return tokenizer(examples["text"])


# 批量分词，加速处理并丢弃原始文本列。
tokenized_datasets = datasets.map(
    tokenize_function, batched=True, num_proc=4, remove_columns=["text"]
)


# 为了训练时的高效迭代，将 Token 拼接为固定窗口。
block_size = 128


def group_texts(examples):
    """把分词结果拼接并按 block_size 切片。"""
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

# 构建配置与模型，保持与 checkpoint 相同的超参设置。
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained(model_checkpoint)
model = AutoModelForCausalLM.from_config(config)

# 训练参数可按需调整，这里仅提供一个基础示例。
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    f"{model_checkpoint}-wikitext2",
    eval_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
)

# 开始训练并评估困惑度。
trainer.train()

import math

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

# 可选：将模型推送到 Hugging Face Hub。
trainer.push_to_hub()

# ======================== 掩码语言模型 (MLM) ========================

model_checkpoint = "bert-base-cased"
tokenizer_checkpoint = "sgugger/bert-like-tokenizer"

# 复用分词逻辑，仅替换检查点。
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
tokenized_datasets = datasets.map(
    tokenize_function, batched=True, num_proc=4, remove_columns=["text"]
)

# 同样把文本拼接成固定长度。
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

from transformers import AutoModelForMaskedLM

config = AutoConfig.from_pretrained(model_checkpoint)
model = AutoModelForMaskedLM.from_config(config)

training_args = TrainingArguments(
    "test-clm",
    eval_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=True,
    push_to_hub_model_id=f"{model_checkpoint}-wikitext2",
)

# 使用随机掩码的数据整理器，让每轮训练的掩码位置都不相同。
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm_probability=0.15
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    data_collator=data_collator,
)

trainer.train()

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

trainer.push_to_hub()
