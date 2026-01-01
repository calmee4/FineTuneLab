# -*- coding: utf-8 -*-
"""
Qwen3-4B GRPO 思维链强化脚本（中英文注释）

本脚本基于 Unsloth 官方 Qwen3_(4B)-GRPO Notebook，去掉了 Colab/营销相关内容，
只保留训练强化推理模型的必需步骤。整体流程：
1. 加载基座 + LoRA；配置自定义思维链模板。
2. 通过小规模 “预格式化微调” 让模型学会 <start_working_out>/<SOLUTION> 的输出形式。
3. 准备 Open-R1 DAPO-Math 数据集，编写用于格式/答案的奖励函数。
4. 使用 GRPOTrainer 进行强化学习训练。
5. 演示推理、保存 LoRA、可选导出到 16bit / GGUF。
"""

import gc
import os
import re
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from safetensors import safe_open
from transformers import TextStreamer
from vllm import SamplingParams
from trl import (
    SFTTrainer,
    SFTConfig,
    GRPOTrainer,
    GRPOConfig,
)

from unsloth import FastLanguageModel

#############################################
# 1. 基本配置
#############################################

# 最大序列长度，可根据显存扩展（长思维链建议 4096+）
# Max sequence length; increase if you need longer reasoning traces.
MAX_SEQ_LENGTH = 2048

# LoRA 秩；越大代表训练参数越多，效果更好但显存/算力要求更高
# LoRA rank; higher rank -> more trainable params -> better but heavier.
LORA_RANK = 32

# 随机种子，保证复现
# Random seed for reproducibility.
RANDOM_SEED = 3407

# 载入 Qwen3-4B 基座
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Base",
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,       # LoRA + 16bit 训练
    fast_inference=True,      # 允许 vLLM 加速推理
    max_lora_rank=LORA_RANK,
    gpu_memory_utilization=0.9,
)

# 注入 LoRA 模块
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_RANK * 2,
    use_gradient_checkpointing="unsloth",
    random_state=RANDOM_SEED,
)

#############################################
# 2. 自定义推理模板
#############################################

# 推理模板实际上就是一句话？
reasoning_start = "<start_working_out>"  # 类似 DeepSeek 的 <think>

# 也就是 这些推理模型的think 也全是一些token嘛
# 有think的开始和think的结束，等think结束就开始输出答案

reasoning_end = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

system_prompt = (
    "You are given a problem.\n"
    "Think about the problem and provide your working out.\n"
    f"Place it between {reasoning_start} and {reasoning_end}.\n"
    f"Then, provide your solution between {solution_start}{solution_end}"
)

# 自定义 chat template（Jinja2 语法）
chat_template = (
    "{% if messages[0]['role'] == 'system' %}"
    "{{ messages[0]['content'] + eos_token }}{% set loop_messages = messages[1:] %}"
    "{% else %}{{ '" + system_prompt + "' + eos_token }}{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}{{ message['content'] }}"
    "{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '" + reasoning_start + "' }}{% endif %}"
)
tokenizer.chat_template = chat_template

#############################################
# 3. 预格式化微调（帮助模型学会输出模板）
#############################################

def load_prefinetune_dataset() -> Dataset:
    """加载 OpenMathReasoning-mini 并转换成 chat 格式"""
    raw_df = load_dataset("unsloth/OpenMathReasoning-mini", split="cot").to_pandas()
    raw_df = raw_df[["expected_answer", "problem", "generated_solution"]]

    # 仅保留答案可转为数字的样本
    mask = pd.to_numeric(raw_df["expected_answer"], errors="coerce").notnull()
    raw_df = raw_df.loc[mask].copy()

    def format_row(row):
        # 去除 <think> 标签
        thoughts = row["generated_solution"].replace("<think>", "").replace("</think>", "").strip()
        final_answer = (
            reasoning_start + thoughts + reasoning_end +
            solution_start + row["expected_answer"] + solution_end
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["problem"]},
            {"role": "assistant", "content": final_answer},
        ]

    raw_df["Messages"] = raw_df.apply(format_row, axis=1)
    raw_df["token_count"] = raw_df["Messages"].apply(
        lambda x: len(tokenizer.apply_chat_template(x))
    )
    # 限制长度，避免太长的预训练样本
    raw_df = raw_df.loc[raw_df["token_count"] <= MAX_SEQ_LENGTH / 2]
    raw_df["text"] = tokenizer.apply_chat_template(
        raw_df["Messages"].tolist(), tokenize=False
    )
    # 不会保留长度过长的样本
    return Dataset.from_pandas(raw_df)


def run_prefinetune():
    dataset = load_prefinetune_dataset()
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=5,
            num_train_epochs=2,
            learning_rate=2e-4,
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=RANDOM_SEED,
            report_to="none",
        ),
    )
    trainer.train()

    # 清理显存
    del dataset
    torch.cuda.empty_cache()
    gc.collect()


#############################################
# 4. 准备 GRPO 用主数据集
#############################################

def load_main_dataset() -> Dataset:
    hf_ds = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")

    def convert_sample(sample):
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": sample["prompt"]},
            ],
            "answer": sample["solution"],
        }

    return hf_ds.map(convert_sample)


#############################################
# 5. 奖励函数集合
#############################################

solution_end_regex = (
    r"</SOLUTION>\s{0,}" + "(?:" + re.escape(tokenizer.eos_token) + ")?"
)
match_format = re.compile(
    rf"{reasoning_end}.*?{solution_start}(.+?){solution_end_regex}\s{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)
match_numbers = re.compile(
    solution_start + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
    flags=re.MULTILINE | re.DOTALL,
)


def reward_format_exact(completions: List[List[Dict[str, str]]], **_) -> List[float]:
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        scores.append(3.0 if match_format.search(response) else 0.0)
    return scores


def reward_format_partial(completions: List[List[Dict[str, str]]], **_) -> List[float]:
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        score = 0.0
        score += 0.5 if response.count(reasoning_end) == 1 else -1.0
        score += 0.5 if response.count(solution_start) == 1 else -1.0
        score += 0.5 if response.count(solution_end) == 1 else -1.0
        scores.append(score)
    return scores


def reward_answer_exact(prompts, completions, answer, **_) -> List[float]:
    scores = []
    responses = [c[0]["content"] for c in completions]
    extracted = [
        (match := match_format.search(r)).group(1) if match else None
        for r in responses
    ]
    for guess, truth in zip(extracted, answer):
        if guess is None:
            scores.append(-2.0)
            continue
        guess_stripped = guess.strip()
        truth_stripped = truth.strip()
        if guess == truth:
            scores.append(5.0)
        elif guess_stripped == truth_stripped:
            scores.append(3.5)
        else:
            try:
                ratio = float(guess_stripped) / float(truth_stripped)
                if 0.9 <= ratio <= 1.1:
                    scores.append(2.0)
                elif 0.8 <= ratio <= 1.2:
                    scores.append(1.5)
                else:
                    scores.append(-2.5)
            except Exception:
                scores.append(-4.5)
    return scores


PRINT_COUNTER = 0
PRINT_EVERY = 5


def reward_answer_numeric(prompts, completions, answer, **_) -> List[float]:
    global PRINT_COUNTER
    responses = [c[0]["content"] for c in completions]
    extracted = [
        (match := match_numbers.search(r)).group(1) if match else None
        for r in responses
    ]

    if PRINT_COUNTER % PRINT_EVERY == 0:
        print(
            "*" * 20,
            f"\nQuestion:\n{prompts[0][-1]['content']}",
            f"\nAnswer:\n{answer[0]}",
            f"\nResponse:\n{responses[0]}",
            f"\nExtracted:\n{extracted[0]}",
        )
    PRINT_COUNTER += 1

    scores = []
    for guess, truth in zip(extracted, answer):
        if guess is None:
            scores.append(-2.5)
            continue
        try:
            truth_num = float(truth.strip())
            guess_num = float(guess.strip().replace(",", ""))
            scores.append(3.5 if guess_num == truth_num else -1.5)
        except Exception:
            scores.append(0.0)
    return scores


#############################################
# 6. 过滤过长 prompt，避免截断
#############################################

def filter_by_prompt_length(dataset: Dataset) -> Dataset:
    tokenized = dataset.map(
        lambda x: {
            "tokens": tokenizer.apply_chat_template(
                x["prompt"], add_generation_prompt=True, tokenize=True
            )
        },
        batched=True,
    )
    tokenized = tokenized.map(lambda x: {"token_len": len(x["tokens"])})
    max_len = int(np.quantile(tokenized["token_len"], 0.9))
    print("90% prompt length:", max_len)

    keep_indices = np.where(np.array(tokenized["token_len"]) <= max_len)[0]
    dataset = dataset.select(keep_indices.tolist())
    return dataset, max_len


#############################################
# 7. GRPO 配置与训练
#############################################

def run_grpo_training():
    run_prefinetune()

    dataset = load_main_dataset()
    dataset, prompt_90_len = filter_by_prompt_length(dataset)

    max_prompt_length = prompt_90_len + 1
    max_completion_length = MAX_SEQ_LENGTH - max_prompt_length

    sampling_params = SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=RANDOM_SEED,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )

    grpo_args = GRPOConfig(
        vllm_sampling_params=sampling_params,
        temperature=1.0,
        learning_rate=5e-6,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=4,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=100,
        save_steps=100,
        report_to="none",
        output_dir="outputs",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_format_exact,
            reward_format_partial,
            reward_answer_exact,
            reward_answer_numeric,
        ],
        args=grpo_args,
        train_dataset=dataset,
    )
    trainer.train()


#############################################
# 8. 推理示例与 LoRA 保存
#############################################

def test_inference(prompt: str, lora_path: str | None = None):
    """使用 fast_generate 进行测试，若提供 LoRA 路径则加载"""
    sampling = SamplingParams(temperature=1.0, top_k=50, max_tokens=1024)
    request = model.load_lora(lora_path) if lora_path else None
    outputs = model.fast_generate(
        [prompt],
        sampling_params=sampling,
        lora_request=request,
    )[0].outputs[0].text
    print("Model output:\n", outputs)


def save_and_verify_lora(save_dir: str = "grpo_saved_lora"):
    """保存 LoRA，并简单检查参数是否已更新"""
    model.save_lora(save_dir)
    with safe_open(os.path.join(save_dir, "adapter_model.safetensors"), framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            assert (tensor == 0).sum().item() != tensor.numel(), "LoRA tensor 全为 0，疑似未训练"
    return save_dir


#############################################
# 9. 可选：合并/导出模型
#############################################

def optional_exports():
    # 这些 API 需要时再手动打开，避免误保存到巨大文件。
    # Uncomment the lines you need.
    if False:
        model.save_pretrained_merged("model_fp16", tokenizer, save_method="merged_16bit")
    if False:
        model.save_pretrained_merged("model_int4", tokenizer, save_method="merged_4bit")
    if False:
        model.save_pretrained("lora_only")
        tokenizer.save_pretrained("lora_only")
    if False:
        model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")


#############################################
# 10. 主入口
#############################################

if __name__ == "__main__":
    # 运行 GRPO 训练流程；若只想体验推理，可以注释掉
    run_grpo_training()

    # 训练前后推理对比
    question = "What is the sqrt of 101?"
    test_inference(question, lora_path=None)  # 未加载 LoRA

    lora_dir = save_and_verify_lora()
    test_inference(question, lora_path=lora_dir)  # 加载训练好的 LoRA

    optional_exports()

