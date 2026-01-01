# -*- coding: utf-8 -*-
"""
Llama3.1-8B GRPO 强化推理脚本（中英文注释）

改编自 Unsloth 的 Llama3_(8B)-GRPO Notebook，去掉所有 Colab/HTML/广告内容，
只保留训练思维链 + GRPO 的核心逻辑。整体步骤：
1. 加载 Llama3.1-8B 基座 + LoRA。
2. 配置思维链模板与 <start_working_out>/<SOLUTION> 标签。
3. 使用小样本预格式化微调，帮助模型先学会这种输出形式。
4. 准备 Open-R1 DAPO Math 数据集以及奖励函数。
5. 运行 GRPO 强化训练，并演示推理、LoRA 保存及可选导出。
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
# 1. 模型配置
#############################################

MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"
MAX_SEQ_LENGTH = 2048
LORA_RANK = 32
RANDOM_SEED = 3407

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,            # 16bit + LoRA
    fast_inference=True,           # 启用 vLLM 快速推理
    max_lora_rank=LORA_RANK,
    gpu_memory_utilization=0.9,
)

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
# 2. 思维链模板
#############################################

reasoning_start = "<start_working_out>"
reasoning_end = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

system_prompt = (
    "You are given a math problem. Work through it carefully.\n"
    f"Show your reasoning between {reasoning_start} and {reasoning_end}.\n"
    f"Then output the final answer between {solution_start}{solution_end}."
)

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
# 3. 预格式化微调
#############################################

def load_prefinetune_dataset() -> Dataset:
    df = load_dataset("unsloth/OpenMathReasoning-mini", split="cot").to_pandas()
    df = df[["expected_answer", "problem", "generated_solution"]]

    mask = pd.to_numeric(df["expected_answer"], errors="coerce").notnull()
    df = df.loc[mask].copy()

    def format_row(row):
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

    df["Messages"] = df.apply(format_row, axis=1)
    df["token_count"] = df["Messages"].apply(lambda x: len(tokenizer.apply_chat_template(x)))
    df = df.loc[df["token_count"] <= MAX_SEQ_LENGTH / 2]
    df["text"] = tokenizer.apply_chat_template(df["Messages"].tolist(), tokenize=False)
    return Dataset.from_pandas(df)


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
    del dataset
    torch.cuda.empty_cache()
    gc.collect()


#############################################
# 4. 主训练数据集
#############################################

def load_main_dataset() -> Dataset:
    hf_ds = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")

    def convert_item(sample):
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": sample["prompt"]},
            ],
            "answer": sample["solution"],
        }

    return hf_ds.map(convert_item)


#############################################
# 5. 奖励函数
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
    return [3.0 if match_format.search(c[0]["content"]) else 0.0 for c in completions]


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
    responses = [c[0]["content"] for c in completions]
    extracted = [
        (match := match_format.search(r)).group(1) if match else None
        for r in responses
    ]
    scores = []
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
# 6. 过滤过长 prompt
#############################################

def filter_by_prompt_length(dataset: Dataset) -> tuple[Dataset, int]:
    tokenized = dataset.map(
        lambda x: {
            "tokens": tokenizer.apply_chat_template(
                x["prompt"], add_generation_prompt=True, tokenize=True
            )
        },
        batched=True,
    )
    tokenized = tokenized.map(lambda x: {"token_len": len(x["tokens"])})
    q90 = int(np.quantile(tokenized["token_len"], 0.9))
    keep = np.where(np.array(tokenized["token_len"]) <= q90)[0]
    dataset = dataset.select(keep.tolist())
    return dataset, q90


#############################################
# 7. GRPO 训练
#############################################

def run_grpo_training():
    run_prefinetune()

    dataset = load_main_dataset()
    dataset, q90_len = filter_by_prompt_length(dataset)

    max_prompt_length = q90_len + 1
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
# 8. 推理与 LoRA 保存
#############################################

def test_inference(prompt: str, lora_path: str | None = None):
    sampling = SamplingParams(temperature=1.0, top_k=50, max_tokens=1024)
    request = model.load_lora(lora_path) if lora_path else None
    outputs = model.fast_generate(
        [prompt],
        sampling_params=sampling,
        lora_request=request,
    )[0].outputs[0].text
    print("Model output:\n", outputs)


def save_and_check_lora(save_dir: str = "llama_grpo_lora"):
    model.save_lora(save_dir)
    with safe_open(os.path.join(save_dir, "adapter_model.safetensors"), framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            assert (tensor == 0).sum().item() != tensor.numel(), "LoRA tensor appears all zeros."
    return save_dir


#############################################
# 9. 可选导出
#############################################

def optional_exports():
    if False:
        model.save_pretrained_merged("llama_grpo_fp16", tokenizer, save_method="merged_16bit")
    if False:
        model.save_pretrained_merged("llama_grpo_int4", tokenizer, save_method="merged_4bit")
    if False:
        model.save_pretrained("llama_lora_only")
        tokenizer.save_pretrained("llama_lora_only")
    if False:
        model.save_pretrained_gguf("llama_grpo_gguf", tokenizer, quantization_method="q4_k_m")


#############################################
# 10. 入口
#############################################

if __name__ == "__main__":
    run_grpo_training()

    question = "What is the square root of 59?"
    test_inference(question, lora_path=None)

    lora_dir = save_and_check_lora()
    test_inference(question, lora_path=lora_dir)

    optional_exports()

