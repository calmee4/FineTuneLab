# -*- coding: utf-8 -*-
"""
Gemma3-4B Vision GSPO（实为 GRPOTrainer）训练脚本，中文精简版。

源自 Unsloth 官方 Notebook，去掉 Colab/安装/营销段落，只保留本地训练必要步骤：
1. 加载 gemma-3-4b-it 4bit 视觉模型；
2. 注入 LoRA，只微调语言侧（attention + MLP）；
3. 下载 MathVista 子集，筛出数值答案，统一 resize/转 RGB；
4. 构造包含 <REASONING>/<SOLUTION> 的对话模板；
5. 定义两个规则奖励（格式 + 数值正确）；
6. 配置 GRPOTrainer（importance_sampling_level="sequence"）并训练；
7. 训练前后推理 & 保存 LoRA。
"""

import re
from typing import List

import torch
from datasets import load_dataset
from transformers import TextStreamer
from trl import GRPOConfig, GRPOTrainer

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

###############################################################################
# 1. 全局配置
###############################################################################

MODEL_NAME = "unsloth/gemma-3-4b-it"
DATASET_NAME = "AI4Math/MathVista"
DATA_SPLIT = "testmini"
SEED = 3407

REASONING_START = "<REASONING>"
REASONING_END = "</REASONING>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"

MAX_STEPS = 60
LEARNING_RATE = 5e-6
PER_DEVICE_BATCH = 1
GRAD_ACCUM_STEPS = 2
NUM_GENERATIONS = 4
MAX_PROMPT_LEN = 1024
MAX_COMPLETION_LEN = 1024


###############################################################################
# 2. 模型 & LoRA
###############################################################################

def load_model_and_tokenizer():
    """加载 4bit 视觉模型并启用 gradient checkpointing。"""
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    return model, tokenizer


def attach_lora(model):
    """
    为 Gemma Vision 注入 LoRA。
    官方示例只微调语言侧（视觉不打开），可按需调整。
    """
    # 先给视觉模块 注入LoRA
    return FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=SEED,
        use_rslora=False,
        loftq_config=None,
        use_gradient_checkpointing="unsloth",
    )


###############################################################################
# 3. 数据处理
###############################################################################

# 判断结果是否正确
def is_numeric_answer(example):
    try:
        float(example["answer"])
        return True
    except Exception:
        return False


def resize_and_rgb(example):
    """将图像 resize 到 512x512 并保证 RGB。"""
    image = example["decoded_image"].resize((512, 512))
    if image.mode != "RGB":
        image = image.convert("RGB")
    example["decoded_image"] = image
    return example


def build_prompt(example):
    """
    构建多模态对话：用户上传图像 + 问题，并要求在指定标签输出推理与答案。
    """
    text_prompt = (
        f"{example['question']}，请将推理写在 {REASONING_START}/{REASONING_END} 之间，"
        f"答案写在 {SOLUTION_START}/{SOLUTION_END} 之间且为浮点数。"
    )
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]
    return {
        "prompt": prompt,
        "image": example["decoded_image"],
        "answer": example["answer"],
    }


def prepare_dataset(tokenizer):
    """加载、筛选、格式化 MathVista 数据集，返回可直接用于训练的 Dataset。"""
    dataset = load_dataset(DATASET_NAME, split=DATA_SPLIT)
    dataset = dataset.filter(is_numeric_answer)
    dataset = dataset.map(resize_and_rgb)
    dataset = dataset.map(build_prompt)
    dataset = dataset.remove_columns("image")
    dataset = dataset.rename_column("decoded_image", "image")

    dataset = dataset.map(
        lambda example: {
            "prompt": tokenizer.apply_chat_template(
                example["prompt"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }
    )
    return dataset


###############################################################################
# 4. 奖励函数（规则打分）
###############################################################################

def formatting_reward_func(completions: List[str], **_) -> List[float]:
    """检测思维链/答案标签是否各出现一次，每满足一个加 1 分。"""
    think_pat = f"{REASONING_START}(.*?){REASONING_END}"
    ans_pat = f"{SOLUTION_START}(.*?){SOLUTION_END}"
    scores = []
    for text in completions:
        score = 0.0
        if len(re.findall(think_pat, text, re.DOTALL)) == 1:
            score += 1.0
        if len(re.findall(ans_pat, text, re.DOTALL)) == 1:
            score += 1.0
        scores.append(score)
    return scores


def correctness_reward_func(prompts, completions, answer, **_) -> List[float]:
    """
    抓取 <SOLUTION>…</SOLUTION> 的内容，完全等于真值则给 2 分；
    否则 0 分。每个 batch 打印一次便于调试。
    """
    ans_pat = f"{SOLUTION_START}(.*?){SOLUTION_END}"
    responses = [re.findall(ans_pat, c, re.DOTALL) for c in completions]
    print(
        "-" * 20,
        f"\nQuestion:\n{prompts[0]}",
        f"\nAnswer:\n{answer[0]}",
        f"\nResponse:\n{completions[0]}",
    )
    scores = []
    for matches, truth in zip(responses, answer):
        prediction = matches[0].replace("\n", "") if len(matches) == 1 else None
        scores.append(2.0 if prediction == truth else 0.0)
    return scores


###############################################################################
# 5. 训练 & 推理
###############################################################################

def build_trainer(model, tokenizer, dataset):
    """创建 GRPOTrainer，importance_sampling_level='sequence' 对齐 GSPO 思路。"""
    FastVisionModel.for_training(model)

    training_args = GRPOConfig(
        learning_rate=LEARNING_RATE,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_PROMPT_LEN,
        max_completion_length=MAX_COMPLETION_LEN,
        importance_sampling_level="sequence",
        mask_truncated_completions=False,
        loss_type="dr_grpo",
        max_steps=MAX_STEPS,
        save_steps=MAX_STEPS,
        max_grad_norm=0.1,
        seed=SEED,
        output_dir="outputs",
        report_to="none",
    )

    return GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        reward_funcs=[formatting_reward_func, correctness_reward_func],
        train_dataset=dataset,
    )


def preview_inference(model, tokenizer, sample):
    """快速推理：输入图像 + 指令，观察输出。"""
    FastVisionModel.for_inference(model)

    instruction = (
        f"{sample['question']}，推理请放在 {REASONING_START}/{REASONING_END}，"
        f"答案放在 {SOLUTION_START}/{SOLUTION_END} 且写浮点数。"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        sample["decoded_image"],
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=128,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        use_cache=True,
    )


###############################################################################
# 6. 主流程
###############################################################################

def main():
    model, tokenizer = load_model_and_tokenizer()
    model = attach_lora(model)

    raw_dataset = load_dataset(DATASET_NAME, split=DATA_SPLIT)
    dataset = prepare_dataset(tokenizer)
    print(f"可用样本数：{len(dataset)}（仅保留数值答案）")

    print("\n=== 训练前推理示例 ===")
    preview_inference(model, tokenizer, raw_dataset[5])

    trainer = build_trainer(model, tokenizer, dataset)
    trainer.train()

    print("\n=== 训练后推理示例 ===")
    preview_inference(model, tokenizer, raw_dataset[100])

    model.save_pretrained("gemma3_gspo_lora")
    tokenizer.save_pretrained("gemma3_gspo_lora")
    print("LoRA 已保存到 gemma3_gspo_lora，可按需 push_to_hub 或合并导出。")


if __name__ == "__main__":
    main()
