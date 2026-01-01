# -*- coding: utf-8 -*-
"""
Qwen3-VL-8B Vision 微调脚本（中文精简版）

本脚本参考 Unsloth 官方 Notebook，保留本地训练所需的核心部分，
去掉 Colab/安装/营销内容。任务示例：把手写公式图像转成 LaTeX 文本。

流程：
1. 加载 Qwen3-VL-8B-Instruct 4bit Vision 模型 + Tokenizer；
2. 注入 LoRA，可选择微调视觉/语言/Attention/MLP 子模块；
3. 读取 LaTeX_OCR 数据集，转成 Vision Chat 对话；
4. 训练前后做推理对比；
5. 使用 SFTTrainer + UnslothVisionDataCollator 完成监督微调；
6. 保存 LoRA，演示可选的再次加载。
"""

import torch
from datasets import load_dataset
from transformers import TextStreamer
from trl import SFTTrainer, SFTConfig

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

###############################################################################
# 1. 基本配置（按需修改）
###############################################################################

MODEL_NAME = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
DATASET_NAME = "unsloth/LaTeX_OCR"
INSTRUCTION = "请把这张数学公式图片写成对应的 LaTeX 代码。"
SEED = 3407

PER_DEVICE_BATCH = 2
GRAD_ACCUM_STEPS = 4
MAX_STEPS = 30          # 正式训练可改用 num_train_epochs=1
LEARNING_RATE = 2e-4
MAX_SEQ_LEN = 2048


###############################################################################
# 2. 加载模型 + 注入 LoRA
###############################################################################

def load_base_model():
    """
    载入 4bit 量化的 Qwen3-VL-8B 模型。
    load_in_4bit=True 显存压力小，use_gradient_checkpointing="unsloth" 支持长上下文。
    """
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    return model, tokenizer


def attach_lora(model):
    """
    为模型添加 LoRA 适配器。
    默认视觉/语言/注意力/MLP 四类模块全部允许微调，可按需关掉。
    """
    return FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
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
        # target_modules="all-linear",  # 需要更细粒度时可手动指定
    )


###############################################################################
# 3. 数据准备：LaTeX OCR -> Vision 对话
###############################################################################

def convert_sample(sample):
    """
    将单条样本转换成多模态对话格式。
    - user 端包含指令文本 + 图片；
    - assistant 端输出对应的 LaTeX 文本。
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": INSTRUCTION},
                    {"type": "image", "image": sample["image"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["text"]}],
            },
        ]
    }


def prepare_dataset():
    """
    下载 LaTeX_OCR 数据集（train split），逐条转换后返回。
    """
    raw_dataset = load_dataset(DATASET_NAME, split="train")
    converted = [convert_sample(sample) for sample in raw_dataset]
    return raw_dataset, converted


###############################################################################
# 4. 推理与训练工具函数
###############################################################################

def preview_inference(model, tokenizer, image, instruction):
    """
    训练前/后快速推理：输入一张图和指令，看看模型目前的输出。
    """
    FastVisionModel.for_inference(model)  # 切换到推理模式，关闭梯度

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
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=128,
        temperature=1.5,
        min_p=0.1,
        use_cache=True,
    )


def train(model, tokenizer, dataset):
    """
    Vision SFT 训练：
    - 必须用 UnslothVisionDataCollator，否则图片无法进 batch；
    - remove_unused_columns 等设置要关闭，以保留 messages 结构。
    """
    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        args=SFTConfig(
            per_device_train_batch_size=PER_DEVICE_BATCH,
            gradient_accumulation_steps=GRAD_ACCUM_STEPS,
            warmup_steps=5,
            max_steps=MAX_STEPS,
            # num_train_epochs=1,
            learning_rate=LEARNING_RATE,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=SEED,
            output_dir="outputs",
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=MAX_SEQ_LEN,
        ),
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    print(f"GPU = {gpu_stats.name}, total VRAM = {gpu_stats.total_memory / 1024**3:.1f} GB")
    start_reserved = torch.cuda.max_memory_reserved() / 1024 ** 3

    metrics = trainer.train().metrics

    end_reserved = torch.cuda.max_memory_reserved() / 1024 ** 3
    print(f"训练耗时: {metrics['train_runtime']:.1f}s (~{metrics['train_runtime']/60:.2f}min)")
    print(f"峰值预留显存: {end_reserved:.2f} GB, LoRA 部分新增 {end_reserved - start_reserved:.2f} GB")


###############################################################################
# 5. 保存与加载 LoRA
###############################################################################

def save_lora(model, tokenizer, save_dir="qwen3vl_lora"):
    """保存 LoRA 适配器（不含基座），下次直接加载即可复现表现。"""
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"LoRA 已保存到 {save_dir}")


def load_lora_for_inference(save_dir="qwen3vl_lora"):
    """可选：演示如何重新加载刚保存的 LoRA 并进入推理模式。"""
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=save_dir,
        load_in_4bit=True,
    )
    FastVisionModel.for_inference(model)
    return model, tokenizer


###############################################################################
# 6. 主流程
###############################################################################

def main():
    model, tokenizer = load_base_model()
    model = attach_lora(model)

    raw_dataset, converted_dataset = prepare_dataset()
    print(f"数据集共 {len(converted_dataset)} 条手写公式样本。")

    # 训练前预览
    print("\n=== 训练前推理示例 ===")
    preview_inference(model, tokenizer, raw_dataset[2]["image"], INSTRUCTION)

    # 训练
    train(model, tokenizer, converted_dataset)

    # 训练后预览
    print("\n=== 训练后推理示例 ===")
    preview_inference(model, tokenizer, raw_dataset[2]["image"], INSTRUCTION)

    # 保存 LoRA
    save_lora(model, tokenizer)

    # 如需立即测试重新加载，可取消注释
    # reloaded_model, reloaded_tokenizer = load_lora_for_inference()
    # preview_inference(reloaded_model, reloaded_tokenizer, raw_dataset[2]["image"], INSTRUCTION)


if __name__ == "__main__":
    main()
