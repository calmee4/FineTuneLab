# -*- coding: utf-8 -*-
"""
Llama3.2-11B Vision 微调脚本（中文注释加强版）

本脚本复刻并精简自 Unsloth 官方 Llama3.2 Vision Notebook，去掉了与 Colab、
营销、安装演示相关的长段落，只保留在本地环境即可运行的核心步骤。
流程概览：
1. 加载 4bit 量化的 Llama3.2-11B-Vision 模型 + Tokenizer。
2. 挂载 LoRA 适配器，可选择微调视觉/语言/Attention/MLP 子模块。
3. 读取 Radiology_mini 医学影像数据，并转成 vision-chat 所需的对话格式。
4. 使用 TRL 的 SFTTrainer（配合 UnslothVisionDataCollator）进行监督微调。
5. 训练前后做推理对比，确认 LoRA 生效，并演示如何保存/再次加载。

全程附带详细中文注释，便于逐行理解。
"""

import torch
from datasets import load_dataset
from transformers import TextStreamer
from trl import SFTTrainer, SFTConfig

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

###############################################################################
# 1. 基础配置
###############################################################################

MODEL_NAME = "unsloth/Llama-3.2-11B-Vision-Instruct"
DATASET_NAME = "unsloth/Radiology_mini"
INSTRUCTION = "你是一名资深放射科医生，请准确描述这张影像。"
SEED = 3407

# 训练相关超参，可按需修改
PER_DEVICE_BATCH = 2
GRAD_ACCUM_STEPS = 4
MAX_STEPS = 30  # 正式训练可改成 num_train_epochs=1 并移除 max_steps
LEARNING_RATE = 2e-4
MAX_SEQ_LEN = 2048


###############################################################################
# 2. 加载模型并挂载 LoRA
###############################################################################

def load_base_model():
    """
    加载 4bit 量化的 Vision 模型：
    - load_in_4bit=True 可以显著降低显存占用；
    - use_gradient_checkpointing="unsloth" 支持长上下文且保持速度。
    """
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    return model, tokenizer


def attach_lora(model):
    """
    为视觉模型注入 LoRA。
    这里默认同时微调视觉 + 语言 + Attention + MLP，必要时你可以按需关闭其中任何子模块。
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
        # target_modules="all-linear",  # 如需指定更细的层，可手动打开
    )


###############################################################################
# 3. 数据准备
###############################################################################

def convert_sample(sample):
    """
    将原始 Radiology_mini 样本转成 vision 对话格式。
    注意：content 中必须同时包含 text 与 image 字段，否则 tokenizer 无法正常拼接。
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
                "content": [{"type": "text", "text": sample["caption"]}],
            },
        ]
    }


def prepare_dataset():
    """
    读取 radiology 数据集 -> 转换格式 -> 返回列表，方便直接交给 SFTTrainer。
    这里用的是 mini 版本（几百条），大数据集可以换成完整 ROCO。
    """
    raw_dataset = load_dataset(DATASET_NAME, split="train")
    converted = [convert_sample(sample) for sample in raw_dataset]
    # 这是对大模型的captioning结构做微调
    return raw_dataset, converted


###############################################################################
# 4. 训练 & 推理工具函数
###############################################################################

def preview_inference(model, tokenizer, image, instruction):
    """
    用于训练前/训练后的快速推理：将图像 + 指令组合后喂给模型，并流式打印输出。
    """
    # 接着就是做推理？
    FastVisionModel.for_inference(model)
    # 关闭梯度

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
#这里的token 会对图片模态单独处理，messages
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
    真正的监督微调入口：
    - UnslothVisionDataCollator 是 vision finetuning 的关键，否则无法把图片编码到 batch 里；
    - remove_unused_columns 等参数必须关闭，让数据集保持自带的 message 结构。
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
            # num_train_epochs=1,  # 想跑满一个 epoch 可改用这个并注释掉 max_steps
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
    print(f"训练耗时: {metrics['train_runtime']:.1f} s (~{metrics['train_runtime']/60:.2f} min)")
    print(f"峰值预留显存: {end_reserved:.2f} GB, 训练新增 {end_reserved - start_reserved:.2f} GB")
    return trainer


###############################################################################
# 5. 保存/加载 LoRA
###############################################################################

def save_lora(model, tokenizer, save_dir="lora_model"):
    """保存 LoRA 适配器（不含基座权重），方便下次直接加载。"""
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"LoRA 权重已保存到 {save_dir}")


def load_lora_for_inference(save_dir="lora_model"):
    """可选：示例如何重新加载刚保存的 LoRA 继续推理。"""
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
    print(f"数据集共 {len(converted_dataset)} 条样本，用于放射影像描述任务。")

    # 训练前做一次推理，感受原始表现
    print("\n=== 训练前推理示例 ===")
    preview_inference(model, tokenizer, raw_dataset[0]["image"], INSTRUCTION)

    # 训练
    train(model, tokenizer, converted_dataset)

    # 训练后再次推理对比
    print("\n=== 训练后推理示例 ===")
    preview_inference(model, tokenizer, raw_dataset[0]["image"], INSTRUCTION)

    # 保存 LoRA
    save_lora(model, tokenizer)

    # 如需立即测试加载效果，可取消下面注释
    # reloaded_model, reloaded_tokenizer = load_lora_for_inference()
    # preview_inference(reloaded_model, reloaded_tokenizer, raw_dataset[0]["image"], INSTRUCTION)


if __name__ == "__main__":
    main()
