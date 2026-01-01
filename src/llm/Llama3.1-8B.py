from unsloth import FastLanguageModel
import torch

# ⭐ 最大序列长度：模型一次能看到的最多 token 数（上下文长度）
#   调大：能处理更长的输入，但显存占用和计算时间都会增加
#   调小：更省显存、更快，但长文本会被截断
max_seq_length = 2048  # 这里选 2048，适合大多数教学和简单微调场景

# ⭐ dtype：模型内部使用的浮点精度类型
#   None：让 unsloth 自动根据显卡选择（T4/V100 用 float16，A100 等新卡用 bfloat16）
#   也可以手动设为 torch.float16 / torch.bfloat16，但不如自动省心
dtype = None

# ⭐ 是否加载为 4bit 量化模型（int4）
#   True：显存占用大幅降低，可以在小显卡上跑，速度也更快；精度会有一点点损失
#   False：使用 16bit 精度，精度更好，但显存需求更高
load_in_4bit = True

# 一些已经提前做了 4bit 量化的模型名称（方便你改成别的模型试验）
# 不参与本脚本运行，只是一个“备选清单”
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 8B，15 万亿 token 训练，下载和推理都更快
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # 还提供了 405B 的 4bit 版本
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # 新版 Mistral 12B
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 小模型
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 系列
]  # 更多模型可在 https://huggingface.co/unsloth 查看

# ⭐ 从 Hugging Face / 本地加载预训练好的基础模型和分词器
model, tokenizer = FastLanguageModel.from_pretrained(
    # 使用的模型名称：可以改成上面 fourbit_models 里的任意一个
    model_name = "unsloth/Meta-Llama-3.1-8B",
    # 最大序列长度：要和上面的 max_seq_length 一致
    max_seq_length = max_seq_length,
    # 数据类型：沿用前面设置的 dtype（None 代表自动选择）
    dtype = dtype,
    # 是否按 4bit 方式加载：沿用前面的 load_in_4bit
    load_in_4bit = load_in_4bit,
    # token = "hf_...",  # 如果加载的是受限模型（比如官方 Llama 2）需要在这里填 Hugging Face 访问令牌
)
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

from datasets import load_dataset
dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

"""现在给模型加上 LoRA 适配器，只训练 1%~10% 的参数就能完成微调，大大节省显存和时间。"""

# ⭐ 把基础模型包装成可微调的 LoRA 模型（PEFT = Parameter-Efficient Fine-Tuning）
model = FastLanguageModel.get_peft_model(
    model,
    # LoRA 的秩（rank）：可以理解为“适配器的宽度”，数值越大表示可学习的参数越多
    # 建议值：8 / 16 / 32 / 64 / 128
    # 调大：表达能力更强，但显存和训练时间都增加
    # 调小：更省显存，但可能学不动复杂任务
    r = 16,
    # 需要应用 LoRA 的模块名称，一般是注意力和 MLP 里的几个线性层
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    # LoRA 的缩放因子，通常和 r 同量级，调大可以稍微增强 LoRA 的影响力
    lora_alpha = 16,
    # LoRA 的 dropout：训练时随机丢弃部分 LoRA 输出，用于正则化
    # 0 表示不丢弃，收敛更稳定，也是 unsloth 针对 0 做了优化
    lora_dropout = 0,
    # 是否在 LoRA 中训练偏置项（bias）："none" 表示不训练偏置，显存占用更少
    bias = "none",
    # 使用梯度检查点技术换取更低显存占用：
    #   "unsloth" 是专门优化过的实现，适合上下文很长的场景
    #   True 也可以开启通用版本；False 则关闭（显存占用会升高）
    use_gradient_checkpointing = "unsloth",
    # 随机种子，保证每次运行初始化一致，方便复现结果
    random_state = 3407,
    # 是否使用 Rank-Stabilized LoRA（RS-LoRA），这里关闭即可
    use_rslora = False,
    # LoftQ 配置：一种量化 + LoRA 的技术，这里不使用
    loftq_config = None,
)

alpaca_prompt = """下面是一条任务指令，以及一段提供更多背景的输入。请根据要求给出合适的回复。

### 指令 (Instruction):
{}

### 输入 (Input):
{}

### 回答 (Response):
{}"""

EOS_TOKEN = tokenizer.eos_token  # ⭐ 结束符号，表示一条样本的结束（避免模型一直生成停不下来）


def formatting_prompts_func(examples):
    """把原始数据集的三列（instruction / input / output）拼成一条训练文本"""
    instructions = examples["instruction"]  # 任务指令
    inputs       = examples["input"]        # 额外输入（可以为空字符串）
    outputs      = examples["output"]       # 参考答案 / 模型要学的回复
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # ⭐ 拼接成 Alpaca 模板，并在最后加上 EOS_TOKEN
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    # datasets 要求返回一个字典，键名是新字段名，这里叫 "text"
    return {"text": texts}


from datasets import load_dataset

# ⭐ 从 Hugging Face Hub 加载 Alpaca 清洗版训练集
dataset = load_dataset("yahma/alpaca-cleaned", split="train")
# 使用刚才的 formatting_prompts_func 把原始字段转换成一列 "text"
# batched=True 表示一次处理一批样本，效率更高
dataset = dataset.map(formatting_prompts_func, batched=True)

"""<a name="Train"></a>
### 训练模型（Train the model）
下面开始训练微调后的 LoRA 模型。
这里为了演示只跑 60 个 step，如果你想完整训练，可以改成 `num_train_epochs=1` 并把 `max_steps=None`。
同时也支持 TRL 提供的 `DPOTrainer` 等其他训练器。
"""

from trl import SFTConfig, SFTTrainer

# ⭐ SFTTrainer：Supervised Fine-Tuning 监督微调工具
trainer = SFTTrainer(
    # 要训练的模型（已经加好了 LoRA）
    model = model,
    # 分词器，用来把文本转成 token id
    tokenizer = tokenizer,
    # 训练用的数据集
    # dataset已经有了
    train_dataset = dataset,
    # 数据集中哪一列是真正要喂给模型的文本，这里是我们在上面 map 出来的 "text"
    dataset_text_field = "text",
    # 最大序列长度，要和上面保持一致
    max_seq_length = max_seq_length,
    # 是否把多条短样本拼接打包成一个长序列来训练
    #   True：对于很多短句子会更快、更省显存，但调试不方便
    #   False：一条样本就是一条序列，逻辑更直观，这里为了教学设为 False
    packing = False,
    # 训练超参数配置
    args = SFTConfig(
        # ⭐ 每块 GPU 上的实际 batch_size
        #   显存吃紧就调小；显存很大可以调大
        per_device_train_batch_size = 2,
        # ⭐ 梯度累积步数：相当于“虚拟 batch_size = batch_size * 累积步数”
        #   例如 2 * 4 = 8，相当于用 batch_size=8 训练，但显存只需放下 2 条样本
        gradient_accumulation_steps = 4,
        # 预热步数：刚开始训练时先用较小学习率，训练更稳定
        warmup_steps = 5,
        # num_train_epochs = 1,  # 如果想按轮数训练，可以打开这一行，把 max_steps 关掉
        # 训练的总 step 数；这里为了快，只跑 60 个 step
        max_steps = 60,
        # ⭐ 学习率：模型参数更新的“步长”
        #   调大：学得更快，但容易发散 / 不稳定
        #   调小：更稳定，但收敛更慢，可能需要更多 step
        learning_rate = 2e-4,
        # 每多少个 step 打印一次日志
        logging_steps = 1,
        # 优化器类型：8bit 的 AdamW，更省显存，适合大模型微调
        optim = "adamw_8bit",
        # 权重衰减：防止过拟合的一种正则化手段，值一般在 0~0.01
        weight_decay = 0.001,
        # 学习率调度策略：linear 表示线性从初始 lr 衰减到 0
        lr_scheduler_type = "linear",
        # 随机种子，保证可复现
        seed = 3407,
        # 输出目录：保存训练中间结果和最终权重
        output_dir = "outputs",
        # 日志上报后端：none 表示只在本地打印
        # 也可以改成 "wandb"、"tensorboard" 等可视化工具
        report_to = "none",
    ),
)

# 显示当前 GPU 显存占用情况（方便你大致了解显存压力）
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# ⭐ 正式开始训练
trainer_stats = trainer.train()

# 显示训练结束后的显存和时间统计
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

"""<a name="Inference"></a>
### 推理（Inference）
下面让模型真正“说两句”，看一看微调后的效果。
你可以随意修改 instruction / input，把 output 留空让模型自己生成。
"""

# alpaca_prompt = 从上面复制过来的模板
FastLanguageModel.for_inference(model)  # ⭐ 启用推理模式，打开更快的推理优化
inputs = tokenizer(
[
    alpaca_prompt.format(
        "我的订单已经3天没发货怎么办?",  # instruction：对模型的指示
        """您好，我立刻为您查询订单状态。根据我们的 政策，""",                   # input：给出前面的几项
        "",                                   # output：留空，让模型来补全
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)

"""你也可以使用 `TextStreamer` 做流式推理，一边生成一边输出，而不是等全部生成完。"""

# alpaca_prompt = 从上面复制过来的模板
FastLanguageModel.for_inference(model)  # 再次确保模型处于推理模式
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Continue the fibonnaci sequence.",  # instruction
        "1, 1, 2, 3, 5, 8",                  # input
        "",                                  # output
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)

"""<a name="Save"></a>
### 保存和加载微调后的模型
这一部分演示如何把微调得到的 LoRA 适配器保存下来。
你可以选择：
- 保存到本地（`save_pretrained`）
- 推送到 Hugging Face Hub（`push_to_hub`）做在线托管

**[注意]** 这里保存的只是 LoRA 适配器本身，而不是完整基座模型。
如果想保存成 16bit / GGUF 等格式，请看下面的示例。
"""

# ⭐ 把 LoRA 适配器保存到本地目录 "lora_model"
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
# 也可以推送到 Hugging Face Hub（需要先在官网生成 token）
# model.push_to_hub("your_name/lora_model", token = "...")
# tokenizer.push_to_hub("your_name/lora_model", token = "...")

"""如果你想在推理时重新加载刚才保存的 LoRA 适配器，把下面的 `False` 改成 `True` 即可。"""

if False:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model",  # 这里填你刚才保存 LoRA 的目录名
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)  # 启用推理模式

# alpaca_prompt = 一定要和上面训练时用的模板保持一致！

inputs = tokenizer(
[
    alpaca_prompt.format(
        "What is a famous tall tower in Paris?",  # instruction
        "",                                       # input（这里留空）
        "",                                       # output（留空，让模型生成）
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)

"""你也可以使用 Hugging Face 的 `AutoModelForPeftCausalLM` 来加载 LoRA。
只有在你不能安装 `unsloth` 时才建议这样做，因为它目前不支持 4bit 加载，推理也会比 unsloth 慢不少。
"""

if False:
    # 一般不推荐这样用，如果可以安装 unsloth，还是优先用 unsloth 更快
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    model = AutoPeftModelForCausalLM.from_pretrained(
        "lora_model",  # 你训练并保存 LoRA 的目录
        load_in_4bit = load_in_4bit,
    )
    tokenizer = AutoTokenizer.from_pretrained("lora_model")

"""### 保存为 float16 以供 VLLM 等推理引擎使用

unsloth 也支持直接保存为 `float16` 或 `int4`（4bit）格式：
* `merged_16bit`：把 LoRA 和基座模型合并，并保存为 float16
* `merged_4bit`：把 LoRA 和基座模型合并，并保存为 4bit 量化模型

你还可以用 `push_to_hub_merged` 把合并后的模型上传到自己的 Hugging Face 账号。
访问 https://huggingface.co/settings/tokens 可以创建访问令牌。
"""

# Merge to 16bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

# Merge to 4bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

# 只保存 LoRA 适配器本身（不合并到基座模型中）
if False:
    model.save_pretrained("model")
    tokenizer.save_pretrained("model")
if False:
    model.push_to_hub("hf/model", token = "")
    tokenizer.push_to_hub("hf/model", token = "")

"""### GGUF / llama.cpp 格式转换
unsloth 也支持直接导出为 GGUF / llama.cpp 可用的格式。
内部会自动克隆 `llama.cpp`，默认量化为 `q8_0`，当然也支持 `q4_k_m` 等其他方案。
使用 `save_pretrained_gguf` 可以本地保存，`push_to_hub_gguf` 可以上传到 Hugging Face。

常见的量化方法（完整列表见 Wiki：https://github.com/unslothai/unsloth/wiki#gguf-quantization-options）：
* `q8_0`  - 转换速度快，占用资源较高，但精度也更好
* `q4_k_m` - 推荐使用。部分权重用 Q6_K，其他用 Q4_K，在精度和体积间折中
* `q5_k_m` - 与 q4_k_m 类似，但整体精度更高一些

【新功能】如果你想微调后直接导出到 Ollama，可以参考官方的 Ollama notebook 示例。
"""

# 保存为 8bit 的 Q8_0 GGUF 文件
if False: model.save_pretrained_gguf("model", tokenizer,)
# 如果要推送到 Hugging Face，需要先在官网创建 token，并把 "hf" 换成你的用户名
if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# 保存为 16bit GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

# 保存为 q4_k_m GGUF（比较推荐的压缩方式）
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")

# 一次性导出多种 GGUF 量化格式（如果你要多种格式，这样会更快）
if False:
    model.push_to_hub_gguf(
        "hf/model",  # 记得把 hf 换成你的 Hugging Face 用户名
        tokenizer,
        quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
        token = "",
    )

"""到这里整个微调流程就结束了：
1. 加载基础模型（支持 4bit）
2. 加 LoRA 适配器，变成可微调模型
3. 准备指令数据（Alpaca 模板）
4. 用 SFTTrainer 做监督微调
5. 运行推理，查看效果
6. 保存 / 导出为 LoRA、合并 16bit/4bit、GGUF 等多种格式
你可以在这个基础上替换自己的数据集、指令模板和超参数，做出属于你自己的 Llama 3.1 微调模型。
"""
