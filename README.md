# 大模型微调与多模态示例框架

> 本仓库整理了当前的单文件脚本，按模块拆分为 LLM 微调、视觉/多模态评测、通用工具与配置，便于直接放到 GitHub 供学习与复现。

## 目录结构
```
.
├─ README.md                 # 项目说明
├─ requirements.txt          # 依赖列表
├─ .gitignore
├─ configs/                  # 示例配置（待补充）
├─ scripts/                  # 运行脚本（待补充）
├─ src/
│  ├─ llm/                   # 语言模型微调/预训练脚本
│  │   ├─ pretrain_llm.py
│  │   ├─ GRPO_llama.py
│  │   ├─ GRPO_Qwen3_4B.py
│  │   ├─ Llama3.1-8B.py
│  │   └─ Llama3.2-11B_vision.py
│  ├─ vision/                # 视觉/多模态评测
│  │   ├─ Qwen3_vl_8b_vision.py
│  │   └─ gemma3_vision_gspo.py
│  └─ utils/                 # 工具与示例
│      ├─ unit1.py
│      └─ lightning_demo.py
├─ outputs/                  # 运行输出（已加入 gitignore）
└─ run_all_numbered.txt      # 批量运行参考清单（可转脚本）
```

> 说明：为保持原始文件可用，现阶段是将单文件复制到 `src/` 对应子目录，原位置仍保留。

## 快速开始
1) 安装依赖（推荐使用虚拟环境/conda）：  
   ```bash
   pip install -r requirements.txt
   ```
2) 准备数据与模型权重：在 `configs/` 中补充数据路径与模型名称，或直接在脚本内修改。
3) 运行示例（脚本已放在 `scripts/`）：  
   - LLM 预训练/微调：  
     ```bash
     chmod +x scripts/run_llm_pretrain.sh
     DATA_PATH=./data/train.jsonl OUTPUT_DIR=./outputs/llm_pretrain ./scripts/run_llm_pretrain.sh
     ```  
   - GRPO 示例：  
     ```bash
     python src/llm/GRPO_llama.py    # 或 GRPO_Qwen3_4B.py
     ```  
   - 视觉/多模态评测：  
     ```bash
     chmod +x scripts/run_vision_eval.sh
     DATA_PATH=./data/vision_samples OUTPUT_DIR=./outputs/vision_eval ./scripts/run_vision_eval.sh
     ```

## 配置与脚本
- `configs/`：放置 YAML/JSON 配置，包含数据路径、模型名、训练超参、输出目录等。
- `scripts/`：放置一键运行脚本（如 train.sh / eval.sh / run_all.sh），可参考 `run_all_numbered.txt` 转为可执行脚本。

## 后续可完善项
- 将脚本中的硬编码路径/超参迁移到 `configs/`，并在 `scripts/` 提供运行示例。
- 增加日志与可视化（tensorboard/plt）示例。
- 为多模态/视觉部分增加 README 或 notebooks 演示。
- 添加测试集划分与评测指标汇总。

## 许可证
根据实际需求选择合适的开源协议（MIT/Apache-2.0 等），并在 README 中注明。
