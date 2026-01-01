# -*- coding: utf-8 -*-
"""
Unit 1 - LunarLander 深度强化学习示例脚本

相比原来的 Colab Notebook，本脚本做了如下整理：
1. 去掉了所有教学用的大段英文说明和 HTML/图片。
2. 去掉了 Colab 专用代码（虚拟显示、强制重启、notebook magic 等）。
3. 只保留了核心强化学习流程代码：环境 -> 向量化环境 -> 模型 -> 训练 -> 评估 -> Hub 上传/加载。
4. 所有保留的注释都带中文说明，方便对照学习。
"""

import os
from typing import Dict, Any

import gymnasium as gym  # Gymnasium 环境库 (RL environment library)
from huggingface_hub import notebook_login  # 用于登录 Hugging Face Hub (login to HF Hub)
from huggingface_sb3 import package_to_hub, load_from_hub  # 与 SB3 集成的 Hub 工具 (HF-SB3 helpers)

from stable_baselines3 import PPO  # PPO 算法实现 (RL algorithm implementation)
from stable_baselines3.common.env_util import make_vec_env  # 创建向量化环境 (vectorized env helper)
from stable_baselines3.common.monitor import Monitor  # 包装环境以记录回报等信息 (monitor wrapper)
from stable_baselines3.common.evaluation import evaluate_policy  # 评估策略 (policy evaluation)


def inspect_single_env() -> None:
    """简单检查 LunarLander 单环境的观测空间和动作空间

    Simple inspection of observation and action spaces for a single env.
    """
    # 创建单个 LunarLander 环境
    # Create a single LunarLander environment
    env = gym.make("LunarLander-v2")
    obs, info = env.reset()

    print("===== OBSERVATION SPACE / 观测空间 =====")
    print("Shape / 形状:", env.observation_space.shape)
    print("Sample observation / 随机观测样本:", env.observation_space.sample())

    print("\n===== ACTION SPACE / 动作空间 =====")
    print("N / 离散动作数量:", env.action_space.n)
    print("Sample action / 随机动作样本:", env.action_space.sample())

    env.close()


def make_vector_env(n_envs: int = 16):
    """创建向量化环境（多个并行的 LunarLander 环境）

    Create a vectorized environment with multiple parallel LunarLander envs.
    """
    # make_vec_env 会创建 n_envs 个相同的环境，并在内部并行运行
    # make_vec_env creates n_envs identical envs running in parallel
    vec_env = make_vec_env("LunarLander-v2", n_envs=n_envs)
    return vec_env


def create_ppo_model(env) -> PPO:
    """创建 PPO 模型（使用 MLP 策略）

    Create a PPO model with an MLP policy.
    """
    # 这里使用 MlpPolicy，因为输入是一个向量（8 维状态）
    # Use MlpPolicy because the input is a vector (8-dim state)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        # 下列超参数是课程中给出的一个比较好的起点
        # The following hyper-parameters are good starting points from the course
        n_steps=1024,       # 每次更新前收集的环境步数 (rollout length)
        batch_size=64,      # 每次梯度更新使用的样本数量 (minibatch size)
        n_epochs=4,         # 每个 batch 上迭代多少次 (number of epochs per update)
        gamma=0.999,        # 折扣因子，越接近 1 越看重长期奖励 (discount factor)
        gae_lambda=0.98,    # GAE 参数，控制 bias-variance 权衡 (GAE lambda)
        ent_coef=0.01,      # 熵系数，鼓励探索 (entropy coefficient)
        verbose=1,          # 日志详细程度 (logging verbosity)
    )
    return model


def train_model(model: PPO, total_timesteps: int = int(1e6)) -> PPO:
    """训练 PPO 模型

    Train the PPO model for a given number of timesteps.
    """
    # total_timesteps 是与环境交互的总步数
    # total_timesteps is the total number of environment steps
    model.learn(total_timesteps=total_timesteps)
    return model


def evaluate_trained_model(model: PPO, n_eval_episodes: int = 10) -> Dict[str, float]:
    """在独立环境上评估训练好的模型

    Evaluate the trained model on a separate env.
    """
    # 使用 Monitor 包装环境，自动记录回报等信息
    # Wrap env with Monitor to record returns etc.
    eval_env = Monitor(gym.make("LunarLander-v2"))
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    return {"mean_reward": mean_reward, "std_reward": std_reward}


def save_model_locally(model: PPO, save_path: str = "ppo-LunarLander-v2") -> None:
    """将训练好的模型保存到本地

    Save the trained model locally.
    """
    os.makedirs(save_path, exist_ok=True)
    model.save(os.path.join(save_path, "model"))
    print(f"Model saved to / 模型已保存到: {save_path}")


def push_model_to_hub(
    model: PPO,
    repo_id: str,
    commit_message: str = "Add trained PPO LunarLander model",
    env_id: str = "LunarLander-v2",
) -> None:
    """将模型推送到 Hugging Face Hub

    Push the trained model to the Hugging Face Hub.
    """
    # 第一步：在浏览器中提前创建好 repo，或让 package_to_hub 自动创建
    # Step 1: create repo beforehand or let package_to_hub create it

    # 登录 Hugging Face 账户（第一次运行时会在终端提示输入 token）
    # Login to HF account (enter your token when prompted)
    notebook_login()

    # package_to_hub 会：
    # 1. 保存模型和配置到临时目录
    # 2. 创建或更新 Hub 仓库
    # 3. 上传模型文件、训练参数、README 等
    package_to_hub(
        model=model,
        model_name="ppo-LunarLander-v2",  # 本地临时目录名称 (local tmp dir name)
        model_architecture="PPO",        # 算法名称 (algorithm name)
        env_id=env_id,                   # 环境 ID (environment id)
        repo_id=repo_id,                 # 你的 Hub 仓库，如 "username/ppo-LunarLander-v2"
        commit_message=commit_message,   # 提交说明 (commit message)
    )


def load_model_from_hub(
    repo_id: str,
    filename: str = "ppo-LunarLander-v2.zip",
    custom_objects: Dict[str, Any] | None = None,
) -> PPO:
    """从 Hugging Face Hub 加载已经训练好的 SB3 模型

    Load a trained SB3 model from the Hugging Face Hub.
    """
    if custom_objects is None:
        # 一般只在旧 Python 版本或自定义超参不兼容时才需要 custom_objects
        # custom_objects is only needed when there are incompatibilities
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    # 下载模型文件到本地缓存
    # Download the model file to local cache
    checkpoint_path = load_from_hub(repo_id=repo_id, filename=filename)

    # 使用 PPO.load 载入模型
    # Load the PPO model from checkpoint
    model = PPO.load(checkpoint_path, custom_objects=custom_objects, print_system_info=True)
    return model


if __name__ == "__main__":
    # 1. 简单检查单环境的观测和动作空间 (inspect spaces)
    inspect_single_env()

    # 2. 创建 16 个并行环境 (create 16 parallel envs)
    vec_env = make_vector_env("LunarLander-v2", n_envs=16)

    # 3. 创建 PPO 模型 (create PPO model)
    ppo_model = create_ppo_model(vec_env)

    # 4. 训练模型 (train model)
    #    注意：1e6 步在 CPU 上会比较慢，先试 1e5 看效果也可以
    #    Note: 1e6 steps may be slow on CPU; you can try 1e5 first.
    ppo_model = train_model(ppo_model, total_timesteps=int(1e5))

    # 5. 本地评估模型性能 (evaluate locally)
    evaluate_trained_model(ppo_model, n_eval_episodes=10)

    # 6. 本地保存模型 (save locally)
    save_model_locally(ppo_model, save_path="ppo-LunarLander-v2")

    # 7. 如需推送到 Hub，取消下面两行注释，并把 repo_id 改成你自己的
    #    If you want to push to HF Hub, uncomment below and set your own repo_id.
    # my_repo_id = "your-username/ppo-LunarLander-v2"
    # push_model_to_hub(ppo_model, repo_id=my_repo_id)

    # 8. 如需从 Hub 加载模型，可使用 load_model_from_hub
    #    To load a model from Hub, use load_model_from_hub as shown below.
    # loaded_model = load_model_from_hub(repo_id=my_repo_id, filename="ppo-LunarLander-v2.zip")
    # evaluate_trained_model(loaded_model, n_eval_episodes=10)

