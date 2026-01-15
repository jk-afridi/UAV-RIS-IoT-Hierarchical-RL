#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import gym
import numpy as np
import torch
import sys
from gym.envs.registration import register

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core import utils

# 設置輸出編碼以避免Windows上的編碼問題
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# 註冊自定義的 RIS_MISO_PDA 環境到 Gym
register(
    id='RIS_MISO_PDA-v0',
    entry_point='core.environment:RIS_MISO_PDA',
)

def train_ddpg(args, env, device):
    """
    訓練DDPG算法
    
    參數:
        args: 命令行參數
        env: 環境
        device: 計算設備
    """
    from algorithms.DDPG import DDPG
    from core.utils import ExperienceReplayBuffer
    
    # 獲取狀態和動作空間維度
    state_dim = env.state_dim
    action_dim = env.action_dim

    # 設置代理參數
    agent_kwargs = {
        "state_dim": state_dim,
        "action_space": env.action_space,
        "M": args.num_antennas,
        "N": args.num_RIS_elements,
        "K": args.num_users,
        "power_t": args.power_t,
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "device": device,
        "discount": args.discount,
        "tau": args.tau,
        "noise_std": 0.3,
        "noise_decay": True
    }

    # 初始化DDPG算法
    agent = DDPG(**agent_kwargs)
    replay_buffer = ExperienceReplayBuffer(state_dim, action_dim, max_size=args.buffer_size)
    replay_buffer.device = device

    # 初始化追蹤變量
    instant_rewards = []
    instant_mismatch_rewards = []
    state, done = env.reset(), False
    max_reward = 0
    max_mismatch_reward = 0
    episode_time_steps = 0

    # 對初始狀態進行白化處理
    state = utils.whiten(state)

    print(f"🚀 開始DDPG訓練 | L={args.num_RIS_elements} | β_min={args.beta_min}")

    # 主訓練循環
    for t in range(int(args.max_time_steps)):
        episode_time_steps += 1

        # 根據當前策略選擇動作
        action = agent.select_action(state, evaluate=False, timestep=t, max_timesteps=args.max_time_steps)

        # 執行動作並獲取環境反饋
        next_state, reward, done, info = env.step(action)

        mismatch_reward = info["true reward"]
        next_state = utils.whiten(next_state)

        # 更新獎勵記錄
        instant_rewards.append(reward)
        instant_mismatch_rewards.append(mismatch_reward)

        if reward > max_reward:
            max_reward = reward

        if mismatch_reward > max_mismatch_reward:
            max_mismatch_reward = mismatch_reward

        # 標準化獎勵
        reward = reward - np.mean(instant_rewards)

        # 存儲經驗到回放緩衝區
        replay_buffer.add(state, action, next_state, reward, float(done))

        state = next_state

        # 定期保存結果和模型
        if (t + 1) % 100 == 0:
            current_avg_reward = np.mean(instant_rewards[-100:]) if len(instant_rewards) >= 100 else np.mean(instant_rewards)
            print(f"🎯 步驟: {t + 1:>6}/{args.max_time_steps} | "
                  f"當前獎勵: {reward:>7.3f} | "
                  f"最大獎勵: {max_reward:>7.3f} | "
                  f"平均獎勵: {current_avg_reward:>7.3f} | "
                  f"緩衝區: {replay_buffer.size:>5}")

        # 更新代理參數（只有當緩衝區有足夠樣本時）
        if replay_buffer.size > args.batch_size:
            agent.update_parameters(replay_buffer, args.batch_size)

    return instant_mismatch_rewards, agent

def train_ppo(args, env, device):
    """
    訓練PPO算法
    
    參數:
        args: 命令行參數
        env: 環境
        device: 計算設備
    """
    from algorithms.PPO import PPO
    
    # 獲取狀態和動作空間維度
    state_dim = env.state_dim
    action_dim = env.action_dim

    # 設置代理參數（PPO特定）
    agent_kwargs = {
        "state_dim": state_dim,
        "action_space": env.action_space,
        "M": args.num_antennas,
        "N": args.num_RIS_elements,
        "K": args.num_users,
        "power_t": args.power_t,
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "device": device,
        "n_steps": 2048,  # 每次更新前收集的步數
        "n_epochs": 10,   # 每次更新的輪數
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "gamma": args.discount,
        "gae_lambda": 0.95
    }

    # 初始化PPO算法
    agent = PPO(**agent_kwargs)

    # 初始化追蹤變量
    instant_rewards = []
    instant_mismatch_rewards = []
    state, done = env.reset(), False
    max_reward = 0
    max_mismatch_reward = 0
    episode_time_steps = 0

    # 對初始狀態進行白化處理
    state = utils.whiten(state)

    print(f"🚀 開始PPO訓練 | L={args.num_RIS_elements} | β_min={args.beta_min}")
    print(f"📊 PPO配置: n_steps={agent_kwargs['n_steps']}, n_epochs={agent_kwargs['n_epochs']}")

    # 主訓練循環
    t = 0
    while t < args.max_time_steps:
        episode_time_steps += 1
        t += 1

        # 根據當前策略選擇動作
        action, log_prob, value = agent.select_action(state, evaluate=False)

        # 執行動作並獲取環境反饋
        next_state, reward, done, info = env.step(action)

        mismatch_reward = info["true reward"]
        next_state = utils.whiten(next_state)

        # 更新獎勵記錄
        instant_rewards.append(reward)
        instant_mismatch_rewards.append(mismatch_reward)

        if reward > max_reward:
            max_reward = reward

        if mismatch_reward > max_mismatch_reward:
            max_mismatch_reward = mismatch_reward

        # 標準化獎勵
        normalized_reward = reward - np.mean(instant_rewards)

        # 存儲轉換到PPO緩衝區
        agent.store_transition(state, action, normalized_reward, log_prob, value, float(done))

        state = next_state

        # 定期保存結果和打印進度
        if t % 100 == 0:
            current_avg_reward = np.mean(instant_rewards[-100:]) if len(instant_rewards) >= 100 else np.mean(instant_rewards)
            print(f"🎯 步驟: {t:>6}/{args.max_time_steps} | "
                  f"當前獎勵: {reward:>7.3f} | "
                  f"最大獎勵: {max_reward:>7.3f} | "
                  f"平均獎勵: {current_avg_reward:>7.3f} | "
                  f"緩衝區: {agent.buffer.ptr:>5}")

        # 檢查是否準備好更新
        if agent.ready_to_update():
            print(f"🔄 執行PPO更新 (步驟 {t})")
            agent.update_parameters()

        # 環境重置
        if done:
            state, done = env.reset(), False
            state = utils.whiten(state)
            episode_time_steps = 0

    return instant_mismatch_rewards, agent

if __name__ == "__main__":
    # 創建參數解析器
    parser = argparse.ArgumentParser()

    # 算法選擇
    parser.add_argument("--algorithm", default="DDPG", type=str, choices=["DDPG", "PPO"], 
                       help='算法選擇: DDPG | PPO')

    # 環境參數
    parser.add_argument("--env", default="RIS_MISO_PDA-v0", help='環境名稱')
    parser.add_argument("--seed", default=0, type=int, help='隨機種子')
    parser.add_argument("--gpu", default="0", type=int, help='GPU序號')

    # 系統參數
    parser.add_argument("--num_antennas", default=4, type=int, help='基站天線數量')
    parser.add_argument("--num_RIS_elements", default=16, type=int, help='RIS元件數量')
    parser.add_argument("--num_users", default=4, type=int, help='用戶數量')
    parser.add_argument("--power_t", default=30, type=float, help='傳輸功率約束 (dBm)')
    parser.add_argument("--beta_min", default=0.3, type=float, help='最小beta值')

    # 訓練參數
    parser.add_argument("--max_time_steps", default=int(2e4), type=int, help='訓練步數')
    parser.add_argument("--buffer_size", default=int(2e4), type=int, help='經驗回放緩衝區大小 (僅DDPG)')
    parser.add_argument("--batch_size", default=16, type=int, help='批量大小 (僅DDPG)')
    parser.add_argument("--save_model", default=True, type=bool, help='是否保存模型')

    # 網絡參數
    parser.add_argument("--actor_lr", default=1e-3, type=float, help='Actor學習率')
    parser.add_argument("--critic_lr", default=1e-3, type=float, help='Critic學習率')
    parser.add_argument("--discount", default=1.0, type=float, help='折扣因子')
    parser.add_argument("--tau", default=1e-3, type=float, help='軟更新係數')

    # 環境噪聲參數
    parser.add_argument("--awgn_var", default=1e-2, type=float, help='AWGN噪聲方差')
    parser.add_argument("--channel_noise_var", default=1e-2, type=float, help='通道噪聲方差')

    args = parser.parse_args()

    # 設置設備
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"使用GPU: {device}")
    else:
        device = torch.device("cpu")
        print("使用CPU")

    # 設置檔案名稱和保存路徑
    file_name = f"{args.algorithm}_mismatch_{args.seed}"
    save_path = f"Beta_min. = {args.beta_min}, K = {args.num_users}, M = {args.num_antennas}, N = {args.num_RIS_elements}, P_t = {float(args.power_t)}"

    # 打印訓練配置信息
    print("=" * 80)
    print(f"🚀 新算法RIS訓練開始")
    print("=" * 80)
    print(f"📋 算法: {args.algorithm} | 環境: {args.env} | 種子: {args.seed}")
    print(f"📊 參數: L={args.num_RIS_elements} | M={args.num_antennas} | K={args.num_users} | β_min={args.beta_min}")
    print(f"⚙️  設備: {device} | 訓練步數: {args.max_time_steps}")
    print(f"💾 保存模型: {args.save_model} | 保存路徑: {save_path}")
    if torch.cuda.is_available():
        print(f"🔥 GPU信息: {torch.cuda.get_device_name(0)} | 顯存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print("=" * 80)

    # 創建保存目錄
    if args.save_model and not os.path.exists(f"./Models/{save_path}"):
        os.makedirs(f"./Models/{save_path}")

    if not os.path.exists(f"./Results/{save_path}"):
        os.makedirs(f"./Results/{save_path}")

    # 設置環境參數（只在mismatch場景下訓練）
    environment_kwargs = {
        "num_antennas": args.num_antennas,
        "num_RIS_elements": args.num_RIS_elements,
        "num_users": args.num_users,
        "mismatch": True,  # mismatch場景
        "channel_est_error": True,
        "cascaded_channels": True,
        "beta_min": args.beta_min,
        "theta_bar": 0.0,
        "kappa_bar": 1.5,
        "AWGN_var": args.awgn_var,
        "channel_noise_var": args.channel_noise_var,
        "seed": args.seed,
    }

    # 創建環境
    env = gym.make(args.env, **environment_kwargs)

    # 設置隨機種子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 根據算法類型進行訓練
    if args.algorithm == "DDPG":
        instant_mismatch_rewards, agent = train_ddpg(args, env, device)
    elif args.algorithm == "PPO":
        instant_mismatch_rewards, agent = train_ppo(args, env, device)
    else:
        raise ValueError(f"不支持的算法: {args.algorithm}")

    # 保存訓練結果
    np.save(f"./Results/{save_path}/{file_name}", instant_mismatch_rewards)
    if args.save_model:
        agent.save(f"./Models/{save_path}/{file_name}")

    # 訓練完成總結
    print("=" * 80)
    print("🎉 訓練完成！")
    print("=" * 80)
    print(f"✅ 最終統計:")
    print(f"   算法: {args.algorithm}")
    print(f"   最大失配獎勵: {max(instant_mismatch_rewards):.4f}")
    print(f"   最終平均獎勵: {np.mean(instant_mismatch_rewards[-100:]):.4f}")
    print(f"   訓練步數: {args.max_time_steps}")
    print(f"💾 結果已保存至: ./Results/{save_path}/{file_name}.npy")
    if args.save_model:
        print(f"💾 模型已保存至: ./Models/{save_path}/{file_name}")
    print("=" * 80) 
