#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from pathlib import Path

import gym
import numpy as np
import torch
from gym.envs.registration import register

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from algorithms import SAC, Beta_Space_Exp_SAC
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

if __name__ == "__main__":
    # 創建參數解析器
    parser = argparse.ArgumentParser()

    # 設置目標函數類型
    parser.add_argument("--objective_function", default="mismatch", type=str, help='是否假設PDA(相位依賴振幅)模型')

    # 訓練相關參數
    parser.add_argument("--policy", default="Beta_Space_Exp_SAC", help='算法選擇 (預設: Beta-Space Exploration)')
    parser.add_argument("--env", default="RIS_MISO_PDA-v0", help='環境名稱')
    parser.add_argument("--seed", default=0, type=int, help='PyTorch和NumPy的隨機種子 (預設: 0)')
    parser.add_argument("--gpu", default="0", type=int, help='多GPU電腦的GPU序號 (預設: 0)')
    parser.add_argument("--max_time_steps", default=int(2e4), type=int, metavar='N', help='訓練步數 (預設: 20000)')
    parser.add_argument("--buffer_size", default=int(2e4), type=int, help='經驗回放緩衝區大小 (預設: 20000)')
    parser.add_argument("--batch_size", default=16, type=int, metavar='N', help='小批量大小 (預設: 16)')
    parser.add_argument("--save_model", default=False, type=bool, help='是否保存模型和優化器參數')
    parser.add_argument("--load_model", default="", help='載入模型的文件名稱，為空則不載入')

    # 環境特定參數
    parser.add_argument("--num_antennas", default=4, type=int, metavar='N', help='基站天線數量 (預設: 4)')
    parser.add_argument("--num_RIS_elements", default=4, type=int, metavar='N', help='RIS元件數量 (預設: 4)')
    parser.add_argument("--num_users", default=4, type=int, metavar='N', help='用戶數量 (預設: 4)')
    parser.add_argument("--power_t", default=30, type=float, metavar='N', help='約束優化的傳輸功率 (dBm, 預設: 30)')
    parser.add_argument("--awgn_var", default=1e-2, type=float, metavar='G', help='加性白高斯噪聲方差 (預設: 0.01)')
    parser.add_argument("--channel_noise_var", default=1e-2, type=float, metavar='G', help='級聯通道中的噪聲方差 (預設: 0.01)')

    # 相位依賴振幅(PDA)模型參數
    parser.add_argument("--mismatch", default=True, type=bool, metavar='N', help='是否假設PDA模型')
    parser.add_argument("--channel_est_error", default=True, type=bool, metavar='N', help='是否假設通道估計誤差')
    parser.add_argument("--cascaded_channels", default=True, type=bool, metavar='N', help='是否假設級聯通道')
    parser.add_argument("--beta_min", default=0.6, type=float, metavar='N', help='PDA計算中的最小beta值 (預設: 0.6)')
    parser.add_argument("--theta_bar", default=0.0, type=float, metavar='N', help='PDA計算中的theta bar值 (預設: 0.0)')
    parser.add_argument("--kappa_bar", default=1.5, type=float, metavar='N', help='PDA計算中的kappa bar值 (預設: 1.5)')

    # 算法特定參數
    parser.add_argument("--discount", default=1.0, metavar='G', help='獎勵的折扣因子 (預設: 1.0)')
    parser.add_argument("--tau", default=1e-3, type=float, metavar='G', help='目標網絡軟/硬更新的學習率 (預設: 0.001)')
    parser.add_argument("--actor_lr", default=1e-3, type=float, metavar='G', help='演員(和探索者)網絡的學習率 (預設: 0.001)')
    parser.add_argument("--critic_lr", default=1e-3, type=float, metavar='G', help='評論家網絡的學習率 (預設: 0.001)')
    parser.add_argument("--decay", default=0.0, type=float, metavar='G', help='網絡的衰減率 (預設: 0.0)')

    # SAC特定參數
    parser.add_argument('--policy_type', default="Gaussian", help='SAC策略類型: Gaussian | Deterministic (預設: Gaussian)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N', help='每個訓練時間步長的評論家函數更新次數 (預設: 1)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G', help='溫度參數α決定熵項相對於獎勵的重要性 (預設: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G', help='自動調整α (預設: False)')

    # Beta空間探索特定參數
    parser.add_argument("--exp_regularization_term", default=0.3, type=float, metavar='G', help='探索正則化項的初始值 (預設: 0.3)')
    parser.add_argument("--linear_schedule_exp_regularization", default=True, type=bool, metavar='G', help='是否線性調度探索正則化項')

    args = parser.parse_args()

    # 根據目標函數設置參數
    if args.objective_function == "mismatch":
        args.mismatch = False
    elif args.objective_function == "golden":
        args.channel_est_error = False

    if "Beta_Space" in args.policy:
        args.mismatch = False
        args.channel_est_error = True

    # 設置設備(CPU/GPU) - 優先使用GPU加速
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"使用GPU: {device}")
    else:
        device = torch.device("cpu")
        print("CUDA不可用，使用CPU")

    # 設置檔案名稱和保存路徑
    file_name = f"{args.policy}_{args.objective_function}_{args.seed}"
    save_path = f"Beta_min. = {args.beta_min}, K = {args.num_users}, M = {args.num_antennas}, N = {args.num_RIS_elements}, P_t = {float(args.power_t)}"

    # 打印訓練配置信息
    print("=" * 80)
    print(f"🚀 RIS訓練開始")
    print("=" * 80)
    print(f"📋 配置: {args.policy} | 環境: {args.env} | 種子: {args.seed} | 場景: {args.objective_function.capitalize()}")
    print(f"📊 參數: L={args.num_RIS_elements} | M={args.num_antennas} | K={args.num_users} | β_min={args.beta_min}")
    print(f"⚙️  設備: {device} | 訓練步數: {args.max_time_steps} | 批次大小: {args.batch_size}")
    print(f"💾 保存模型: {args.save_model} | 保存路徑: {save_path}")
    if torch.cuda.is_available():
        print(f"🔥 GPU信息: {torch.cuda.get_device_name(0)} | 顯存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print("=" * 80)

    # 創建保存目錄
    if args.save_model and not os.path.exists(f"./Models/{save_path}"):
        os.makedirs(f"./Models/{save_path}")

    if not os.path.exists(f"./Results/{save_path}"):
        os.makedirs(f"./Results/{save_path}")

    # 設置環境參數
    environment_kwargs = {
        "num_antennas": args.num_antennas,
        "num_RIS_elements": args.num_RIS_elements,
        "num_users": args.num_users,
        "mismatch": args.mismatch,
        "channel_est_error": args.channel_est_error,
        "cascaded_channels": args.cascaded_channels,
        "beta_min": args.beta_min,
        "theta_bar": args.theta_bar,
        "kappa_bar": args.kappa_bar,
        "AWGN_var": args.awgn_var,
        "channel_noise_var": args.channel_noise_var,
        "seed": args.seed,
    }

    # 創建環境
    env = gym.make(args.env, **environment_kwargs)

    # 設置隨機種子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 獲取狀態和動作空間維度
    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = 1.0

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
        "policy_type": args.policy_type,
        "alpha": args.alpha,
        "target_update_interval": args.target_update_interval,
        "automatic_entropy_tuning": args.automatic_entropy_tuning,
        "device": device,
        "discount": args.discount,
        "tau": args.tau
    }

    # 初始化算法
    if args.policy == "SAC":
        agent = SAC.SAC(**agent_kwargs)
        replay_buffer = utils.ExperienceReplayBuffer(state_dim, action_dim, max_size=args.buffer_size)
    elif args.policy == "Beta_Space_Exp_SAC":
        agent = Beta_Space_Exp_SAC.Beta_Space_Exp_SAC(**agent_kwargs, beta_min=args.beta_min)
        replay_buffer = utils.BetaExperienceReplayBuffer(state_dim, action_dim, args.num_RIS_elements, args.buffer_size)
    else:
        raise NotImplementedError("無效的算法名稱")
    
    # 確保replay_buffer使用正確的設備
    replay_buffer.device = device
    print(f"經驗回放緩衝區設備: {replay_buffer.device}")

    # 載入預訓練模型（如果指定）
    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        agent.load(f"./models/{policy_file}")

    # 初始化追蹤變量
    instant_rewards = []
    instant_mismatch_rewards = []
    state, done = env.reset(), False
    max_reward = 0
    max_mismatch_reward = 0
    episode_time_steps = 0
    episode_num = 0

    # 對初始狀態進行白化處理
    state = utils.whiten(state)
    exp_regularization = args.exp_regularization_term

    # 主訓練循環
    for t in range(int(args.max_time_steps)):
        episode_time_steps += 1

        # 根據當前策略選擇動作
        if args.policy == "SAC":
            action = agent.select_action(state)
        elif "Beta_Space" in args.policy:
            action, beta = agent.select_action(state, exp_regularization)
        else:
            action = (agent.select_action(np.array(state)) + np.random.normal(0, max_action * args.exploration_noise, size=action_dim)).clip(-max_action, max_action)

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
        if "Beta_Space" in args.policy:
            replay_buffer.add(state, action, beta, next_state, reward, float(done))
        else:
            replay_buffer.add(state, action, next_state, reward, float(done))

        state = next_state

        # 定期保存結果和模型
        if (t + 1) % 100 == 0:
            current_avg_reward = np.mean(instant_rewards[-100:]) if len(instant_rewards) >= 100 else np.mean(instant_rewards)
            current_avg_mismatch_reward = np.mean(instant_mismatch_rewards[-100:]) if len(instant_mismatch_rewards) >= 100 else np.mean(instant_mismatch_rewards)
            print(f"🎯 步驟: {t + 1:>6}/{args.max_time_steps} | "
                  f"當前獎勵: {reward:>7.3f} | "
                  f"最大獎勵: {max_reward:>7.3f} | " 
                  f"平均獎勵: {current_avg_reward:>7.3f} | "
                  f"緩衝區: {replay_buffer.size:>5}")
            np.save(f"./Results/{save_path}/{file_name}", instant_mismatch_rewards)
            if args.save_model:
                agent.save(f"./Models/{save_path}/{file_name}")

        # 更新代理參數（只有當緩衝區有足夠樣本時）
        if replay_buffer.size > args.batch_size:
            if "Beta_Space" in args.policy:
                agent.update_parameters(replay_buffer, exp_regularization, args.batch_size)
            else:
                agent.update_parameters(replay_buffer, args.batch_size)

        # 如果啟用線性調度，更新探索正則化項
        if args.linear_schedule_exp_regularization:
            exp_regularization = args.exp_regularization_term - (args.exp_regularization_term * (t / args.max_time_steps))
    
    # 訓練完成總結
    print("=" * 80)
    print("🎉 訓練完成！")
    print("=" * 80)
    print(f"✅ 最終統計:")
    print(f"   最大獎勵: {max_reward:.4f}")
    print(f"   最大失配獎勵: {max_mismatch_reward:.4f}")
    print(f"   最終平均獎勵: {np.mean(instant_rewards[-100:]):.4f}")
    print(f"   訓練步數: {args.max_time_steps}")
    print(f"   經驗數量: {replay_buffer.size}")
    print(f"💾 結果已保存至: ./Results/{save_path}/{file_name}.npy")
    if args.save_model:
        print(f"💾 模型已保存至: ./Models/{save_path}/{file_name}")
    print("=" * 80)
