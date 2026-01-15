#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal

from core.utils import weights_init_

"""
近端策略優化算法(Proximal Policy Optimization, PPO)的實現
參考論文: https://arxiv.org/abs/1707.06347

PPO是一種在策略強化學習算法
特點：
- 使用Clipped Surrogate Objective防止策略更新過大
- 採用廣義優勢估計(GAE)來減少方差
- On-policy方法，每次更新後丟棄舊數據
- 通過熵獎勵促進探索
"""

class PPOActor(nn.Module):
    """
    PPO的Actor網絡（高斯策略）
    輸出動作分布的參數，專門針對RIS系統設計
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, M, N, K, power_t, device, action_space=None):
        """
        初始化PPO Actor網絡
        
        參數:
            num_inputs (int): 狀態空間維度
            num_actions (int): 動作空間維度
            hidden_dim (int): 隱藏層維度
            M (int): 基站天線數
            N (int): RIS元件數
            K (int): 用戶數
            power_t (float): 傳輸功率約束
            device: 計算設備(CPU/GPU)
            action_space: 動作空間範圍
        """
        super(PPOActor, self).__init__()

        self.device = device

        # Actor網絡架構
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, num_actions)

        # 學習標準差參數（獨立於狀態）
        self.log_std = nn.Parameter(torch.zeros(num_actions))

        # 應用權重初始化
        self.apply(weights_init_)

        # 系統參數
        self.M = M  # 基站天線數
        self.N = N  # RIS元件數
        self.K = K  # 用戶數
        self.power_t = power_t  # 傳輸功率約束

        # 動作縮放設置
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, state):
        """
        前向傳播，計算動作分布參數
        
        參數:
            state: 輸入狀態
        返回:
            (均值, 標準差)
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        std = torch.exp(self.log_std)
        return mean, std

    def compute_power(self, action):
        """
        計算發射功率
        
        參數:
            action: 動作
        返回:
            歸一化的發射功率
        """
        # 提取預編碼矩陣的實部和虛部
        G_real = action[:, :self.M ** 2].cpu().data.numpy()
        G_imag = action[:, self.M ** 2:2 * self.M ** 2].cpu().data.numpy()

        # 構建複數預編碼矩陣
        G = G_real.reshape(G_real.shape[0], self.M, self.K) + 1j * G_imag.reshape(G_imag.shape[0], self.M, self.K)

        # 計算功率
        GG_H = np.matmul(G, np.transpose(G.conj(), (0, 2, 1)))
        current_power_t = torch.sqrt(torch.from_numpy(np.real(np.trace(GG_H, axis1=1, axis2=2)))).reshape(-1, 1).to(self.device)

        return current_power_t

    def compute_phase(self, action):
        """
        計算RIS相位矩陣的歸一化係數
        
        參數:
            action: 動作
        返回:
            (實部歸一化係數, 虛部歸一化係數)
        """
        # 提取RIS相位矩陣的實部和虛部
        Phi_real = action[:, -2 * self.N:-self.N].detach()
        Phi_imag = action[:, -self.N:].detach()

        # 計算歸一化係數
        return (torch.sum(torch.abs(Phi_real), dim=1).reshape(-1, 1) * np.sqrt(2), 
                torch.sum(torch.abs(Phi_imag), dim=1).reshape(-1, 1) * np.sqrt(2))

    def _normalize_action(self, action):
        """
        歸一化動作以滿足RIS系統約束（復用SAC的約束處理機制）
        
        參數:
            action: 原始動作
        返回:
            normalized_action: 歸一化後的動作
        """
        # 歸一化發射功率和相位
        current_power_t = self.compute_power(action.detach()).expand(-1, 2 * self.M ** 2) / np.sqrt(self.power_t)
        real_normal, imag_normal = self.compute_phase(action.detach())

        real_normal = real_normal.expand(-1, self.N)
        imag_normal = imag_normal.expand(-1, self.N)

        # 組合歸一化係數並應用
        division_term = torch.cat([current_power_t, real_normal, imag_normal], dim=1)
        normalized_action = action / division_term

        return normalized_action

    def get_action_and_value(self, state, value_net, action=None):
        """
        獲取動作和價值（用於PPO訓練）
        
        參數:
            state: 狀態
            value_net: 價值網絡
            action: 可選的特定動作（用於計算對數概率）
        返回:
            action, log_prob, entropy, value
        """
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        
        if action is None:
            # 採樣動作
            raw_action = dist.sample()
        else:
            raw_action = action
            
        # 計算對數概率
        log_prob = dist.log_prob(raw_action).sum(axis=-1)
        
        # 計算熵
        entropy = dist.entropy().sum(axis=-1)
        
        # 使用tanh激活並縮放
        action_tanh = torch.tanh(raw_action)
        action_scaled = action_tanh * self.action_scale + self.action_bias
        
        # 應用約束歸一化
        action_normalized = self._normalize_action(action_scaled)
        
        # 獲取價值估計
        value = value_net(state)
        
        return action_normalized, log_prob, entropy, value.squeeze()

    def to(self, device):
        """
        將模型移動到指定設備（CPU/GPU）
        """
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(PPOActor, self).to(device)


class PPOCritic(nn.Module):
    """
    PPO的Critic網絡（價值函數）
    估計狀態價值
    """
    def __init__(self, num_inputs, hidden_dim):
        """
        初始化Critic網絡
        
        參數:
            num_inputs (int): 狀態空間維度
            hidden_dim (int): 隱藏層維度
        """
        super(PPOCritic, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # 應用權重初始化
        self.apply(weights_init_)

    def forward(self, state):
        """
        前向傳播計算狀態價值
        
        參數:
            state: 輸入狀態
        返回:
            狀態價值估計
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        value = self.linear3(x)
        return value


class PPOBuffer:
    """
    PPO的經驗緩衝區
    用於收集軌跡數據並計算GAE
    """
    def __init__(self, size, state_dim, action_dim, device, gamma=0.99, gae_lambda=0.95):
        self.size = size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # 初始化緩衝區
        self.states = torch.zeros((size, state_dim), dtype=torch.float32).to(device)
        self.actions = torch.zeros((size, action_dim), dtype=torch.float32).to(device)
        self.rewards = torch.zeros(size, dtype=torch.float32).to(device)
        self.values = torch.zeros(size, dtype=torch.float32).to(device)
        self.log_probs = torch.zeros(size, dtype=torch.float32).to(device)
        self.dones = torch.zeros(size, dtype=torch.float32).to(device)
        
        self.ptr = 0
        self.full = False

    def store(self, state, action, reward, value, log_prob, done):
        """存儲一步經驗"""
        self.states[self.ptr] = torch.FloatTensor(state).to(self.device)
        self.actions[self.ptr] = torch.FloatTensor(action).to(self.device)
        self.rewards[self.ptr] = torch.tensor(reward, dtype=torch.float32).to(self.device)
        self.values[self.ptr] = torch.tensor(value, dtype=torch.float32).to(self.device)
        self.log_probs[self.ptr] = torch.tensor(log_prob, dtype=torch.float32).to(self.device)
        self.dones[self.ptr] = torch.tensor(done, dtype=torch.float32).to(self.device)
        
        self.ptr = (self.ptr + 1) % self.size
        if self.ptr == 0:
            self.full = True

    def finish_path(self, last_value=0):
        """
        計算GAE和returns
        
        參數:
            last_value: 軌跡結束時的價值估計
        """
        path_slice = slice(0, self.ptr)
        
        # 計算rewards-to-go和GAE
        rewards = self.rewards[path_slice]
        values = self.values[path_slice]
        dones = self.dones[path_slice]
        
        # 添加最後一個價值
        values_with_last = torch.cat([values, torch.tensor([last_value]).to(self.device)])
        
        # 計算advantage和returns
        deltas = rewards + self.gamma * values_with_last[1:] * (1 - dones) - values_with_last[:-1]
        
        advantages = torch.zeros_like(rewards)
        advantage = 0
        for t in reversed(range(len(rewards))):
            advantage = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * advantage
            advantages[t] = advantage
            
        returns = advantages + values
        
        return advantages, returns

    def get(self):
        """
        獲取緩衝區中的所有數據
        
        返回:
            states, actions, advantages, returns, log_probs
        """
        assert self.ptr > 0, "Buffer is empty!"
        
        path_slice = slice(0, self.ptr)
        advantages, returns = self.finish_path()
        
        # 標準化advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        data = dict(
            states=self.states[path_slice],
            actions=self.actions[path_slice],
            advantages=advantages,
            returns=returns,
            log_probs=self.log_probs[path_slice]
        )
        
        # 清空緩衝區
        self.ptr = 0
        
        return data


class PPO(object):
    """
    近端策略優化（PPO）算法的主要實現類
    """
    def __init__(self, state_dim,
                 action_space,
                 M,
                 N,
                 K,
                 power_t,
                 actor_lr,
                 critic_lr,
                 device,
                 n_steps=2048,
                 n_epochs=10,
                 clip_range=0.2,
                 ent_coef=0.01,
                 vf_coef=0.5,
                 gamma=0.99,
                 gae_lambda=0.95):
        """
        初始化PPO算法
        
        參數:
            state_dim: 狀態空間維度
            action_space: 動作空間
            M: 基站天線數
            N: RIS元件數
            K: 用戶數
            power_t: 傳輸功率約束
            actor_lr: Actor網絡學習率
            critic_lr: Critic網絡學習率
            device: 計算設備
            n_steps: 每次更新前收集的步數
            n_epochs: 每次更新的輪數
            clip_range: PPO clipping範圍
            ent_coef: 熵係數
            vf_coef: 價值函數損失係數
            gamma: 折扣因子
            gae_lambda: GAE的lambda參數
        """
        # 將功率從dBm轉換為線性單位
        power_t = 10 ** (power_t / 10)

        # 根據狀態維度確定隱藏層大小
        hidden_size = 1 if state_dim == 0 else 2 ** (state_dim - 1).bit_length()

        # 訓練參數
        self.device = device
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # 初始化網絡
        self.actor = PPOActor(state_dim, action_space.shape[0], hidden_size, M, N, K, power_t, self.device, action_space).to(self.device)
        self.critic = PPOCritic(state_dim, hidden_size).to(self.device)

        # 初始化優化器
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

        # 初始化緩衝區
        self.buffer = PPOBuffer(n_steps, state_dim, action_space.shape[0], device, gamma, gae_lambda)

        # 訓練統計
        self.updates = 0

    def select_action(self, state, evaluate=False):
        """
        根據當前狀態選擇動作
        
        參數:
            state: 當前狀態
            evaluate: 是否處於評估模式
        返回:
            action, log_prob, value
        """
        state = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            action, log_prob, entropy, value = self.actor.get_action_and_value(state, self.critic)
        
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()

    def store_transition(self, state, action, reward, log_prob, value, done):
        """
        存儲轉換到緩衝區
        
        參數:
            state: 狀態
            action: 動作
            reward: 獎勵
            log_prob: 對數概率
            value: 價值估計
            done: 是否結束
        """
        self.buffer.store(state, action, reward, value, log_prob, done)

    def update_parameters(self):
        """
        更新PPO參數
        """
        # 獲取緩衝區數據
        data = self.buffer.get()
        
        states = data['states']
        actions = data['actions']
        advantages = data['advantages']
        returns = data['returns']
        old_log_probs = data['log_probs']

        # 多輪更新
        for _ in range(self.n_epochs):
            # 重新計算動作的對數概率、熵和價值
            _, new_log_probs, entropy, new_values = self.actor.get_action_and_value(states, self.critic, actions)

            # 計算比率
            ratio = torch.exp(new_log_probs - old_log_probs)

            # 計算PPO損失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # 計算價值函數損失
            critic_loss = F.mse_loss(new_values, returns)

            # 計算熵損失
            entropy_loss = -entropy.mean()

            # 總損失
            total_loss = actor_loss + self.vf_coef * critic_loss + self.ent_coef * entropy_loss

            # 更新網絡
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        self.updates += 1

    def ready_to_update(self):
        """
        檢查是否準備好更新（緩衝區是否已滿）
        """
        return self.buffer.ptr >= self.n_steps

    def save(self, filename):
        """
        保存模型參數
        
        參數:
            filename: 文件名前綴
        """
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

    def load(self, filename):
        """
        載入模型參數
        
        參數:
            filename: 文件名前綴
        """
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer")) 
