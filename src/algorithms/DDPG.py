#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from core.utils import soft_update, hard_update, weights_init_

"""
深度確定性策略梯度算法(Deep Deterministic Policy Gradient, DDPG)的實現
參考論文: https://arxiv.org/abs/1509.02971

DDPG是一種用於連續動作空間的異策略強化學習算法
特點：
- 使用確定性策略而非隨機策略
- 採用Actor-Critic架構
- 使用經驗回放和目標網絡提高訓練穩定性
- 通過探索噪聲實現exploration
"""

class Actor(nn.Module):
    """
    DDPG的Actor網絡（確定性策略）
    輸出確定性動作，專門針對RIS系統設計
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, M, N, K, power_t, device, action_space=None):
        """
        初始化Actor網絡
        
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
        super(Actor, self).__init__()

        self.device = device

        # Actor網絡架構
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_actions)

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
        前向傳播，輸出確定性動作
        
        參數:
            state: 輸入狀態
        返回:
            action: 確定性動作
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        # 使用tanh激活函數將輸出限制在[-1,1]範圍
        action = torch.tanh(self.linear3(x))
        
        # 縮放動作到目標範圍
        action = action * self.action_scale + self.action_bias
        
        # 應用RIS系統約束歸一化（復用SAC的機制）
        action = self._normalize_action(action)
        
        return action

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

    def to(self, device):
        """
        將模型移動到指定設備（CPU/GPU）
        """
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(Actor, self).to(device)


class Critic(nn.Module):
    """
    DDPG的Critic網絡（Q函數）
    實現雙Q學習以減少過度估計
    """
    def __init__(self, num_inputs, num_actions, hidden_dim):
        """
        初始化Critic網絡
        
        參數:
            num_inputs (int): 狀態空間維度
            num_actions (int): 動作空間維度
            hidden_dim (int): 隱藏層維度
        """
        super(Critic, self).__init__()

        # Q1網絡架構
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2網絡架構
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        # 應用權重初始化
        self.apply(weights_init_)

    def forward(self, state, action):
        """
        計算兩個Q值
        
        參數:
            state: 狀態
            action: 動作
        返回:
            (Q1值, Q2值)
        """
        xu = torch.cat([state, action], 1)  # 將狀態和動作連接

        # Q1的前向傳播
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        # Q2的前向傳播
        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

    def Q1(self, state, action):
        """
        只計算Q1值（用於Actor更新）
        
        參數:
            state: 狀態
            action: 動作
        返回:
            Q1值
        """
        xu = torch.cat([state, action], 1)
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        return x1


class DDPG(object):
    """
    深度確定性策略梯度（DDPG）算法的主要實現類
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
                 discount=0.99,
                 tau=0.001,
                 noise_std=0.3,
                 noise_decay=True):
        """
        初始化DDPG算法
        
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
            discount: 折扣因子
            tau: 軟更新係數
            noise_std: 探索噪聲標準差
            noise_decay: 是否使用噪聲衰減
        """
        # 將功率從dBm轉換為線性單位
        power_t = 10 ** (power_t / 10)

        # 根據狀態維度確定隱藏層大小
        hidden_size = 1 if state_dim == 0 else 2 ** (state_dim - 1).bit_length()

        # 初始化訓練參數
        self.gamma = discount
        self.tau = tau
        self.device = device
        self.updates = 0

        # 探索噪聲參數
        self.noise_std = noise_std
        self.initial_noise_std = noise_std
        self.noise_decay = noise_decay

        # 初始化Actor網絡和優化器
        self.actor = Actor(state_dim, action_space.shape[0], hidden_size, M, N, K, power_t, self.device, action_space).to(self.device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.actor_target = Actor(state_dim, action_space.shape[0], hidden_size, M, N, K, power_t, self.device, action_space).to(self.device)

        # 初始化Critic網絡和優化器
        self.critic = Critic(state_dim, action_space.shape[0], hidden_size).to(device=self.device)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.critic_target = Critic(state_dim, action_space.shape[0], hidden_size).to(self.device)

        # 複製參數到目標網絡
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def select_action(self, state, evaluate=False, timestep=0, max_timesteps=20000):
        """
        根據當前狀態選擇動作
        
        參數:
            state: 當前狀態
            evaluate: 是否處於評估模式
            timestep: 當前時間步
            max_timesteps: 最大時間步數
        返回:
            選擇的動作
        """
        self.actor.eval()  # 設置為評估模式

        # 將狀態轉換為tensor並移動到指定設備
        state = torch.FloatTensor(state).to(self.device)

        # 獲取確定性動作
        with torch.no_grad():
            action = self.actor(state)

        if not evaluate:
            # 訓練模式：添加探索噪聲
            if self.noise_decay:
                # 線性衰減噪聲
                current_noise_std = self.initial_noise_std * (1 - timestep / max_timesteps)
                current_noise_std = max(current_noise_std, 0.05)  # 最小噪聲
            else:
                current_noise_std = self.noise_std

            # 添加高斯噪聲
            noise = torch.normal(0, current_noise_std, size=action.shape).to(self.device)
            action = action + noise

            # 確保動作在合理範圍內
            action = torch.clamp(action, -5, 5)

        return action.detach().cpu().numpy()

    def update_parameters(self, memory, batch_size=16):
        """
        更新DDPG參數
        
        參數:
            memory: 經驗回放緩衝區
            batch_size: 批量大小
        """
        self.actor.train()  # 設置為訓練模式
        self.critic.train()

        # 從經驗回放緩衝區採樣
        state_batch, action_batch, next_state_batch, reward_batch, mask_batch = memory.sample(batch_size=batch_size)

        with torch.no_grad():
            # 計算目標動作和目標Q值
            next_action = self.actor_target(next_state_batch)
            target_Q1, target_Q2 = self.critic_target(next_state_batch, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward_batch + (mask_batch * self.gamma * target_Q)

        # 獲取當前Q值估計
        current_Q1, current_Q2 = self.critic(state_batch, action_batch)

        # 計算Critic損失
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # 更新Critic網絡
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 計算Actor損失
        actor_loss = -self.critic.Q1(state_batch, self.actor(state_batch)).mean()

        # 更新Actor網絡
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 軟更新目標網絡
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        self.updates += 1

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
        
        # 同步目標網絡
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic) 
