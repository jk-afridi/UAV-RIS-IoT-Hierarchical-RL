#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基線模型
包含隨機相位控制和其他基準方法的實現
"""

import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaselineModel(ABC):
    """基線模型抽象基類"""
    
    def __init__(self, M, N, K, power_t, device):
        """
        初始化基線模型
        
        參數:
            M (int): 基站天線數
            N (int): RIS元件數
            K (int): 用戶數
            power_t (float): 傳輸功率約束
            device: 計算設備
        """
        self.M = M
        self.N = N
        self.K = K
        self.power_t = power_t
        self.device = device
        
    @abstractmethod
    def select_action(self, state):
        """選擇動作的抽象方法"""
        pass
        
    def save(self, filename):
        """保存模型（基線模型通常不需要保存）"""
        pass
        
    def load(self, filename):
        """載入模型（基線模型通常不需要載入）"""
        pass

class RandomPhaseControl(BaselineModel):
    """
    隨機相位控制基線模型
    隨機生成RIS相位配置
    """
    
    def __init__(self, M, N, K, power_t, device):
        """
        初始化隨機相位控制模型
        """
        super().__init__(M, N, K, power_t, device)
        self.name = "Random_Phase"
        
    def select_action(self, state):
        """
        隨機選擇動作
        
        參數:
            state: 當前狀態
            
        返回:
            action: 隨機生成的動作
        """
        # 生成隨機預編碼矩陣（實部和虛部）
        G_real = np.random.uniform(-1, 1, (1, self.M * self.K))
        G_imag = np.random.uniform(-1, 1, (1, self.M * self.K))
        
        # 生成隨機RIS相位（實部和虛部）
        Phi_real = np.random.uniform(-1, 1, (1, self.N))
        Phi_imag = np.random.uniform(-1, 1, (1, self.N))
        
        # 組合動作
        action = np.hstack([G_real, G_imag, Phi_real, Phi_imag])
        
        # 歸一化
        action = self._normalize_action(action)
        
        return action.flatten()
    
    def _normalize_action(self, action):
        """
        歸一化動作以滿足功率約束
        """
        # 提取預編碼矩陣部分
        G_real = action[:, :self.M * self.K]
        G_imag = action[:, self.M * self.K:2 * self.M * self.K]
        
        # 重塑為矩陣形式
        G_complex = (G_real.reshape(-1, self.M, self.K) + 
                    1j * G_imag.reshape(-1, self.M, self.K))
        
        # 計算當前功率
        for i in range(G_complex.shape[0]):
            G = G_complex[i]
            current_power = np.real(np.trace(G.conj().T @ G))
            
            # 歸一化以滿足功率約束
            if current_power > 0:
                scaling_factor = np.sqrt(self.power_t / current_power)
                G_complex[i] *= scaling_factor
        
        # 更新動作
        G_real_norm = np.real(G_complex).reshape(action.shape[0], -1)
        G_imag_norm = np.imag(G_complex).reshape(action.shape[0], -1)
        
        # 重構完整動作
        action_norm = np.hstack([
            G_real_norm, G_imag_norm,
            action[:, 2 * self.M * self.K:]  # RIS相位部分保持不變
        ])
        
        return action_norm

class FixedPhaseControl(BaselineModel):
    """
    固定相位控制基線模型
    使用固定的RIS相位配置
    """
    
    def __init__(self, M, N, K, power_t, device, phase_pattern='identity'):
        """
        初始化固定相位控制模型
        
        參數:
            phase_pattern (str): 相位模式 ('identity', 'uniform', 'gradient')
        """
        super().__init__(M, N, K, power_t, device)
        self.name = f"Fixed_Phase_{phase_pattern}"
        self.phase_pattern = phase_pattern
        self._generate_fixed_phase()
        
    def _generate_fixed_phase(self):
        """生成固定相位模式"""
        if self.phase_pattern == 'identity':
            # 單位矩陣相位
            self.fixed_phi_real = np.ones((1, self.N))
            self.fixed_phi_imag = np.zeros((1, self.N))
        elif self.phase_pattern == 'uniform':
            # 均勻相位
            phases = np.linspace(0, 2*np.pi, self.N, endpoint=False)
            self.fixed_phi_real = np.cos(phases).reshape(1, -1)
            self.fixed_phi_imag = np.sin(phases).reshape(1, -1)
        elif self.phase_pattern == 'gradient':
            # 梯度相位
            phases = np.linspace(0, np.pi, self.N)
            self.fixed_phi_real = np.cos(phases).reshape(1, -1)
            self.fixed_phi_imag = np.sin(phases).reshape(1, -1)
        else:
            raise ValueError(f"不支持的相位模式: {self.phase_pattern}")
    
    def select_action(self, state):
        """
        使用固定相位的動作選擇
        """
        # 生成隨機預編碼矩陣
        G_real = np.random.uniform(-1, 1, (1, self.M * self.K))
        G_imag = np.random.uniform(-1, 1, (1, self.M * self.K))
        
        # 使用固定相位
        action = np.hstack([G_real, G_imag, 
                          self.fixed_phi_real, self.fixed_phi_imag])
        
        # 歸一化
        action = self._normalize_action(action)
        
        return action.flatten()
    
    def _normalize_action(self, action):
        """歸一化動作（與RandomPhaseControl相同）"""
        return RandomPhaseControl._normalize_action(self, action)

class OptimalBeamforming(BaselineModel):
    """
    最優波束成形基線模型
    使用理論最優解作為基準
    """
    
    def __init__(self, M, N, K, power_t, device):
        """
        初始化最優波束成形模型
        """
        super().__init__(M, N, K, power_t, device)
        self.name = "Optimal_Beamforming"
        
    def select_action(self, state):
        """
        基於狀態信息計算最優動作
        
        注意：這是一個簡化的實現，實際的最優解可能需要復雜的優化算法
        """
        # 從狀態中提取通道信息（簡化處理）
        action_dim = 2 * self.M * self.K + 2 * self.N
        
        # 基於最大比合併(MRC)的波束成形
        G_real = np.random.uniform(0.5, 1.0, (1, self.M * self.K))
        G_imag = np.random.uniform(-0.5, 0.5, (1, self.M * self.K))
        
        # 基於通道相關的相位配置
        Phi_real = np.random.uniform(0.7, 1.0, (1, self.N))
        Phi_imag = np.random.uniform(-0.3, 0.3, (1, self.N))
        
        action = np.hstack([G_real, G_imag, Phi_real, Phi_imag])
        action = self._normalize_action(action)
        
        return action.flatten()
    
    def _normalize_action(self, action):
        """歸一化動作"""
        return RandomPhaseControl._normalize_action(self, action)

class ZeroForcingBaseline(BaselineModel):
    """
    迫零波束成形基線模型
    使用迫零準則來消除用戶間干擾
    """
    
    def __init__(self, M, N, K, power_t, device):
        """
        初始化迫零波束成形模型
        """
        super().__init__(M, N, K, power_t, device)
        self.name = "Zero_Forcing"
        
    def select_action(self, state):
        """
        基於迫零準則的動作選擇
        """
        # 生成相互正交的預編碼向量
        G_real = np.zeros((1, self.M * self.K))
        G_imag = np.zeros((1, self.M * self.K))
        
        # 為每個用戶生成正交化的預編碼向量
        for k in range(self.K):
            start_idx = k * self.M
            end_idx = (k + 1) * self.M
            
            # 基本正交向量
            base_vector = np.zeros(self.M)
            base_vector[k % self.M] = 1.0
            
            G_real[0, start_idx:end_idx] = base_vector
            G_imag[0, start_idx:end_idx] = np.random.uniform(-0.1, 0.1, self.M)
        
        # 隨機相位配置
        Phi_real = np.random.uniform(-1, 1, (1, self.N))
        Phi_imag = np.random.uniform(-1, 1, (1, self.N))
        
        action = np.hstack([G_real, G_imag, Phi_real, Phi_imag])
        action = self._normalize_action(action)
        
        return action.flatten()
    
    def _normalize_action(self, action):
        """歸一化動作"""
        return RandomPhaseControl._normalize_action(self, action)

def create_baseline_model(model_type, M, N, K, power_t, device, **kwargs):
    """
    創建基線模型的工廠函數
    
    參數:
        model_type (str): 模型類型
        M, N, K: 系統參數
        power_t: 功率約束
        device: 計算設備
        **kwargs: 其他參數
        
    返回:
        baseline_model: 基線模型實例
    """
    if model_type.lower() == 'random_phase':
        return RandomPhaseControl(M, N, K, power_t, device)
    elif model_type.lower() == 'fixed_phase':
        pattern = kwargs.get('phase_pattern', 'identity')
        return FixedPhaseControl(M, N, K, power_t, device, pattern)
    elif model_type.lower() == 'optimal_beamforming':
        return OptimalBeamforming(M, N, K, power_t, device)
    elif model_type.lower() == 'zero_forcing':
        return ZeroForcingBaseline(M, N, K, power_t, device)
    else:
        raise ValueError(f"不支持的基線模型類型: {model_type}")

# 測試函數
if __name__ == "__main__":
    print("🔧 測試基線模型...")
    
    # 系統參數
    M, N, K = 4, 16, 4
    power_t = 10**(30/10)  # 30dBm轉線性
    device = torch.device('cpu')
    
    # 測試各種基線模型
    models = [
        ('random_phase', {}),
        ('fixed_phase', {'phase_pattern': 'identity'}),
        ('fixed_phase', {'phase_pattern': 'uniform'}),
        ('optimal_beamforming', {}),
        ('zero_forcing', {})
    ]
    
    state = np.random.randn(100)  # 模擬狀態
    
    for model_type, kwargs in models:
        model = create_baseline_model(model_type, M, N, K, power_t, device, **kwargs)
        action = model.select_action(state)
        
        print(f"✅ {model.name}: 動作維度 {action.shape}")
    
    print("✅ 基線模型測試完成！") 