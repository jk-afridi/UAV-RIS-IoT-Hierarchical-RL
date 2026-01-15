#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智慧城市RIS環境
包含實際的物理位置約束和服務範圍限制
"""

import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from .environment import RIS_MISO_PDA

class SmartCityRISEnvironment(RIS_MISO_PDA):
    """
    智慧城市中的RIS環境
    
    環境設定：
    - 100x100米的城市區域
    - 基地臺位於中央(50, 50)，高度15米
    - RIS高度15米，服務範圍限制60度
    - 10個RIS位置，每個服務4個用戶
    """
    
    def __init__(self, num_antennas=4,
                 num_RIS_elements=16,
                 num_users=4,
                 mismatch=False,
                 channel_est_error=False,
                 cascaded_channels=False,
                 beta_min=0.6,
                 theta_bar=0.0,
                 kappa_bar=1.5,
                 AWGN_var=1e-2,
                 channel_noise_var=1e-2,
                 seed=0,
                 city_size=100,
                 bs_height=15,
                 ris_height=15,
                 max_service_angle=60):
        """
        初始化智慧城市RIS環境
        
        參數:
            city_size (float): 城市大小 (米)
            bs_height (float): 基地臺高度 (米)
            ris_height (float): RIS高度 (米)
            max_service_angle (float): 最大服務角度 (度)
        """
        
        # 城市環境參數
        self.city_size = city_size
        self.bs_height = bs_height
        self.ris_height = ris_height
        self.max_service_angle = np.radians(max_service_angle)  # 轉換為弧度
        
        # 基地臺位置 (城市中央)
        self.bs_position = np.array([city_size/2, city_size/2, bs_height])
        
        # 初始化RIS和用戶位置
        self.num_ris_groups = 10  # 10個RIS群組
        self.users_per_ris = 4    # 每個RIS服務4個用戶
        self.total_users = self.num_ris_groups * self.users_per_ris  # 總共40個用戶
        
        # 生成RIS位置和對應的用戶位置
        self._generate_ris_positions()
        self._generate_user_positions()
        
        # 當前服務的RIS群組ID
        self.current_ris_group = 0
        
        # 調用父類初始化
        super().__init__(
            num_antennas=num_antennas,
            num_RIS_elements=num_RIS_elements,
            num_users=num_users,  # 每次只服務4個用戶
            mismatch=mismatch,
            channel_est_error=channel_est_error,
            cascaded_channels=cascaded_channels,
            beta_min=beta_min,
            theta_bar=theta_bar,
            kappa_bar=kappa_bar,
            AWGN_var=AWGN_var,
            channel_noise_var=channel_noise_var,
            seed=seed
        )
        
    def _generate_ris_positions(self):
        """生成RIS位置"""
        np.random.seed(42)  # 固定種子以保證可重現性
        
        # 在城市中隨機分布RIS位置，避免太靠近邊界
        margin = 10  # 邊界裕度
        
        self.ris_positions = []
        for i in range(self.num_ris_groups):
            x = np.random.uniform(margin, self.city_size - margin)
            y = np.random.uniform(margin, self.city_size - margin)
            z = self.ris_height
            self.ris_positions.append(np.array([x, y, z]))
        
        self.ris_positions = np.array(self.ris_positions)
        
    def _generate_user_positions(self):
        """根據RIS位置和服務範圍生成用戶位置"""
        self.user_positions = []
        self.user_ris_mapping = {}  # 用戶到RIS的映射關係
        
        for ris_idx, ris_pos in enumerate(self.ris_positions):
            ris_users = []
            
            # 計算RIS的服務範圍
            max_service_radius = self.ris_height / np.tan(self.max_service_angle)
            
            for user_idx in range(self.users_per_ris):
                # 在服務範圍內隨機生成用戶位置
                attempts = 0
                while attempts < 100:  # 最多嘗試100次
                    # 在圓形範圍內隨機選擇位置
                    angle = np.random.uniform(0, 2*np.pi)
                    radius = np.random.uniform(1, max_service_radius * 0.9)  # 留些裕度
                    
                    user_x = ris_pos[0] + radius * np.cos(angle)
                    user_y = ris_pos[1] + radius * np.sin(angle)
                    user_z = 1.5  # 用戶高度1.5米
                    
                    # 檢查是否在城市範圍內
                    if (0 <= user_x <= self.city_size and 
                        0 <= user_y <= self.city_size):
                        
                        user_pos = np.array([user_x, user_y, user_z])
                        
                        # 驗證服務角度
                        if self._check_service_angle(ris_pos, user_pos):
                            ris_users.append(user_pos)
                            global_user_idx = ris_idx * self.users_per_ris + user_idx
                            self.user_ris_mapping[global_user_idx] = ris_idx
                            break
                    
                    attempts += 1
                
                if attempts >= 100:
                    # 如果無法找到合適位置，使用默認位置
                    print(f"警告：無法為RIS {ris_idx} 的用戶 {user_idx} 找到合適位置")
                    user_pos = np.array([ris_pos[0] + 5, ris_pos[1] + 5, 1.5])
                    ris_users.append(user_pos)
                    global_user_idx = ris_idx * self.users_per_ris + user_idx
                    self.user_ris_mapping[global_user_idx] = ris_idx
            
            self.user_positions.extend(ris_users)
        
        self.user_positions = np.array(self.user_positions)
        
    def _check_service_angle(self, ris_pos, user_pos):
        """檢查RIS到用戶的角度是否在服務範圍內"""
        # 計算水平距離
        horizontal_dist = np.sqrt((ris_pos[0] - user_pos[0])**2 + 
                                 (ris_pos[1] - user_pos[1])**2)
        
        # 計算垂直距離
        vertical_dist = ris_pos[2] - user_pos[2]
        
        # 計算角度
        if horizontal_dist == 0:
            angle = np.pi/2
        else:
            angle = np.arctan(vertical_dist / horizontal_dist)
        
        return angle <= self.max_service_angle
        
    def _compute_channel_with_distance(self, pos1, pos2):
        """根據距離計算通道矩陣"""
        distance = np.linalg.norm(pos1 - pos2)
        
        # 路徑損失模型 (自由空間路徑損失)
        frequency = 2.4e9  # 2.4 GHz
        c = 3e8  # 光速
        path_loss_db = 20 * np.log10(distance) + 20 * np.log10(frequency) + 20 * np.log10(4*np.pi/c)
        path_loss_linear = 10**(-path_loss_db/20)
        
        # 生成瑞利衰落通道
        channel_real = np.random.normal(0, np.sqrt(0.5), (self.L, self.M)) * path_loss_linear
        channel_imag = np.random.normal(0, np.sqrt(0.5), (self.L, self.M)) * path_loss_linear
        
        return channel_real + 1j * channel_imag
        
    def set_current_ris_group(self, ris_group_id):
        """設置當前服務的RIS群組"""
        if 0 <= ris_group_id < self.num_ris_groups:
            self.current_ris_group = ris_group_id
        else:
            raise ValueError(f"RIS群組ID必須在0到{self.num_ris_groups-1}之間")
            
    def get_current_users(self):
        """獲取當前RIS群組服務的用戶"""
        start_idx = self.current_ris_group * self.users_per_ris
        end_idx = start_idx + self.users_per_ris
        return self.user_positions[start_idx:end_idx]
        
    def reset(self):
        """重置環境"""
        self.episode_t = 0
        self.info["true reward"] = 0
        
        # 獲取當前RIS位置
        current_ris_pos = self.ris_positions[self.current_ris_group]
        current_users = self.get_current_users()
        
        # 根據物理位置計算通道矩陣
        # H1: 基地臺到RIS的通道
        self.H_1 = self._compute_channel_with_distance(self.bs_position, current_ris_pos)
        
        # H2: RIS到用戶的通道
        h2_channels = []
        for user_pos in current_users:
            h2_user = self._compute_channel_with_distance(current_ris_pos, user_pos)
            h2_channels.append(h2_user[:, 0])  # 每個用戶只有一根天線
        
        self.H_2 = np.column_stack(h2_channels)
        
        # 初始化其他參數
        self.G = np.eye(self.M, dtype=complex)
        self.Phi = np.eye(self.L, dtype=complex)
        self.Phi_mismatch = np.eye(self.L, dtype=complex)
        
        # 構建初始狀態
        init_action_G = np.hstack((np.real(self.G.reshape(1, -1)), 
                                  np.imag(self.G.reshape(1, -1))))
        init_action_Phi = np.hstack((np.real(np.diag(self.Phi)).reshape(1, -1), 
                                   np.imag(np.diag(self.Phi)).reshape(1, -1)))
        init_action = np.hstack((init_action_G, init_action_Phi))
        
        # 計算功率和通道信息
        power_t = np.real(np.diag(self.G.conjugate().T @ self.G)).reshape(1, -1) ** 2
        
        if self.cascaded_channels:
            D = self._compute_D()
            H_2_tilde = self._compute_H_2_tilde(D)
            power_r = np.linalg.norm(H_2_tilde, axis=0).reshape(1, -1) ** 2
            D_real, D_imag = np.real(D).reshape(1, -1), np.imag(D).reshape(1, -1)
            self.state = np.hstack((init_action, power_t, power_r, D_real, D_imag))
        else:
            H_2_tilde = self.H_2.T @ self.Phi @ self.H_1 @ self.G
            power_r = np.linalg.norm(H_2_tilde, axis=0).reshape(1, -1) ** 2
            H_1_real, H_1_imag = np.real(self.H_1).reshape(1, -1), np.imag(self.H_1).reshape(1, -1)
            H_2_real, H_2_imag = np.real(self.H_2).reshape(1, -1), np.imag(self.H_2).reshape(1, -1)
            self.state = np.hstack((init_action, power_t, power_r, H_1_real, H_1_imag, H_2_real, H_2_imag))
        
        return self.state
        
    def visualize_environment(self, save_path=None):
        """可視化環境佈局"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # 設置中文字體
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 繪製城市邊界
        ax.add_patch(plt.Rectangle((0, 0), self.city_size, self.city_size, 
                                  fill=False, edgecolor='black', linewidth=2))
        
        # 繪製基地臺
        ax.scatter(self.bs_position[0], self.bs_position[1], 
                  c='red', marker='^', s=200, label=f'基地臺 (高度: {self.bs_height}m)')
        
        # 繪製RIS位置和服務範圍
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_ris_groups))
        
        for i, (ris_pos, color) in enumerate(zip(self.ris_positions, colors)):
            # 繪製RIS位置
            ax.scatter(ris_pos[0], ris_pos[1], c=[color], marker='s', s=150, 
                      label=f'RIS {i+1}' if i < 5 else '')  # 只顯示前5個標籤避免擁擠
            
            # 繪製服務範圍圓圈
            max_radius = self.ris_height / np.tan(self.max_service_angle)
            circle = plt.Circle((ris_pos[0], ris_pos[1]), max_radius, 
                              fill=False, color=color, alpha=0.3, linestyle='--')
            ax.add_patch(circle)
            
            # 繪製對應的用戶
            start_idx = i * self.users_per_ris
            end_idx = start_idx + self.users_per_ris
            user_group = self.user_positions[start_idx:end_idx]
            
            ax.scatter(user_group[:, 0], user_group[:, 1], 
                      c=[color], marker='o', s=50, alpha=0.7)
            
            # 繪製RIS到用戶的連線
            for user_pos in user_group:
                ax.plot([ris_pos[0], user_pos[0]], [ris_pos[1], user_pos[1]], 
                       color=color, alpha=0.3, linewidth=1)
        
        # 設置圖表屬性
        ax.set_xlim(-5, self.city_size + 5)
        ax.set_ylim(-5, self.city_size + 5)
        ax.set_xlabel('X座標 (公尺)', fontsize=12)
        ax.set_ylabel('Y座標 (公尺)', fontsize=12)
        ax.set_title('智慧城市RIS佈局\n' + 
                    f'城市大小: {self.city_size}×{self.city_size}m, ' +
                    f'RIS數量: {self.num_ris_groups}, ' +
                    f'總用戶數: {self.total_users}', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"環境佈局圖已保存至: {save_path}")
        
        plt.show()
        
        return fig, ax
    
    def get_environment_info(self):
        """獲取環境信息"""
        info = {
            'city_size': self.city_size,
            'bs_position': self.bs_position.tolist(),
            'bs_height': self.bs_height,
            'ris_height': self.ris_height,
            'max_service_angle_deg': np.degrees(self.max_service_angle),
            'num_ris_groups': self.num_ris_groups,
            'users_per_ris': self.users_per_ris,
            'total_users': self.total_users,
            'ris_positions': self.ris_positions.tolist(),
            'user_positions': self.user_positions.tolist(),
            'user_ris_mapping': self.user_ris_mapping
        }
        return info 
