from ._gym_compat import gym, spaces
import numpy as np

class RIS_MISO_PDA(gym.Env):
    """
    RIS-aided MISO系統的環境類，實現相位依賴振幅(PDA)模型
    RIS: 智能反射表面 (Reconfigurable Intelligent Surface)
    MISO: 多輸入單輸出 (Multiple-Input Single-Output)
    PDA: 相位依賴振幅 (Phase-Dependent Amplitude)
    """
    def __init__(self, num_antennas,
                 num_RIS_elements,
                 num_users,
                 mismatch=False,
                 channel_est_error=False,
                 cascaded_channels=False,
                 beta_min=0.6,
                 theta_bar=0.0,
                 kappa_bar=1.5,
                 AWGN_var=1e-2,
                 channel_noise_var=1e-2,
                 seed=0):
        """
        初始化RIS-MISO-PDA環境
        
        參數:
            num_antennas (int): 基站天線數量
            num_RIS_elements (int): RIS元件數量
            num_users (int): 用戶數量
            mismatch (bool): 是否考慮PDA模型失配
            channel_est_error (bool): 是否考慮通道估計誤差
            cascaded_channels (bool): 是否使用級聯通道模型
            beta_min (float): PDA模型中的最小振幅值
            theta_bar (float): PDA模型中的相位偏移值
            kappa_bar (float): PDA模型中的形狀參數
            AWGN_var (float): 加性白高斯噪聲方差
            channel_noise_var (float): 通道噪聲方差
            seed (int): 隨機數種子
        """
        super(RIS_MISO_PDA, self).__init__()

        self._max_episode_steps = np.inf  # 設置最大步數為無限

        # 系統參數設置
        self.M = num_antennas            # 基站天線數量
        self.L = num_RIS_elements        # RIS元件數量
        self.K = num_users               # 用戶數量

        # 模型配置
        self.mismatch = mismatch                    # PDA模型失配
        self.channel_est_error = channel_est_error  # 通道估計誤差
        self.cascaded_channels = cascaded_channels  # 級聯通道模型
        self.beta_min = beta_min                    # 最小振幅值
        self.theta_bar = theta_bar                  # 相位偏移
        self.kappa_bar = kappa_bar                  # 形狀參數

        # 確保基站天線數等於用戶數
        assert self.M == self.K

        # 噪聲參數
        self.awgn_var = AWGN_var                # 加性白高斯噪聲方差
        self.channel_noise_var = channel_noise_var  # 通道噪聲方差

        # 設置動作和狀態空間維度
        power_size = 2 * self.K  # 功率相關維度
        self.action_dim = 2 * self.M * self.K + 2 * self.L  # 動作空間維度

        # 根據是否使用級聯通道模型設置通道大小
        if self.cascaded_channels:
            channel_size = 2 * self.K * self.L * self.M
        else:
            channel_size = 2 * (self.L * self.M + self.L * self.K)

        # 計算狀態空間維度
        self.state_dim = power_size + channel_size + self.action_dim

        # 定義觀察和動作空間
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=float)
        self.action_space = spaces.Box(low=-5, high=5, shape=(self.action_dim,), dtype=float)

        # 初始化通道和系統矩陣
        self.H_1 = None  # 基站到RIS的通道矩陣
        self.H_2 = None  # RIS到用戶的通道矩陣
        self.G = np.eye(self.M, dtype=complex)    # 預編碼矩陣
        self.Phi = np.eye(self.L, dtype=complex)  # RIS相位移相矩陣
        self.Phi_mismatch = np.eye(self.L, dtype=complex)  # PDA模型失配的相位移相矩陣

        # 初始化狀態和完成標誌
        self.state = None
        self.done = None
        self.episode_t = None
        self.info = {'episode': None, 'true reward': None}

        # 設置隨機種子
        self.seed(seed)

    def seed(self, seed):
        """設置隨機數種子"""
        np.random.seed(seed)

    def _compute_PDA(self, angles):
        """
        計算相位依賴振幅(PDA)值
        
        參數:
            angles: 相位角度
            
        返回:
            betas: 對應的振幅值
        """
        betas = (1 - self.beta_min) * ((np.sin(angles - self.theta_bar) + 1) / 2) ** self.kappa_bar + self.beta_min
        return betas

    def _compute_D(self):
        """
        計算級聯通道矩陣D
        
        返回:
            D: 級聯通道矩陣
        """
        # 計算初始D矩陣
        D = np.diag(self.H_2[:, 0]) @ self.H_1

        # 對每個用戶計算並堆疊通道矩陣
        for column_idx in np.arange(1, self.H_2.shape[1]):
            D_k = np.diag(self.H_2[:, column_idx]) @ self.H_1
            D = np.vstack((D, D_k))

        # 如果考慮通道估計誤差，添加高斯噪聲
        if self.channel_est_error:
            D += np.random.normal(0, np.sqrt(self.channel_noise_var / 2), D.shape) + \
                 1j * np.random.normal(0, np.sqrt(self.channel_noise_var / 2), D.shape)

        return D

    def _compute_H_2_tilde(self, D):
        """
        計算等效通道矩陣H_2_tilde
        
        參數:
            D: 級聯通道矩陣
            
        返回:
            H_2_tilde: 等效通道矩陣
        """
        if self.cascaded_channels:
            return np.diag(list(np.diag(self.Phi)) * self.K) @ D @ self.G
        else:
            return self.H_2.T @ self.Phi @ self.H_1 @ self.G

    def reset(self):
        """
        重置環境到初始狀態
        
        返回:
            state: 初始狀態
        """
        self.episode_t = 0
        self.info["true reward"] = 0

        # 隨機初始化通道矩陣
        self.H_1 = np.random.normal(0, np.sqrt(0.5), (self.L, self.M)) + \
                   1j * np.random.normal(0, np.sqrt(0.5), (self.L, self.M))
        self.H_2 = np.random.normal(0, np.sqrt(0.5), (self.L, self.K)) + \
                   1j * np.random.normal(0, np.sqrt(0.5), (self.L, self.K))

        # 初始化動作
        init_action_G = np.hstack((np.real(self.G.reshape(1, -1)), np.imag(self.G.reshape(1, -1))))
        init_action_Phi = np.hstack((np.real(np.diag(self.Phi)).reshape(1, -1), 
                                   np.imag(np.diag(self.Phi)).reshape(1, -1)))
        init_action = np.hstack((init_action_G, init_action_Phi))

        # 提取相位信息
        Phi_real = init_action[:, -2 * self.L:-self.L]
        Phi_imag = init_action[:, -self.L:]
        angles = np.arctan2(Phi_real, Phi_imag)
        betas = self._compute_PDA(angles)

        # 根據是否考慮失配設置Phi矩陣
        if self.mismatch:
            self.Phi = np.eye(self.L, dtype=complex) * (Phi_real + 1j * Phi_imag) * betas
        else:
            self.Phi = np.eye(self.L, dtype=complex) * (Phi_real + 1j * Phi_imag)
            self.Phi_mismatch = self.Phi * betas

        # 計算發射功率
        power_t = np.real(np.diag(self.G.conjugate().T @ self.G)).reshape(1, -1) ** 2

        # 計算通道和接收功率
        D = self._compute_D()
        H_2_tilde = self._compute_H_2_tilde(D)
        power_r = np.linalg.norm(H_2_tilde, axis=0).reshape(1, -1) ** 2

        # 根據是否使用級聯通道模型構建狀態
        if self.cascaded_channels:
            D_real, D_imag = np.real(D).reshape(1, -1), np.imag(D).reshape(1, -1)
            self.state = np.hstack((init_action, power_t, power_r, D_real, D_imag))
        else:
            H_1_real, H_1_imag = np.real(self.H_1).reshape(1, -1), np.imag(self.H_1).reshape(1, -1)
            H_2_real, H_2_imag = np.real(self.H_2).reshape(1, -1), np.imag(self.H_2).reshape(1, -1)
            self.state = np.hstack((init_action, power_t, power_r, H_1_real, H_1_imag, H_2_real, H_2_imag))

        return self.state

    def _compute_reward(self, Phi):
        """
        計算獎勵值
        
        參數:
            Phi: RIS相位移相矩陣
            
        返回:
            reward: 獎勵值
            opt_reward: 最優獎勵值
        """
        reward = 0
        opt_reward = 0

        # 對每個用戶計算獎勵
        for k in range(self.K):
            h_2_k = self.H_2[:, k].reshape(-1, 1)
            g_k = self.G[:, k].reshape(-1, 1)

            # 計算信號功率
            x = np.abs(h_2_k.T @ Phi @ self.H_1 @ g_k) ** 2
            x = x.item()

            # 計算干擾功率
            G_removed = np.delete(self.G, k, axis=1)
            interference = np.sum(np.abs(h_2_k.T @ Phi @ self.H_1 @ G_removed) ** 2)
            y = interference + (self.K - 1) * self.awgn_var

            # 計算SINR和獎勵
            rho_k = x / y
            reward += np.log(1 + rho_k) / np.log(2)
            opt_reward += np.log(1 + x / ((self.K - 1) * self.awgn_var)) / np.log(2)

        return reward, opt_reward

    def step(self, action, custom_betas=None):
        """
        執行一步環境交互
        
        參數:
            action: 代理選擇的動作
            custom_betas: 自定義的beta值（可選）
            
        返回:
            next_state: 下一個狀態
            reward: 獎勵值
            done: 是否結束
            info: 額外信息
        """
        self.episode_t += 1
        action = action.reshape(1, -1)

        # 提取動作中的預編碼矩陣和相位信息
        G_real = action[:, :self.M ** 2]
        G_imag = action[:, self.M ** 2:2 * self.M ** 2]
        Phi_real = action[:, -2 * self.L:-self.L]
        Phi_imag = action[:, -self.L:]

        # 更新預編碼矩陣
        self.G = G_real.reshape(self.M, self.K) + 1j * G_imag.reshape(self.M, self.K)

        # 計算相位角度和對應的beta值
        angles = np.arctan2(Phi_real, Phi_imag)
        betas = self._compute_PDA(angles)

        # 根據是否考慮失配更新Phi矩陣
        if self.mismatch:
            self.Phi = np.diag(np.exp(1j * angles).flatten() * betas.flatten())
        else:
            self.Phi = np.diag(np.exp(1j * angles).flatten())

            if custom_betas is not None:
                Phi_real_custom = action[:, -2 * self.L:-self.L] / custom_betas
                Phi_imag_custom = action[:, -self.L:] / custom_betas
                actual_angles = np.arctan2(Phi_real_custom, Phi_imag_custom)
                actual_betas = self._compute_PDA(angles)
                self.Phi_mismatch = np.eye(self.L, dtype=complex) * (Phi_real_custom + 1j * Phi_imag_custom) * actual_betas
            else:
                self.Phi_mismatch = self.Phi * betas

        # 計算發射功率
        power_t = np.real(np.diag(self.G.conjugate().T @ self.G)).reshape(1, -1) ** 2

        # 計算通道和等效通道矩陣
        D = self._compute_D()
        H_2_tilde = self._compute_H_2_tilde(D)

        # 計算接收功率
        power_r = np.linalg.norm(H_2_tilde, axis=0).reshape(1, -1) ** 2

        # 根據是否使用級聯通道模型構建狀態
        if self.cascaded_channels:
            # 使用級聯通道模型時的狀態構建
            D_real, D_imag = np.real(D).reshape(1, -1), np.imag(D).reshape(1, -1)
            self.state = np.hstack((action, power_t, power_r, D_real, D_imag))
        else:
            # 使用分離通道模型時的狀態構建
            H_1_real, H_1_imag = np.real(self.H_1).reshape(1, -1), np.imag(self.H_1).reshape(1, -1)
            H_2_real, H_2_imag = np.real(self.H_2).reshape(1, -1), np.imag(self.H_2).reshape(1, -1)
            self.state = np.hstack((action, power_t, power_r, H_1_real, H_1_imag, H_2_real, H_2_imag))

        # 計算當前配置下的獎勵
        reward, opt_reward = self._compute_reward(self.Phi)

        # 根據是否考慮失配計算真實獎勵
        if self.mismatch:
            true_reward = reward
        else:
            true_reward, opt_reward = self._compute_reward(self.Phi_mismatch)

        # 判斷是否達到終止條件
        done = opt_reward == reward or self.episode_t >= self._max_episode_steps

        # 更新信息字典
        self.info["true reward"] = true_reward

        return self.state, reward, done, self.info

    def close(self):
        """
        關閉環境
        這是一個空方法，因為該環境不需要特殊的清理操作
        """
        pass