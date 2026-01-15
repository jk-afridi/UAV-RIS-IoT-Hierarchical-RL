import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal

from core.utils import soft_update, hard_update, weights_init_

"""
基於軟演員評論家(SAC)算法的Beta空間探索算法實現
這是一個專門針對RIS系統優化的強化學習算法
特點：
- 結合了SAC的優勢和針對RIS的特殊探索策略
- 考慮了相位依賴振幅(PDA)的影響
- 使用Beta空間來優化RIS的相位配置
"""

# 常量定義
LOG_SIG_MAX = 2         # 對數標準差的最大值
LOG_SIG_MIN = -20       # 對數標準差的最小值
epsilon = 1e-6          # 數值穩定性的小常數


class ValueNetwork(nn.Module):
    """
    狀態值函數網絡
    用於估計狀態的長期價值
    """
    def __init__(self, num_inputs, hidden_dim):
        """
        初始化值網絡
        
        參數:
            num_inputs (int): 輸入維度（狀態空間維度）
            hidden_dim (int): 隱藏層維度
        """
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # 應用權重初始化
        self.apply(weights_init_)

    def forward(self, state):
        """
        前向傳播計算狀態值
        
        參數:
            state: 輸入狀態
        返回:
            狀態值估計
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class Critic(nn.Module):
    """
    評論家網絡（Q函數）
    實現雙Q學習以減少過度估計
    """
    def __init__(self, num_inputs, num_actions, hidden_dim):
        """
        初始化評論家網絡
        
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


class GaussianPolicy(nn.Module):
    """
    高斯策略網絡（演員）
    輸出動作的均值和標準差，專門針對RIS系統設計
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, M, N, K, power_t, device, action_space=None):
        """
        初始化高斯策略網絡
        
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
        super(GaussianPolicy, self).__init__()

        self.device = device

        # 策略網絡架構
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, num_actions)     # 輸出動作均值
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)  # 輸出動作對數標準差

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
        計算動作分布的參數
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def compute_power(self, action):
        """
        計算並歸一化發射功率
        """
        G_real = action[:, :self.M ** 2].cpu().data.numpy()
        G_imag = action[:, self.M ** 2:2 * self.M ** 2].cpu().data.numpy()
        G = G_real.reshape(G_real.shape[0], self.M, self.K) + 1j * G_imag.reshape(G_imag.shape[0], self.M, self.K)
        GG_H = np.matmul(G, np.transpose(G.conj(), (0, 2, 1)))
        current_power_t = torch.sqrt(torch.from_numpy(np.real(np.trace(GG_H, axis1=1, axis2=2)))).reshape(-1, 1).to(self.device)
        return current_power_t

    def compute_phase(self, action):
        """
        計算RIS相位矩陣的歸一化係數
        """
        Phi_real = action[:, -2 * self.N:-self.N].detach()
        Phi_imag = action[:, -self.N:].detach()
        return (torch.sum(torch.abs(Phi_real), dim=1).reshape(-1, 1) * np.sqrt(2),
                torch.sum(torch.abs(Phi_imag), dim=1).reshape(-1, 1) * np.sqrt(2))

    def sample(self, state):
        """
        從策略分布中採樣動作
        
        參數:
            state: 輸入狀態
        返回:
            (採樣的動作, 對數概率, 均值動作)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        # 歸一化功率和相位
        current_power_t = self.compute_power(action.detach()).expand(-1, 2 * self.M ** 2) / np.sqrt(self.power_t)
        real_normal, imag_normal = self.compute_phase(action.detach())
        real_normal = real_normal.expand(-1, self.N)
        imag_normal = imag_normal.expand(-1, self.N)
        division_term = torch.cat([current_power_t, real_normal, imag_normal], dim=1)
        action /= division_term

        return action, log_prob, mean

    def to(self, device):
        """
        將模型移動到指定設備
        """
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class Explorer(nn.Module):
    """
    探索者網絡
    用於在Beta空間中進行探索
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, beta_min):
        """
        初始化探索者網絡
        
        參數:
            num_inputs (int): 輸入維度
            num_actions (int): 動作維度
            hidden_dim (int): 隱藏層維度
            beta_min (float): 最小beta值
        """
        super(Explorer, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.noise = nn.Linear(hidden_dim, num_actions)

        # 應用權重初始化
        self.apply(weights_init_)

        # 動作縮放設置
        self.action_scale = (1 - beta_min) / 2.
        self.action_bias = (1 + beta_min) / 2.

    def forward(self, state, exp_regularization):
        """
        生成探索噪聲
        
        參數:
            state: 輸入狀態
            exp_regularization: 探索正則化係數
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        noise = torch.tanh(self.noise(x)) * self.action_scale * exp_regularization + self.action_bias
        return noise

    def to(self, device):
        """
        將模型移動到指定設備
        """
        self.noise = self.noise.to(device)
        return super(Explorer, self).to(device)


class Beta_Space_Exp_SAC(object):
    """
    Beta空間探索SAC算法的主要實現類
    結合了SAC的優勢和Beta空間探索的特性
    """
    def __init__(self, state_dim,
                 action_space,
                 M,
                 N,
                 K,
                 power_t,
                 actor_lr,
                 critic_lr,
                 policy_type,
                 alpha,
                 target_update_interval,
                 automatic_entropy_tuning,
                 device,
                 beta_min=0.6,
                 discount=0.99,
                 tau=0.001):
        """
        初始化Beta空間探索SAC算法
        
        參數:
            state_dim: 狀態空間維度
            action_space: 動作空間
            M: 基站天線數
            N: RIS元件數
            K: 用戶數
            power_t: 傳輸功率約束(dBm)
            actor_lr: 演員網絡學習率
            critic_lr: 評論家網絡學習率
            policy_type: 策略類型
            alpha: 熵正則化係數
            target_update_interval: 目標網絡更新間隔
            automatic_entropy_tuning: 是否自動調整熵係數
            device: 計算設備
            beta_min: 最小beta值
            discount: 折扣因子
            tau: 軟更新係數
        """
        # 將功率從dBm轉換為線性單位
        power_t = 10 ** (power_t / 10)

        # 根據狀態維度確定隱藏層大小
        hidden_size = 1 if state_dim == 0 else 2 ** (state_dim - 1).bit_length()

        # 初始化訓練參數
        self.gamma = discount
        self.tau = tau
        self.alpha = alpha
        self.policy_type = policy_type
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.device = device
        self.updates = 0

        # 初始化網絡
        self.critic = Critic(state_dim, action_space.shape[0], hidden_size).to(device=self.device)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.critic_target = Critic(state_dim, action_space.shape[0], hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        # 設置自動熵調整
        if self.automatic_entropy_tuning is True:
            # 修復action_space.shape的類型問題
            action_dim = action_space.shape[0] if hasattr(action_space, 'shape') else len(action_space.high)
            self.target_entropy = -float(action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=actor_lr)

        # 初始化演員網絡
        self.actor = GaussianPolicy(state_dim, action_space.shape[0], hidden_size, M, N, K, power_t, self.device).to(self.device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)

        # 初始化探索者網絡
        self.explorer = Explorer(state_dim, N, hidden_size, beta_min).to(self.device)
        self.explorer_optimizer = Adam(self.explorer.parameters(), lr=actor_lr)
        self.explorer_target = Explorer(state_dim, N, hidden_size, beta_min).to(self.device)
        hard_update(self.explorer_target, self.explorer)

    def select_action(self, state, exp_regularization, evaluate=False):
        """
        根據當前狀態選擇動作
        
        參數:
            state: 當前狀態
            exp_regularization: 探索正則化係數
            evaluate: 是否處於評估模式
            
        返回:
            選擇的動作和beta值
        """
        self.actor.eval()  # 設置為評估模式

        state = torch.FloatTensor(state).to(self.device)

        if evaluate is False:
            # 訓練模式：使用探索者網絡生成beta值
            beta = self.explorer(state, exp_regularization)
            action, _, _ = self.actor.sample(state)
            # 將beta值應用到相位相關的動作分量
            action[:, -2 * beta.shape[1]:] *= beta.repeat(1, 2)
        else:
            # 評估模式：直接使用策略網絡的均值動作
            _, _, action = self.actor.sample(state)

        return action.detach().cpu().numpy(), beta.detach().cpu().numpy()

    def update_parameters(self, memory, exp_regularization, batch_size=16):
        """
        更新網絡參數
        
        參數:
            memory: 經驗回放緩衝區
            exp_regularization: 探索正則化係數
            batch_size: 批量大小
        """
        self.actor.train()  # 設置為訓練模式

        # 從經驗回放緩衝區採樣
        state_batch, action_batch, beta_batch, next_state_batch, reward_batch, mask_batch = memory.sample(batch_size=batch_size)

        with torch.no_grad():
            # 為下一個狀態選擇目標平滑化動作
            next_beta = self.explorer_target(next_state_batch, exp_regularization)
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state_batch)
            next_state_action[:, -2 * next_beta.shape[1]:] *= next_beta.repeat(1, 2)

            # 計算目標Q值
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target

        # 計算當前Q值
        qf1, qf2 = self.critic(state_batch, action_batch)

        # 計算評論家損失
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # 更新評論家網絡
        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        with torch.no_grad():
            beta = self.explorer(state_batch, exp_regularization)

        # 計算策略損失
        pi, log_pi, _ = self.actor.sample(state_batch)
        pi[:, -2 * beta.shape[1]:] *= beta.repeat(1, 2)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        # 更新演員網絡
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # 計算探索者損失
        beta = self.explorer(state_batch, exp_regularization)
        action_batch[:, -2 * beta.shape[1]:] *= (beta.repeat(1, 2) / beta_batch.repeat(1, 2))

        qf1, qf2 = self.critic(state_batch, action_batch)
        explorer_loss = -(F.mse_loss(qf1, next_q_value) + F.mse_loss(qf2, next_q_value))

        # 更新探索者網絡
        self.explorer_optimizer.zero_grad()
        explorer_loss.backward()
        self.explorer_optimizer.step()

        # 如果啟用了自動熵調整，更新熵係數
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

        self.updates += 1

        # 定期軟更新目標評論家網絡
        if self.updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

    def save(self, file_name):
        """
        保存模型參數
        
        參數:
            file_name: 文件名前綴
        """
        torch.save(self.actor.state_dict(), file_name + "_actor")
        torch.save(self.actor_optimizer.state_dict(), file_name + "_actor_optimizer")
        torch.save(self.critic.state_dict(), file_name + "_critic")
        torch.save(self.critic_optimizer.state_dict(), file_name + "_critic_optimizer")
        torch.save(self.explorer.state_dict(), file_name + "_explorer")
        torch.save(self.explorer_optimizer.state_dict(), file_name + "_explorer_optimizer")

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
        self.critic_target.load_state_dict(self.critic.state_dict())
