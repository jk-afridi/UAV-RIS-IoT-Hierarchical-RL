import numpy as np
import torch
from torch import nn

def get_device():
    """
    獲取全局統一的計算設備
    優先使用GPU，確保所有組件使用相同設備
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")

def soft_update(target, source, tau):
    """
    執行目標網絡的軟更新
    
    使用公式: θ_target = (1 - τ) * θ_target + τ * θ_source
    
    參數:
        target: 目標網絡
        source: 源網絡
        tau: 軟更新係數 (0 < τ < 1)
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    """
    執行目標網絡的硬更新（完全複製）
    
    直接將源網絡的參數複製到目標網絡
    
    參數:
        target: 目標網絡
        source: 源網絡
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def weights_init_(m):
    """
    初始化網絡權重
    
    使用Xavier均勻初始化線性層的權重，偏置初始化為0
    
    參數:
        m: 神經網絡模塊
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ExperienceReplayBuffer(object):
    """
    經驗回放緩衝區
    用於存儲和採樣訓練數據的緩衝區
    """
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        """
        初始化經驗回放緩衝區
        
        參數:
            state_dim (int): 狀態空間維度
            action_dim (int): 動作空間維度
            max_size (int): 緩衝區最大容量
        """
        self.max_size = max_size
        self.ptr = 0           # 指向下一個要寫入的位置
        self.size = 0          # 當前存儲的轉換數量

        # 初始化存儲空間
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        # 設置計算設備（使用統一設備管理）
        self.device = get_device()

    def add(self, state, action, next_state, reward, done):
        """
        添加一個轉換到緩衝區
        
        參數:
            state: 當前狀態
            action: 執行的動作
            next_state: 下一個狀態
            reward: 獲得的獎勵
            done: 是否是終止狀態
        """
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        # 更新指針和大小
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """
        從緩衝區隨機採樣一個批次
        
        參數:
            batch_size (int): 批次大小
            
        返回:
            tuple: (狀態, 動作, 下一狀態, 獎勵, 非終止標誌) 的張量批次
        """
        # 確保參數為整數類型
        batch_size = int(batch_size)
        size = int(self.size)
        
        # 隨機選擇索引
        index = np.random.randint(0, size, size=batch_size)

        # 返回轉換為張量的數據批次
        return (
            torch.FloatTensor(self.state[index]).to(self.device),
            torch.FloatTensor(self.action[index]).to(self.device),
            torch.FloatTensor(self.next_state[index]).to(self.device),
            torch.FloatTensor(self.reward[index]).to(self.device),
            torch.FloatTensor(self.not_done[index]).to(self.device)
        )


class BetaExperienceReplayBuffer(object):
    """
    Beta空間探索的經驗回放緩衝區
    擴展了標準經驗回放緩衝區以包含beta值
    """
    def __init__(self, state_dim, action_dim, N, max_size=int(1e6)):
        """
        初始化Beta經驗回放緩衝區
        
        參數:
            state_dim (int): 狀態空間維度
            action_dim (int): 動作空間維度
            N (int): RIS元件數量
            max_size (int): 緩衝區最大容量
        """
        self.max_size = max_size
        self.ptr = 0           # 指向下一個要寫入的位置
        self.size = 0          # 當前存儲的轉換數量

        # 初始化存儲空間
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.beta = np.zeros((max_size, N))  # 額外存儲beta值
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        # 設置計算設備（使用統一設備管理）
        self.device = get_device()

    def add(self, state, action, beta, next_state, reward, done):
        """
        添加一個帶有beta值的轉換到緩衝區
        
        參數:
            state: 當前狀態
            action: 執行的動作
            beta: RIS元件的beta值
            next_state: 下一個狀態
            reward: 獲得的獎勵
            done: 是否是終止狀態
        """
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.beta[self.ptr] = beta
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        # 更新指針和大小
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """
        從緩衝區隨機採樣一個批次
        
        參數:
            batch_size (int): 批次大小
            
        返回:
            tuple: (狀態, 動作, beta值, 下一狀態, 獎勵, 非終止標誌) 的張量批次
        """
        # 確保參數為整數類型
        batch_size = int(batch_size)
        size = int(self.size)
        
        # 隨機選擇索引
        index = np.random.randint(0, size, size=batch_size)

        # 返回轉換為張量的數據批次
        return (
            torch.FloatTensor(self.state[index]).to(self.device),
            torch.FloatTensor(self.action[index]).to(self.device),
            torch.FloatTensor(self.beta[index]).to(self.device),
            torch.FloatTensor(self.next_state[index]).to(self.device),
            torch.FloatTensor(self.reward[index]).to(self.device),
            torch.FloatTensor(self.not_done[index]).to(self.device)
        )

def whiten(state):
    """
    對狀態進行白化處理
    將數據標準化為均值為0，標準差為1的分布
    
    參數:
        state: 輸入狀態
        
    返回:
        標準化後的狀態
    """
    import numpy as np
    if np.std(state) == 0:
        return state - np.mean(state)
    return (state - np.mean(state)) / np.std(state)