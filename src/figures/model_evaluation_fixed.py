#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修復版RIS模型評估腳本 (新JSON格式適配版)
解決模型加載、基線模型調用、用戶位置生成等問題
適配new_test.py生成的新JSON格式
"""

import os
import torch
import numpy as np
import argparse
import json
import glob
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# 導入依賴模塊
import gym
from gym.envs.registration import register
from algorithms import SAC
from algorithms import Beta_Space_Exp_SAC
from core.smart_city_environment import SmartCityRISEnvironment
from algorithms.baseline_models import create_baseline_model
from figures.visualization import RISVisualization
from core.utils import whiten

# 註冊環境
try:
    register(
        id='RIS_MISO_PDA-v0',
        entry_point='core.environment:RIS_MISO_PDA',
    )
except:
    pass

def find_latest_environment_data():
    """尋找最新的環境數據檔案 - 適配新的JSON格式"""
    # 新的檔案路徑格式
    search_patterns = [
        "./outputs/dbscan/enhanced_ris_environments/environment_data_*.json",
        "./outputs/dbscan/fixed_ris_environments/fixed_environment_data_*.json"  # 向後兼容
    ]
    
    all_files = []
    for pattern in search_patterns:
        files = glob.glob(pattern)
        all_files.extend(files)
    
    if not all_files:
        print("❌ 未找到環境數據檔案")
        print("💡 請先運行 new_test.py 生成環境數據")
        return None
    
    latest_file = max(all_files, key=os.path.getctime)
    print(f"🔍 找到最新環境數據: {os.path.basename(latest_file)}")
    return latest_file

def load_environment_data(file_path):
    """載入環境數據 - 適配新的JSON格式"""
    print(f"📥 載入環境數據: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 檢查JSON格式並適配
    if "deployment" in data:
        # 新格式 (new_test.py)
        print("✅ 檢測到新格式環境數據")
        return {
            'ris_positions': data['deployment']['ris_positions'],
            'devices': data['deployment']['devices'],
            'ris_device_mapping': {int(k): v for k, v in data['deployment']['ris_device_mapping'].items()},
            'parameters': data['parameters'],
            'buildings': data['environment']['buildings'],
            'los_grid': data['environment']['los_grid'] if 'los_grid' in data['environment'] else None
        }
    else:
        # 舊格式 (原有格式)
        print("✅ 檢測到舊格式環境數據，進行適配")
        return data

class FixedModelEvaluator:
    """修復版模型評估器 - 適配新JSON格式"""
    
    def __init__(self, env_config, environment_data=None, device='cpu'):
        """
        初始化模型評估器
        
        參數:
            env_config: 環境配置
            environment_data: 環境數據 (新增)
            device: 計算設備
        """
        self.env_config = env_config
        self.environment_data = environment_data
        self.device = torch.device(device)
        self.env = SmartCityRISEnvironment(**env_config)
        self.visualizer = RISVisualization()
        
        # 如果有環境數據，設置RIS相關信息
        if environment_data:
            self.setup_ris_environment()
        
        # 結果存儲
        self.results = {}
        
    def setup_ris_environment(self):
        """設置RIS環境信息"""
        if not self.environment_data:
            return
            
        self.ris_positions = self.environment_data.get('ris_positions', [])
        self.devices = self.environment_data.get('devices', [])
        self.ris_device_mapping = self.environment_data.get('ris_device_mapping', {})
        
        print(f"🔧 環境設置完成:")
        print(f"   📍 RIS位置: {len(self.ris_positions)} 個")
        print(f"   📱 裝置總數: {len(self.devices)} 個")
        print(f"   🔗 RIS服務組: {len(self.ris_device_mapping)} 組")
        
        # 更新環境的RIS組數
        if hasattr(self.env, 'num_ris_groups'):
            self.env.num_ris_groups = len(self.ris_positions)
    
    def load_trained_model(self, model_type, model_path, L_value, beta_min):
        """
        載入訓練好的模型
        
        參數:
            model_type: 模型類型 ('SAC' 或 'Beta_Space_Exp_SAC')
            model_path: 模型檔案路徑
            L_value: RIS元素數量
            beta_min: beta最小值
            
        返回:
            loaded_model: 載入的模型
        """
        try:
            # 創建正確的模型配置
            model_config = self._create_clean_model_config(L_value, beta_min)
            
            if model_type == 'SAC':
                model = SAC.SAC(**model_config)
            elif model_type == 'Beta_Space_Exp_SAC':
                model = Beta_Space_Exp_SAC.Beta_Space_Exp_SAC(**model_config)
            else:
                raise ValueError(f"不支持的模型類型: {model_type}")
            
            # 載入模型參數
            if os.path.exists(model_path + "_actor"):
                model.load(model_path)
                print(f"✅ 成功載入模型: {os.path.basename(model_path)}")
                return model
            else:
                print(f"⚠️ 模型檔案不存在: {model_path}")
                return None
                
        except Exception as e:
            print(f"❌ 載入模型失敗: {e}")
            return None
    
    def _create_clean_model_config(self, L_value, beta_min):
        """創建乾淨的模型配置，只包含SAC需要的參數"""
        
        # 計算狀態和動作維度
        env_temp = SmartCityRISEnvironment(
            num_antennas=4,
            num_RIS_elements=L_value,
            num_users=4,
            beta_min=beta_min
        )
        
        # 創建動作空間
        action_space = gym.spaces.Box(
            low=-5, high=5, 
            shape=(2 * 4 * 4 + 2 * L_value,), 
            dtype=np.float32
        )
        
        # 只保留SAC需要的參數
        model_config = {
            'state_dim': env_temp.state_dim,
            'action_space': action_space,
            'M': 4,  # 基站天線數
            'N': L_value,  # RIS元素數（對應num_RIS_elements）
            'K': 4,  # 用戶數
            'power_t': 30.0,
            'actor_lr': 1e-3,
            'critic_lr': 1e-3,
            'policy_type': "Gaussian",
            'alpha': 0.2,
            'target_update_interval': 1,
            'automatic_entropy_tuning': True,
            'device': self.device
        }
        
        return model_config
    
    def evaluate_single_model(self, model, model_name, num_episodes=50):
        """
        評估單個模型在所有RIS位置的性能
        
        參數:
            model: 要評估的模型
            model_name: 模型名稱  
            num_episodes: 每個RIS位置的評估次數
            
        返回:
            evaluation_results: 評估結果
        """
        print(f"\n🎯 開始評估模型: {model_name}")
        
        all_ris_results = {}
        total_rewards = []
        
        # 確定RIS組數
        num_ris_groups = len(self.ris_positions) if self.ris_positions else getattr(self.env, 'num_ris_groups', 10)
        
        # 對每個RIS位置進行評估
        for ris_id in range(num_ris_groups):
            print(f"  📍 評估RIS {ris_id + 1}/{num_ris_groups}")
            
            # 設置當前RIS群組
            try:
                if hasattr(self.env, 'set_current_ris_group'):
                    self.env.set_current_ris_group(ris_id)
                elif hasattr(self.env, 'current_ris_group'):
                    self.env.current_ris_group = ris_id
            except Exception as e:
                print(f"  ⚠️ 設置RIS群組失敗: {e}")
                continue
            
            ris_rewards = []
            
            # 多次評估取平均
            for episode in range(num_episodes):
                try:
                    episode_reward = self._run_single_episode(model, model_name)
                    ris_rewards.append(episode_reward)
                except Exception as e:
                    print(f"    警告：第{episode+1}次評估失敗: {e}")
                    continue
            
            if ris_rewards:
                # 計算該RIS的平均性能
                avg_ris_reward = np.mean(ris_rewards)
                all_ris_results[ris_id] = avg_ris_reward
                total_rewards.extend(ris_rewards)
                print(f"    RIS {ris_id + 1} 平均獎勵: {avg_ris_reward:.3f}")
            else:
                print(f"    RIS {ris_id + 1} 評估失敗")
                all_ris_results[ris_id] = 0.0
        
        # 計算總體統計
        if total_rewards:
            evaluation_results = {
                'model_name': model_name,
                'ris_results': all_ris_results,
                'total_average': np.mean(total_rewards),
                'total_std': np.std(total_rewards),
                'total_max': np.max(total_rewards),
                'total_min': np.min(total_rewards),
                'num_episodes_per_ris': num_episodes,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"  📊 {model_name} 總體平均獎勵: {evaluation_results['total_average']:.3f}")
        else:
            evaluation_results = {
                'model_name': model_name,
                'ris_results': all_ris_results,
                'total_average': 0.0,
                'total_std': 0.0,
                'total_max': 0.0,
                'total_min': 0.0,
                'num_episodes_per_ris': num_episodes,
                'timestamp': datetime.now().isoformat()
            }
            print(f"  ❌ {model_name} 評估完全失敗")
        
        return evaluation_results
    
    def _run_single_episode(self, model, model_name):
        """
        運行單個評估回合
        
        參數:
            model: 模型實例
            model_name: 模型名稱
            
        返回:
            episode_reward: 回合總獎勵
        """
        try:
            # 重置環境
            state = self.env.reset()
            if state is None:
                raise ValueError("環境重置返回None")
            
            # 白化處理
            state = whiten(state)
            
            episode_reward = 0
            max_steps = 100  # 限制最大步數
            
            for step in range(max_steps):
                # 根據模型類型選擇動作
                if hasattr(model, 'select_action'):
                    # 訓練過的模型
                    if 'Random' in model_name or 'Baseline' in model_name:
                        # 基線模型 - 不傳evaluate參數
                        action = model.select_action(state)
                    else:
                        # 訓練過的RL模型
                        action = model.select_action(state)
                else:
                    # 如果沒有select_action方法，隨機選擇動作
                    action = self.env.action_space.sample()
                
                # 執行動作
                next_state, reward, done, info = self.env.step(action)
                
                # 使用真實獎勵（考慮PDA失配）
                if isinstance(info, dict) and "true reward" in info:
                    episode_reward += info["true reward"]
                else:
                    episode_reward += reward
                
                if done:
                    break
                
                state = whiten(next_state)
            
            return episode_reward
            
        except Exception as e:
            print(f"    單回合評估失敗: {e}")
            return 0.0
    
    def evaluate_baseline_models(self, baseline_types, L_value, num_episodes=50):
        """
        評估基線模型
        
        參數:
            baseline_types: 基線模型類型列表
            L_value: RIS元素數量
            num_episodes: 評估回合數
            
        返回:
            baseline_results: 基線模型結果
        """
        print(f"\n🔧 評估基線模型...")
        
        baseline_results = {}
        
        for baseline_type in baseline_types:
            try:
                # 創建基線模型
                baseline_model = create_baseline_model(
                    baseline_type, 
                    M=4, N=L_value, K=4, 
                    power_t=30.0, device=self.device
                )
                
                if baseline_model is None:
                    print(f"  ⚠️ 無法創建基線模型: {baseline_type}")
                    continue
                
                # 評估基線模型
                baseline_name = f"{baseline_type.title()}_Baseline"
                results = self.evaluate_single_model(
                    baseline_model, baseline_name, num_episodes
                )
                
                baseline_results[baseline_type] = results
                
            except Exception as e:
                print(f"  ❌ 基線模型 {baseline_type} 評估失敗: {e}")
                continue
        
        return baseline_results
    
    def save_results(self, results, save_dir):
        """保存評估結果"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 保存JSON格式結果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(save_dir, f'evaluation_results_{timestamp}.json')
        
        # 轉換numpy類型為Python原生類型
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        json_results[key][k] = v.tolist()
                    elif isinstance(v, (np.int64, np.float64)):
                        json_results[key][k] = v.item()
                    else:
                        json_results[key][k] = v
            else:
                json_results[key] = value
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"📁 結果已保存到: {json_path}")

def create_improved_env_config(L_value, beta_min, environment_data=None):
    """創建改進的環境配置，減少用戶位置生成失敗"""
    base_config = {
        'num_antennas': 4,
        'num_RIS_elements': L_value,
        'num_users': 4,
        'mismatch': False,
        'channel_est_error': True,
        'cascaded_channels': True,
        'beta_min': beta_min,
        'theta_bar': 0.0,
        'kappa_bar': 1.5,
        'AWGN_var': 1e-2,
        'channel_noise_var': 1e-2,
        'seed': 42,
        'city_size': 100,
        'bs_height': 15,
        'ris_height': 15,
        'max_service_angle': 70  # 增加服務角度減少位置生成失敗
    }
    
    # 如果有環境數據，使用其中的參數
    if environment_data and 'parameters' in environment_data:
        params = environment_data['parameters']
        base_config.update({
            'city_size': params.get('city_size', 100),
            'bs_height': params.get('bs_height', 15),
            'ris_height': params.get('ris_height', 15),
        })
    
    return base_config

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='修復版RIS模型評估 (新JSON格式適配版)')
    parser.add_argument('--models_dir', default='./Models', help='模型目錄')
    parser.add_argument('--results_dir', default='./outputs/evaluation/evaluation_results_fixed_new', help='結果保存目錄')
    parser.add_argument('--num_episodes', type=int, default=30, help='每個RIS位置的評估次數')
    parser.add_argument('--device', default='cpu', help='計算設備')
    parser.add_argument('--env_data', default=None, help='環境數據檔案路徑 (可選)')
    
    args = parser.parse_args()
    
    # 載入環境數據
    env_data_path = args.env_data if args.env_data else find_latest_environment_data()
    if not env_data_path:
        print("❌ 無法找到環境數據檔案，將使用默認配置")
        environment_data = None
    else:
        environment_data = load_environment_data(env_data_path)
    
    # 要評估的L值和對應參數
    evaluation_configs = [
        {'L': 16, 'beta_min': 0.3},
        {'L': 32, 'beta_min': 0.6},
        {'L': 64, 'beta_min': 0.6}
    ]
    
    # 要評估的方法
    methods = [
        {'name': 'SAC_golden', 'policy': 'SAC', 'type': 'SAC'},
        {'name': 'SAC_mismatch', 'policy': 'SAC', 'type': 'SAC'}, 
        {'name': 'Beta_Space_Exp_SAC_mismatch', 'policy': 'Beta_Space_Exp_SAC', 'type': 'Beta_Space_Exp_SAC'}
    ]
    
    # 基線模型類型
    baseline_types = ['random_phase']
    
    all_results = {}
    
    # 對每個L值進行評估
    for config in evaluation_configs:
        L = config['L']
        beta_min = config['beta_min']
        
        print(f"\n{'='*60}")
        print(f"評估 L = {L} 的模型 (新JSON格式適配版)")
        print(f"{'='*60}")
        
        # 創建環境配置
        env_config = create_improved_env_config(L, beta_min, environment_data)
        
        # 創建評估器
        evaluator = FixedModelEvaluator(env_config, environment_data, args.device)
        
        L_results = {}
        
        # 評估訓練過的模型
        for method in methods:
            method_name = method['name']
            method_type = method['type']
            
            # 構建模型路徑
            save_path = f"Beta_min. = {beta_min}, K = 4, M = 4, N = {L}, P_t = 30.0"
            model_path = os.path.join(args.models_dir, save_path, f"{method_name}_0")
            
            if os.path.exists(model_path + "_actor"):
                print(f"\n🚀 開始評估: {method_name}")
                
                # 載入並評估模型
                model = evaluator.load_trained_model(method_type, model_path, L, beta_min)
                
                if model is not None:
                    results = evaluator.evaluate_single_model(
                        model, method_name, args.num_episodes
                    )
                    L_results[method_name] = results
                else:
                    print(f"❌ {method_name} 模型載入失敗")
            else:
                print(f"⚠️ {method_name} 模型文件不存在: {model_path}")
        
        # 評估基線模型
        baseline_results = evaluator.evaluate_baseline_models(baseline_types, L, args.num_episodes)
        L_results.update(baseline_results)
        
        all_results[f'L_{L}'] = L_results
    
    # 保存所有結果
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    
    # 創建總結
    print(f"\n{'='*60}")
    print("📊 評估總結 (新JSON格式適配版)")
    print(f"{'='*60}")
    
    for L_key, L_results in all_results.items():
        print(f"\n{L_key} 結果:")
        for method_name, results in L_results.items():
            if isinstance(results, dict) and 'total_average' in results:
                print(f"  {method_name:.<35} {results['total_average']:.3f}")
            else:
                print(f"  {method_name:.<35} 評估失敗")
    
    # 保存結果
    evaluator = FixedModelEvaluator(create_improved_env_config(16, 0.3, environment_data), environment_data, args.device)
    evaluator.save_results(all_results, args.results_dir)
    
    print(f"\n✅ 所有評估完成！結果保存在: {args.results_dir}")
    print(f"🔧 環境數據來源: {env_data_path if env_data_path else '默認配置'}")

if __name__ == "__main__":
    main() 
