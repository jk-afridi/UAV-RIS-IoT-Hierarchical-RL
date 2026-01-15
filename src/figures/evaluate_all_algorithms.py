#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整算法評估腳本
支持SAC、DDPG、PPO和Beta_Space_Exp_SAC的性能評估和比較
"""

import os
import numpy as np
import torch
import argparse
import json
from datetime import datetime
import matplotlib.pyplot as plt
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 確保能導入自定義模塊

from core.smart_city_environment import SmartCityRISEnvironment
from algorithms.baseline_models import create_baseline_model
from figures.visualization import RISVisualization

class CompleteModelEvaluator:
    """完整模型評估器，支持所有算法"""
    
    def __init__(self, env_config, environment_data=None, device='cpu'):
        self.env_config = env_config
        self.environment_data = environment_data
        self.device = device
        self.visualization = RISVisualization()
        
        print(f"🔧 評估器初始化完成，設備: {device}")
    
    def load_trained_model(self, algorithm_type, model_path, L_value, beta_min):
        """
        載入訓練好的模型
        
        參數:
            algorithm_type: 算法類型 ('SAC', 'DDPG', 'PPO', 'Beta_Space_Exp_SAC')
            model_path: 模型路徑
            L_value: RIS元素數量
            beta_min: 最小beta值
        """
        try:
            # 創建乾淨的模型配置
            model_config = self._create_model_config(L_value, beta_min)
            
            # 根據算法類型載入模型
            if algorithm_type == 'SAC':
                from algorithms.SAC import SAC
                model = SAC(**model_config)
            elif algorithm_type == 'DDPG':
                from algorithms.DDPG import DDPG
                model = DDPG(**model_config)
            elif algorithm_type == 'PPO':
                from algorithms.PPO import PPO
                model = PPO(**model_config)
            elif algorithm_type == 'Beta_Space_Exp_SAC':
                from algorithms.Beta_Space_Exp_SAC import Beta_Space_Exp_SAC
                model_config['beta_min'] = beta_min
                model = Beta_Space_Exp_SAC(**model_config)
            else:
                raise ValueError(f"不支持的算法類型: {algorithm_type}")
            
            # 載入模型參數
            model.load(model_path)
            print(f"✅ 成功載入 {algorithm_type} 模型")
            return model
            
        except Exception as e:
            print(f"❌ 載入 {algorithm_type} 模型失敗: {e}")
            return None
    
    def _create_model_config(self, L_value, beta_min):
        """創建模型配置 - 使用與訓練時一致的環境"""
        # 使用與train_new_algorithms.py一致的環境配置
        import gym
        from gym.envs.registration import register
        
        # 註冊環境（避免重複註冊錯誤）
        try:
            register(
                id='RIS_MISO_PDA-v0',
                entry_point='core.environment:RIS_MISO_PDA',
            )
        except:
            pass
        
        # 創建與訓練時相同的環境配置
        environment_kwargs = {
            "num_antennas": 4,
            "num_RIS_elements": L_value,
            "num_users": 4,
            "mismatch": True,  # mismatch場景
            "channel_est_error": True,
            "cascaded_channels": True,
            "beta_min": beta_min,
            "theta_bar": 0.0,
            "kappa_bar": 1.5,
            "AWGN_var": 1e-2,
            "channel_noise_var": 1e-2,
            "seed": 0,
        }
        
        # 創建環境來獲取正確的維度
        temp_env = gym.make('RIS_MISO_PDA-v0', **environment_kwargs)
        
        # 創建簡化的action_space對象而不是直接使用gym對象
        action_dim = 2 * 4 * 4 + 2 * L_value  # M*K*2 + L*2
        
        model_config = {
            'state_dim': temp_env.state_dim,
            'action_space': temp_env.action_space,
            'action_dim': action_dim,
            'M': 4,
            'N': L_value,
            'K': 4,
            'power_t': 30.0,
            'actor_lr': 1e-3,
            'critic_lr': 1e-3,
            'device': self.device
        }
        
        temp_env.close()
        
        # PPO特定配置
        if hasattr(self, '_is_ppo_config'):
            model_config.update({
                'n_steps': 2048,
                'n_epochs': 10,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'gamma': 1.0,
                'gae_lambda': 0.95
            })
        # DDPG特定配置
        elif hasattr(self, '_is_ddpg_config'):
            model_config.update({
                'discount': 1.0,
                'tau': 1e-3,
                'noise_std': 0.3,
                'noise_decay': True
            })
        # SAC特定配置
        else:
            model_config.update({
                'policy_type': "Gaussian",
                'alpha': 0.2,
                'target_update_interval': 1,
                'automatic_entropy_tuning': True,
                'discount': 1.0,
                'tau': 1e-3
            })
        
        return model_config
    
    def evaluate_single_model(self, model, algorithm_name, num_episodes=30):
        """
        評估單個模型
        
        參數:
            model: 訓練好的模型
            algorithm_name: 算法名稱
            num_episodes: 評估集數
        """
        print(f"🔍 評估 {algorithm_name}...")
        
        all_rewards = []
        ris_position_rewards = {}
        successful_episodes = 0
        
        # 創建評估環境 - 使用與訓練一致的環境
        import gym
        from gym.envs.registration import register
        
        # 註冊環境
        try:
            register(
                id='RIS_MISO_PDA-v0',
                entry_point='core.environment:RIS_MISO_PDA',
            )
        except:
            pass
        
        env = gym.make('RIS_MISO_PDA-v0', **self.env_config)
        
        for episode in range(num_episodes):
            try:
                # 使用基本的環境重置（不需要RIS位置設置）
                
                state = env.reset()
                episode_reward = 0
                done = False
                steps = 0
                max_steps = 100
                
                while not done and steps < max_steps:
                    # 根據算法類型選擇動作
                    if hasattr(model, 'select_action'):
                        if 'PPO' in algorithm_name:
                            action, _, _ = model.select_action(state, evaluate=True)
                        elif 'Beta_Space' in algorithm_name:
                            action, _ = model.select_action(state, exp_regularization=0, evaluate=True)
                        else:
                            action = model.select_action(state, evaluate=True)
                    else:
                        action = model.select_action(state)
                    
                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    state = next_state
                    steps += 1
                
                all_rewards.append(episode_reward)
                
                # 記錄每個RIS位置的獎勵（使用episode作為位置ID）
                ris_position_id = episode % 10  # 假設有10個位置
                if ris_position_id not in ris_position_rewards:
                    ris_position_rewards[ris_position_id] = []
                ris_position_rewards[ris_position_id].append(episode_reward)
                
                successful_episodes += 1
                
                if (episode + 1) % 10 == 0:
                    avg_reward = np.mean(all_rewards[-10:])
                    print(f"  Episode {episode + 1}/{num_episodes}, 最近10集平均獎勵: {avg_reward:.3f}")
                    
            except Exception as e:
                print(f"  ⚠️ Episode {episode + 1} 評估失敗: {e}")
                continue
        
        if successful_episodes == 0:
            print(f"❌ {algorithm_name} 評估完全失敗")
            return None
        
        # 計算統計數據
        results = {
            'algorithm': algorithm_name,
            'total_episodes': successful_episodes,
            'average': np.mean(all_rewards),
            'std': np.std(all_rewards),
            'min': np.min(all_rewards),
            'max': np.max(all_rewards),
            'median': np.median(all_rewards),
            'ris_position_performance': {}
        }
        
        # 計算每個RIS位置的平均性能
        for pos, rewards in ris_position_rewards.items():
            results['ris_position_performance'][pos] = {
                'average': np.mean(rewards),
                'count': len(rewards)
            }
        
        print(f"✅ {algorithm_name} 評估完成: {results['average']:.3f} ± {results['std']:.3f}")
        return results
    
    def evaluate_baseline_model_special(self, model, algorithm_name, num_episodes=30):
        """特殊評估基線模型（不使用evaluate參數）"""
        print(f"🔍 評估基線模型: {algorithm_name}...")
        
        all_rewards = []
        ris_position_rewards = {}
        successful_episodes = 0
        
        # 創建評估環境
        import gym
        from gym.envs.registration import register
        
        try:
            register(
                id='RIS_MISO_PDA-v0',
                entry_point='core.environment:RIS_MISO_PDA',
            )
        except:
            pass
        
        env = gym.make('RIS_MISO_PDA-v0', **self.env_config)
        
        for episode in range(num_episodes):
            try:
                state = env.reset()
                episode_reward = 0
                done = False
                steps = 0
                max_steps = 100
                
                while not done and steps < max_steps:
                    # 基線模型不使用evaluate參數
                    action = model.select_action(state)
                    
                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    state = next_state
                    steps += 1
                
                all_rewards.append(episode_reward)
                
                # 記錄每個RIS位置的獎勵
                ris_position_id = episode % 10
                if ris_position_id not in ris_position_rewards:
                    ris_position_rewards[ris_position_id] = []
                ris_position_rewards[ris_position_id].append(episode_reward)
                
                successful_episodes += 1
                
            except Exception as e:
                print(f"  ⚠️ Episode {episode + 1} 評估失敗: {e}")
                continue
        
        env.close()
        
        if successful_episodes == 0:
            print(f"❌ {algorithm_name} 評估完全失敗")
            return None
        
        # 計算統計結果
        avg_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        max_reward = np.max(all_rewards)
        min_reward = np.min(all_rewards)
        
        results = {
            'algorithm': algorithm_name,
            'episodes': successful_episodes,
            'average': avg_reward,
            'std': std_reward,
            'max': max_reward,
            'min': min_reward,
            'all_rewards': all_rewards,
            'ris_position_rewards': ris_position_rewards
        }
        
        print(f"✅ {algorithm_name} 評估完成: 平均獎勵 {avg_reward:.3f} ± {std_reward:.3f}")
        
        return results
    
    def evaluate_baseline_models(self, baseline_types, L_value, num_episodes=30):
        """評估基線模型"""
        print(f"🔍 評估基線模型...")
        
        baseline_results = {}
        
        for baseline_type in baseline_types:
            try:
                print(f"  評估基線: {baseline_type}")
                
                # 創建基線模型
                baseline_model = create_baseline_model(
                    baseline_type, M=4, N=L_value, K=4, 
                    power_t=30.0, device=self.device
                )
                
                # 評估基線模型（特殊處理）
                results = self.evaluate_baseline_model_special(baseline_model, baseline_type, num_episodes)
                
                if results:
                    baseline_results[baseline_type] = results
                    
            except Exception as e:
                print(f"❌ 基線 {baseline_type} 評估失敗: {e}")
        
        return baseline_results
    
    def compare_all_algorithms(self, trained_models, baseline_types, num_episodes=30):
        """
        比較所有算法
        
        參數:
            trained_models: [(algorithm_type, model_path, model_config, algorithm_name)]
            baseline_types: 基線算法類型列表
            num_episodes: 評估集數
        """
        print(f"🏁 開始完整算法比較...")
        
        all_results = {}
        
        # 評估訓練好的模型
        for algorithm_type, model_path, model_config, algorithm_name in trained_models:
            if os.path.exists(model_path + "_actor"):
                print(f"\n🚀 評估訓練模型: {algorithm_name}")
                
                # 設置算法特定標誌
                if algorithm_type == 'PPO':
                    self._is_ppo_config = True
                elif algorithm_type == 'DDPG':
                    self._is_ddpg_config = True
                
                model = self.load_trained_model(
                    algorithm_type, model_path, 
                    model_config.get('N', 16), 
                    model_config.get('beta_min', 0.6)
                )
                
                if model:
                    results = self.evaluate_single_model(model, algorithm_name, num_episodes)
                    if results:
                        all_results[algorithm_name] = results
                
                # 清除標誌
                if hasattr(self, '_is_ppo_config'):
                    delattr(self, '_is_ppo_config')
                if hasattr(self, '_is_ddpg_config'):
                    delattr(self, '_is_ddpg_config')
            else:
                print(f"⚠️ 模型文件不存在: {model_path}")
        
        # 評估基線模型
        baseline_results = self.evaluate_baseline_models(
            baseline_types, 
            self.env_config['num_RIS_elements'], 
            num_episodes
        )
        all_results.update(baseline_results)
        
        # 生成性能排名
        performance_ranking = sorted(
            [(name, {'average': result['average'], 'std': result['std']}) 
             for name, result in all_results.items()],
            key=lambda x: x[1]['average'], reverse=True
        )
        
        comparison_results = {
            'individual_results': all_results,
            'performance_ranking': performance_ranking,
            'evaluation_config': {
                'num_episodes': num_episodes,
                'L_value': self.env_config['num_RIS_elements'],
                'beta_min': self.env_config['beta_min'],
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return comparison_results
    
    def generate_visualizations(self, comparison_results, save_dir):
        """生成可視化圖表"""
        print(f"🎨 生成可視化圖表...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 性能比較條形圖
        self._plot_performance_comparison(comparison_results, save_dir)
        
        # 2. RIS位置性能熱力圖
        self._plot_ris_position_heatmap(comparison_results, save_dir)
        
        # 3. 詳細性能分析圖
        self._plot_detailed_analysis(comparison_results, save_dir)
    
    def _plot_performance_comparison(self, comparison_results, save_dir):
        """繪製性能比較圖"""
        try:
            performance_ranking = comparison_results['performance_ranking']
            
            algorithms = [item[0] for item in performance_ranking]
            averages = [item[1]['average'] for item in performance_ranking]
            stds = [item[1]['std'] for item in performance_ranking]
            
            # 算法顏色映射
            colors = {
                'SAC (Golden)': '#2ca02c',
                'SAC (Mismatch)': '#ff7f0e', 
                'Beta-Space SAC': '#1f77b4',
                'DDPG': '#d62728',
                'PPO': '#9467bd',
                'Random_Phase': '#8c564b'
            }
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            x_pos = np.arange(len(algorithms))
            bar_colors = [colors.get(alg, '#666666') for alg in algorithms]
            
            bars = ax.bar(x_pos, averages, yerr=stds, capsize=5, 
                         color=bar_colors, alpha=0.8, edgecolor='black')
            
            ax.set_xlabel('算法', fontsize=12)
            ax.set_ylabel('平均性能 (bps/Hz)', fontsize=12)
            ax.set_title(f'算法性能比較 (L = {comparison_results["evaluation_config"]["L_value"]})', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(algorithms, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            # 添加數值標籤
            for i, (bar, avg, std) in enumerate(zip(bars, averages, stds)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                       f'{avg:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            filename = f'performance_comparison_L{comparison_results["evaluation_config"]["L_value"]}.png'
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  💾 已保存: {filename}")
            
        except Exception as e:
            print(f"❌ 性能比較圖生成失敗: {e}")
    
    def _plot_ris_position_heatmap(self, comparison_results, save_dir):
        """繪製RIS位置性能熱力圖"""
        try:
            individual_results = comparison_results['individual_results']
            
            # 收集所有RIS位置和算法
            all_positions = set()
            algorithms = []
            
            for alg_name, result in individual_results.items():
                if 'ris_position_performance' in result:
                    algorithms.append(alg_name)
                    all_positions.update(result['ris_position_performance'].keys())
            
            if not all_positions or not algorithms:
                print("  ⚠️ 沒有足夠的RIS位置數據，跳過熱力圖生成")
                return
            
            all_positions = sorted(list(all_positions))
            
            # 創建性能矩陣
            performance_matrix = np.zeros((len(algorithms), len(all_positions)))
            
            for i, alg_name in enumerate(algorithms):
                result = individual_results[alg_name]
                ris_perf = result.get('ris_position_performance', {})
                
                for j, pos in enumerate(all_positions):
                    if pos in ris_perf:
                        performance_matrix[i, j] = ris_perf[pos]['average']
                    else:
                        performance_matrix[i, j] = np.nan
            
            # 繪製熱力圖
            fig, ax = plt.subplots(figsize=(10, 8))
            
            im = ax.imshow(performance_matrix, cmap='YlOrRd', aspect='auto')
            
            # 設置標籤
            ax.set_xticks(range(len(all_positions)))
            ax.set_xticklabels([f'RIS-{pos}' for pos in all_positions])
            ax.set_yticks(range(len(algorithms)))
            ax.set_yticklabels(algorithms)
            
            # 添加數值標註
            for i in range(len(algorithms)):
                for j in range(len(all_positions)):
                    if not np.isnan(performance_matrix[i, j]):
                        text = ax.text(j, i, f'{performance_matrix[i, j]:.2f}',
                                     ha="center", va="center", color="black")
            
            ax.set_title(f'RIS位置性能熱力圖 (L = {comparison_results["evaluation_config"]["L_value"]})', 
                        fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=ax, label='性能 (bps/Hz)')
            
            plt.tight_layout()
            
            filename = f'ris_position_heatmap_L{comparison_results["evaluation_config"]["L_value"]}.png'
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  💾 已保存: {filename}")
            
        except Exception as e:
            print(f"❌ RIS位置熱力圖生成失敗: {e}")
    
    def _plot_detailed_analysis(self, comparison_results, save_dir):
        """繪製詳細分析圖"""
        try:
            individual_results = comparison_results['individual_results']
            
            # 準備數據
            algorithms = []
            averages = []
            mins = []
            maxs = []
            medians = []
            
            for alg_name, result in individual_results.items():
                algorithms.append(alg_name)
                averages.append(result['average'])
                mins.append(result['min'])
                maxs.append(result['max'])
                medians.append(result['median'])
            
            # 創建子圖
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            x_pos = np.arange(len(algorithms))
            
            # 子圖1: 平均性能
            ax1.bar(x_pos, averages, alpha=0.8, color='skyblue', edgecolor='black')
            ax1.set_title('平均性能', fontweight='bold')
            ax1.set_ylabel('性能 (bps/Hz)')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(algorithms, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # 子圖2: 性能範圍
            ranges = [max_val - min_val for max_val, min_val in zip(maxs, mins)]
            ax2.bar(x_pos, ranges, alpha=0.8, color='lightcoral', edgecolor='black')
            ax2.set_title('性能變異範圍', fontweight='bold')
            ax2.set_ylabel('範圍 (bps/Hz)')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(algorithms, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # 子圖3: 最大性能
            ax3.bar(x_pos, maxs, alpha=0.8, color='lightgreen', edgecolor='black')
            ax3.set_title('最大性能', fontweight='bold')
            ax3.set_ylabel('性能 (bps/Hz)')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(algorithms, rotation=45, ha='right')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # 子圖4: 中位數性能
            ax4.bar(x_pos, medians, alpha=0.8, color='plum', edgecolor='black')
            ax4.set_title('中位數性能', fontweight='bold')
            ax4.set_ylabel('性能 (bps/Hz)')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(algorithms, rotation=45, ha='right')
            ax4.grid(True, alpha=0.3, axis='y')
            
            plt.suptitle(f'詳細性能分析 (L = {comparison_results["evaluation_config"]["L_value"]})', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            filename = f'detailed_analysis_L{comparison_results["evaluation_config"]["L_value"]}.png'
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  💾 已保存: {filename}")
            
        except Exception as e:
            print(f"❌ 詳細分析圖生成失敗: {e}")
    
    def save_results(self, comparison_results, save_path):
        """保存評估結果"""
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_results, f, indent=2, ensure_ascii=False, default=str)
            print(f"💾 評估結果已保存: {save_path}")
        except Exception as e:
            print(f"❌ 保存結果失敗: {e}")

def create_improved_env_config(L_value, beta_min, environment_data=None):
    """創建改進的環境配置 - 與訓練時一致"""
    base_config = {
        'num_antennas': 4,
        'num_RIS_elements': L_value,
        'num_users': 4,
        'mismatch': True,  # 使用mismatch場景與訓練一致
        'channel_est_error': True,
        'cascaded_channels': True,
        'beta_min': beta_min,
        'theta_bar': 0.0,
        'kappa_bar': 1.5,
        'AWGN_var': 1e-2,
        'channel_noise_var': 1e-2,
        'seed': 0,  # 與訓練時保持一致
    }
    
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
    parser = argparse.ArgumentParser(description='完整算法評估')
    parser.add_argument('--models_dir', default='./Models', help='模型目錄')
    parser.add_argument('--results_dir', default='./outputs/evaluation/evaluation_results_complete', help='結果保存目錄')
    parser.add_argument('--num_episodes', type=int, default=30, help='每個RIS位置的評估次數')
    parser.add_argument('--device', default='cpu', help='計算設備')
    
    args = parser.parse_args()
    
    # 要評估的L值和對應參數
    evaluation_configs = [
        {'L': 16, 'beta_min': 0.3},
        {'L': 32, 'beta_min': 0.6},
        {'L': 64, 'beta_min': 0.6}
    ]
    
    # 要評估的算法（包括新算法）
    algorithms = [
        {'name': 'SAC_golden', 'type': 'SAC'},
        {'name': 'SAC_mismatch', 'type': 'SAC'}, 
        {'name': 'Beta_Space_Exp_SAC_mismatch', 'type': 'Beta_Space_Exp_SAC'},
        {'name': 'DDPG_mismatch', 'type': 'DDPG'},
        {'name': 'PPO_mismatch', 'type': 'PPO'}
    ]
    
    # 基線模型類型
    baseline_types = ['random_phase']
    
    all_results = {}
    
    print("🚀 開始完整算法評估...")
    
    # 對每個L值進行評估
    for config in evaluation_configs:
        L = config['L']
        beta_min = config['beta_min']
        
        print(f"\n{'='*60}")
        print(f"評估 L = {L} 的所有算法")
        print(f"{'='*60}")
        
        # 創建環境配置
        env_config = create_improved_env_config(L, beta_min)
        
        # 創建評估器
        evaluator = CompleteModelEvaluator(env_config, None, args.device)
        
        # 準備訓練模型信息
        trained_models = []
        
        for algorithm in algorithms:
            algorithm_name = algorithm['name']
            algorithm_type = algorithm['type']
            
            # 構建模型路徑
            save_path = f"Beta_min. = {beta_min}, K = 4, M = 4, N = {L}, P_t = 30.0"
            model_path = os.path.join(args.models_dir, save_path, f"{algorithm_name}_0")
            
            # 創建模型配置
            model_config = {
                'N': L,
                'beta_min': beta_min
            }
            
            trained_models.append((algorithm_type, model_path, model_config, algorithm_name))
        
        # 執行完整比較
        try:
            comparison_results = evaluator.compare_all_algorithms(
                trained_models, baseline_types, args.num_episodes
            )
            
            # 生成可視化
            results_subdir = os.path.join(args.results_dir, f'L_{L}')
            evaluator.generate_visualizations(comparison_results, results_subdir)
            
            # 保存結果
            results_file = os.path.join(results_subdir, 'complete_evaluation_results.json')
            evaluator.save_results(comparison_results, results_file)
            
            # 打印排名
            print(f"\n📊 L = {L} 性能排名:")
            for i, (algorithm, perf) in enumerate(comparison_results['performance_ranking']):
                print(f"  {i+1}. {algorithm}: {perf['average']:.3f} ± {perf['std']:.3f}")
            
            all_results[f'L_{L}'] = comparison_results
                
        except Exception as e:
            print(f"❌ L = {L} 評估失敗: {e}")
            continue
    
    # 保存所有結果
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    
    final_results_file = os.path.join(args.results_dir, 'all_algorithms_evaluation.json')
    try:
        with open(final_results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n💾 所有結果已保存: {final_results_file}")
    except Exception as e:
        print(f"❌ 保存最終結果失敗: {e}")
    
    print(f"\n✅ 完整算法評估完成！結果保存在: {args.results_dir}")
    print(f"📋 評估了 {len(algorithms)} 個深度學習算法 + {len(baseline_types)} 個基線算法")
    print(f"🔬 涵蓋 {len(evaluation_configs)} 個不同的L值配置")

if __name__ == "__main__":
    main() 
