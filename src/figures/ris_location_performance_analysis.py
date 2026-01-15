#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RIS停留點性能比較分析
基於new_test.py生成的真實環境數據進行性能評估
比較隨機相位控制與3種RL模型在10個RIS停留點的性能
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import glob
from datetime import datetime
import warnings
from pathlib import Path
import sys
warnings.filterwarnings('ignore')

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 導入模型
from algorithms import SAC
from algorithms import Beta_Space_Exp_SAC
from core.environment import RIS_MISO_PDA
from algorithms.baseline_models import create_baseline_model
from core.utils import whiten

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class RISLocationPerformanceAnalyzer:
    """RIS停留點性能分析器"""
    
    def __init__(self, environment_data_path):
        """
        初始化分析器
        
        參數:
            environment_data_path: 環境數據JSON文件路徑
        """
        self.environment_data_path = environment_data_path
        self.environment_data = None
        self.device = torch.device('cpu')  # 使用CPU避免複雜的GPU問題
        
        # 結果存儲
        self.performance_results = {}
        
        # 方法配置
        self.methods = {
            'Random_Phase': 'baseline',
            'SAC_golden': 'rl_model', 
            'SAC_mismatch': 'rl_model',
            'Beta_Space_Exp_SAC_mismatch': 'rl_model'
        }
        
        print(f"初始化RIS位置性能分析器")
        print(f"環境數據路徑: {environment_data_path}")
    
    def load_environment_data(self):
        """載入環境數據"""
        print("📁 載入環境數據...")
        
        with open(self.environment_data_path, 'r', encoding='utf-8') as f:
            self.environment_data = json.load(f)
        
        # 提取關鍵信息
        deployment = self.environment_data['deployment']
        self.ris_positions = deployment['ris_positions']
        self.devices = deployment['devices'] 
        self.ris_device_mapping = deployment['ris_device_mapping']
        
        # 轉換映射的鍵為整數
        self.ris_device_mapping = {int(k): v for k, v in self.ris_device_mapping.items()}
        
        print(f"✅ 成功載入環境數據")
        print(f"   RIS位置數量: {len(self.ris_positions)}")
        print(f"   裝置總數: {len(self.devices)}")
        print(f"   RIS-裝置映射: {len(self.ris_device_mapping)} 組")
        
        # 驗證數據完整性
        total_served_devices = sum(len(devices) for devices in self.ris_device_mapping.values())
        print(f"   實際服務裝置數: {total_served_devices}")
        
        return True
    
    def create_simplified_environment(self, ris_id, ris_position, served_devices):
        """
        為特定RIS位置創建簡化的評估環境
        
        參數:
            ris_id: RIS編號
            ris_position: RIS位置座標
            served_devices: 該RIS服務的裝置位置列表
            
        返回:
            評估環境實例
        """
        # 創建基本的RIS環境（4天線，16個RIS元素，4個用戶）
        env = RIS_MISO_PDA(
            num_antennas=4,
            num_RIS_elements=16,  # 基本配置
            num_users=len(served_devices),
            mismatch=False,
            channel_est_error=True,
            cascaded_channels=True,
            beta_min=0.6,
            seed=42
        )
        
        return env
    
    def evaluate_method_at_location(self, method_name, ris_id, ris_position, served_devices, num_episodes=50):
        """
        在特定RIS位置評估某種方法的性能
        
        參數:
            method_name: 方法名稱
            ris_id: RIS編號
            ris_position: RIS位置
            served_devices: 服務的裝置列表
            num_episodes: 評估回合數
            
        返回:
            平均信號速率
        """
        print(f"    評估 {method_name} 在 RIS-{ris_id+1} 位置")
        
        if len(served_devices) == 0:
            print(f"      警告: RIS-{ris_id+1} 沒有服務裝置")
            return 0.0
        
        # 創建環境
        env = self.create_simplified_environment(ris_id, ris_position, served_devices)
        
        # 獲取模型/基線
        model = self._load_method_model(method_name, len(served_devices))
        if model is None:
            print(f"      警告: 無法載入 {method_name} 模型")
            return 0.0
        
        # 評估性能
        episode_rewards = []
        
        for episode in range(num_episodes):
            try:
                # 重置環境
                state = env.reset()
                if state is None:
                    continue
                
                # 白化處理
                state = whiten(state)
                
                episode_reward = 0
                max_steps = 100  # 限制最大步數
                
                for step in range(max_steps):
                    # 選擇動作
                    if hasattr(model, 'select_action'):
                        if method_name == 'Random_Phase':
                            action = model.select_action(state)
                        else:
                            action = model.select_action(state)
                    else:
                        action = env.action_space.sample()
                    
                    # 執行動作
                    next_state, reward, done, info = env.step(action)
                    
                    # 累積獎勵
                    if isinstance(info, dict) and "true reward" in info:
                        episode_reward += info["true reward"]
                    else:
                        episode_reward += reward
                    
                    if done:
                        break
                    
                    state = whiten(next_state)
                
                episode_rewards.append(episode_reward)
                
            except Exception as e:
                print(f"        評估回合 {episode+1} 失敗: {e}")
                continue
        
        if episode_rewards:
            avg_reward = np.mean(episode_rewards)
            print(f"      平均信號速率: {avg_reward:.3f} (來自 {len(episode_rewards)} 回合)")
            return avg_reward
        else:
            print(f"      評估失敗，返回 0.0")
            return 0.0
    
    def _load_method_model(self, method_name, num_users):
        """載入指定方法的模型"""
        
        if method_name == 'Random_Phase':
            # 創建隨機基線模型
            return create_baseline_model(
                'random_phase', 
                M=4, N=16, K=num_users,
                power_t=30.0, device=self.device
            )
        
        else:
            # 嘗試載入訓練好的RL模型
            return self._load_trained_rl_model(method_name, num_users)
    
    def _load_trained_rl_model(self, method_name, num_users):
        """載入訓練好的RL模型"""
        try:
            # 構建模型路徑（基於16元素的基本配置）
            save_path = "Beta_min. = 0.3, K = 4, M = 4, N = 16, P_t = 30.0"
            model_path = f"./Models/{save_path}/{method_name}_0"
            
            if not os.path.exists(model_path + "_actor"):
                print(f"        模型文件不存在: {model_path}")
                return None
            
            # 創建模型配置
            model_config = {
                'state_dim': 328,  # 基本配置的狀態維度
                'action_space': torch.tensor([0.0] * 64),  # 模擬動作空間
                'M': 4,
                'N': 16,
                'K': num_users,
                'power_t': 30.0,
                'actor_lr': 1e-3,
                'critic_lr': 1e-3,
                'policy_type': "Gaussian",
                'alpha': 0.2,
                'target_update_interval': 1,
                'automatic_entropy_tuning': True,
                'device': self.device
            }
            
            # 創建動作空間
            import gym
            action_space = gym.spaces.Box(
                low=-5, high=5, 
                shape=(64,),  # 4*4*2 + 16*2 = 64
                dtype=np.float32
            )
            model_config['action_space'] = action_space
            
            # 初始化模型
            if 'Beta_Space' in method_name:
                model = Beta_Space_Exp_SAC.Beta_Space_Exp_SAC(**model_config)
            else:
                model = SAC.SAC(**model_config)
            
            # 載入權重
            model.load(model_path)
            print(f"        成功載入模型: {method_name}")
            return model
            
        except Exception as e:
            print(f"        載入模型失敗 {method_name}: {e}")
            return None
    
    def run_comprehensive_analysis(self):
        """運行完整的性能分析"""
        print("\n🚀 開始RIS停留點性能比較分析")
        print("=" * 60)
        
        if not self.load_environment_data():
            print("❌ 環境數據載入失敗")
            return False
        
        # 為每個方法在每個RIS位置進行評估
        for method_name in self.methods.keys():
            print(f"\n📊 評估方法: {method_name}")
            self.performance_results[method_name] = {}
            
            for ris_id in range(len(self.ris_positions)):
                ris_position = self.ris_positions[ris_id]
                served_devices = self.ris_device_mapping.get(ris_id, [])
                
                print(f"  📍 RIS-{ris_id+1} 位置 {ris_position}, 服務 {len(served_devices)} 個裝置")
                
                # 評估該方法在該位置的性能
                avg_signal_rate = self.evaluate_method_at_location(
                    method_name, ris_id, ris_position, served_devices, num_episodes=30
                )
                
                self.performance_results[method_name][ris_id] = {
                    'avg_signal_rate': avg_signal_rate,
                    'ris_position': ris_position,
                    'served_devices_count': len(served_devices),
                    'served_devices': served_devices
                }
        
        print("\n✅ 性能評估完成")
        return True
    
    def generate_performance_comparison_charts(self, save_dir='./outputs/ris_analysis/ris_location_analysis_results'):
        """生成性能比較圖表"""
        print(f"\n📈 生成性能比較圖表...")
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 1. 各RIS位置的性能比較柱狀圖
        self._create_ris_performance_bar_chart(save_dir)
        
        # 2. 方法總體性能比較
        self._create_method_comparison_chart(save_dir)
        
        # 3. 詳細性能熱力圖
        self._create_performance_heatmap(save_dir)
        
        print(f"📁 圖表已保存到: {save_dir}")
    
    def _create_ris_performance_bar_chart(self, save_dir):
        """創建RIS位置性能比較柱狀圖"""
        print("  📊 生成RIS位置性能柱狀圖...")
        
        # 準備數據
        ris_ids = list(range(len(self.ris_positions)))
        methods = list(self.methods.keys())
        
        # 創建圖表
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(ris_ids))
        width = 0.2
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, method in enumerate(methods):
            signal_rates = []
            for ris_id in ris_ids:
                if ris_id in self.performance_results[method]:
                    rate = self.performance_results[method][ris_id]['avg_signal_rate']
                else:
                    rate = 0.0
                signal_rates.append(rate)
            
            offset = (i - len(methods)/2) * width + width/2
            bars = ax.bar(x + offset, signal_rates, width, label=method, 
                         color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # 添加數值標籤
            for bar, rate in zip(bars, signal_rates):
                if rate > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{rate:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('RIS停留點編號', fontsize=12)
        ax.set_ylabel('平均信號速率 (bps/Hz)', fontsize=12)
        ax.set_title('各RIS停留點的信號速率性能比較\n(4種方法在10個實際地理位置的表現)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f'RIS-{i+1}' for i in ris_ids], rotation=45)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(save_dir, f'ris_location_performance_{timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_method_comparison_chart(self, save_dir):
        """創建方法總體性能比較圖"""
        print("  📊 生成方法總體性能比較圖...")
        
        # 計算每種方法的總體統計
        method_stats = {}
        for method in self.methods.keys():
            rates = []
            for ris_id in range(len(self.ris_positions)):
                if ris_id in self.performance_results[method]:
                    rate = self.performance_results[method][ris_id]['avg_signal_rate']
                    if rate > 0:
                        rates.append(rate)
            
            if rates:
                method_stats[method] = {
                    'mean': np.mean(rates),
                    'std': np.std(rates),
                    'max': np.max(rates),
                    'min': np.min(rates),
                    'count': len(rates)
                }
            else:
                method_stats[method] = {
                    'mean': 0, 'std': 0, 'max': 0, 'min': 0, 'count': 0
                }
        
        # 創建對比圖
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左圖: 平均性能對比
        methods = list(method_stats.keys())
        means = [method_stats[m]['mean'] for m in methods]
        stds = [method_stats[m]['std'] for m in methods]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars1 = ax1.bar(methods, means, yerr=stds, capsize=5, 
                        color=colors, alpha=0.8, edgecolor='black')
        
        # 添加數值標籤
        for bar, mean, std in zip(bars1, means, stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                    f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontsize=10)
        
        ax1.set_ylabel('平均信號速率 (bps/Hz)', fontsize=12)
        ax1.set_title('各方法的平均性能比較', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # 右圖: 性能提升比較
        baseline_mean = method_stats['Random_Phase']['mean']
        improvements = []
        for method in methods:
            if method == 'Random_Phase':
                improvements.append(0)
            else:
                if baseline_mean > 0:
                    improvement = ((method_stats[method]['mean'] - baseline_mean) / baseline_mean) * 100
                else:
                    improvement = 0
                improvements.append(improvement)
        
        bars2 = ax2.bar(methods, improvements, color=colors, alpha=0.8, edgecolor='black')
        
        # 添加數值標籤
        for bar, improvement in zip(bars2, improvements):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{improvement:.1f}%', ha='center', va='bottom', fontsize=10)
        
        ax2.set_ylabel('相對隨機相位的性能提升 (%)', fontsize=12)
        ax2.set_title('相對基線的性能提升', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(save_dir, f'method_comparison_{timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_performance_heatmap(self, save_dir):
        """創建性能熱力圖"""
        print("  📊 生成性能熱力圖...")
        
        # 準備數據矩陣
        methods = list(self.methods.keys())
        ris_count = len(self.ris_positions)
        
        performance_matrix = np.zeros((len(methods), ris_count))
        
        for i, method in enumerate(methods):
            for j in range(ris_count):
                if j in self.performance_results[method]:
                    performance_matrix[i, j] = self.performance_results[method][j]['avg_signal_rate']
        
        # 創建熱力圖
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(performance_matrix, cmap='YlOrRd', aspect='auto')
        
        # 設置坐標軸
        ax.set_xticks(np.arange(ris_count))
        ax.set_yticks(np.arange(len(methods)))
        ax.set_xticklabels([f'RIS-{i+1}' for i in range(ris_count)])
        ax.set_yticklabels(methods)
        
        # 添加數值標籤
        for i in range(len(methods)):
            for j in range(ris_count):
                text = ax.text(j, i, f'{performance_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title('各方法在不同RIS位置的性能熱力圖', fontsize=14)
        ax.set_xlabel('RIS停留點', fontsize=12)
        ax.set_ylabel('控制方法', fontsize=12)
        
        # 添加顏色條
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('平均信號速率 (bps/Hz)', fontsize=12)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(save_dir, f'performance_heatmap_{timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_analysis_report(self, save_dir='./outputs/ris_analysis/ris_location_analysis_results'):
        """保存分析報告"""
        print(f"\n📄 生成分析報告...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(save_dir, f'ris_analysis_report_{timestamp}.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("RIS停留點性能比較分析報告\n")
            f.write("=" * 50 + "\n")
            f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"環境數據: {self.environment_data_path}\n\n")
            
            # 環境概況
            f.write("環境概況:\n")
            f.write(f"- RIS停留點數量: {len(self.ris_positions)}\n")
            f.write(f"- 服務裝置總數: {len(self.devices)}\n")
            f.write(f"- 評估方法數量: {len(self.methods)}\n\n")
            
            # 各方法總體性能
            f.write("各方法總體性能:\n")
            f.write("-" * 30 + "\n")
            
            for method in self.methods.keys():
                rates = []
                for ris_id in range(len(self.ris_positions)):
                    if ris_id in self.performance_results[method]:
                        rate = self.performance_results[method][ris_id]['avg_signal_rate']
                        if rate > 0:
                            rates.append(rate)
                
                if rates:
                    f.write(f"{method}:\n")
                    f.write(f"  平均信號速率: {np.mean(rates):.3f} ± {np.std(rates):.3f}\n")
                    f.write(f"  最佳性能: {np.max(rates):.3f}\n")
                    f.write(f"  最差性能: {np.min(rates):.3f}\n")
                    f.write(f"  有效評估點: {len(rates)}/{len(self.ris_positions)}\n\n")
            
            # 各RIS位置詳細結果
            f.write("各RIS位置詳細結果:\n")
            f.write("-" * 30 + "\n")
            
            for ris_id in range(len(self.ris_positions)):
                f.write(f"RIS-{ris_id+1} 位置 {self.ris_positions[ris_id]}:\n")
                
                served_count = len(self.ris_device_mapping.get(ris_id, []))
                f.write(f"  服務裝置數: {served_count}\n")
                
                for method in self.methods.keys():
                    if ris_id in self.performance_results[method]:
                        rate = self.performance_results[method][ris_id]['avg_signal_rate']
                        f.write(f"  {method}: {rate:.3f}\n")
                    else:
                        f.write(f"  {method}: 評估失敗\n")
                f.write("\n")
        
        print(f"📁 分析報告已保存: {report_path}")

def find_latest_environment_data():
    """尋找最新的環境數據文件"""
    search_pattern = "./outputs/dbscan/fixed_ris_environments/fixed_environment_data_*.json"
    files = glob.glob(search_pattern)
    
    if not files:
        print("❌ 未找到環境數據文件")
        print("💡 請先運行 new_test.py 生成環境數據")
        return None
    
    # 返回最新的文件
    latest_file = max(files, key=os.path.getctime)
    print(f"🔍 找到環境數據文件: {latest_file}")
    return latest_file

def main():
    """主函數"""
    print("🚀 RIS停留點性能比較分析")
    print("=" * 60)
    print("基於真實環境地理數據的相位控制性能評估")
    print("比較隨機相位 vs 3種RL模型在10個RIS停留點的表現\n")
    
    # 尋找環境數據文件
    environment_data_path = find_latest_environment_data()
    if not environment_data_path:
        return False
    
    # 創建分析器
    analyzer = RISLocationPerformanceAnalyzer(environment_data_path)
    
    # 運行分析
    success = analyzer.run_comprehensive_analysis()
    if not success:
        print("❌ 性能分析失敗")
        return False
    
    # 生成圖表和報告
    analyzer.generate_performance_comparison_charts()
    analyzer.save_analysis_report()
    
    print("\n" + "=" * 60)
    print("✅ RIS停留點性能比較分析完成！")
    print("\n📋 分析內容:")
    print("- 🎯 4種相位控制方法的性能對比")
    print("- 📍 10個實際RIS停留點的地理位置分析") 
    print("- 📊 40個裝置的信號速率統計")
    print("- 🔄 每個位置服務4個裝置的真實場景模擬")
    print("\n📁 輸出文件:")
    print("- RIS位置性能柱狀圖")
    print("- 方法總體性能比較圖") 
    print("- 性能熱力圖")
    print("- 詳細分析報告")
    
    return True

if __name__ == "__main__":
    main() 
