#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可視化模塊
用於顯示RIS服務效率、訓練進度和基線比較
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 設置中文字體 - 強化版本
import matplotlib
matplotlib.rcParams['font.family'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# 確保中文字體可用
try:
    import matplotlib.font_manager as fm
    # 嘗試設置中文字體
    for font_name in ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']:
        try:
            font_path = fm.findfont(font_name)
            if font_path:
                plt.rcParams['font.family'] = font_name
                break
        except:
            continue
except ImportError:
    pass

class RISVisualization:
    """RIS實驗結果可視化類"""
    
    def __init__(self):
        """初始化可視化類"""
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        self.method_colors = {
            'SAC_golden': '#2ca02c',
            'SAC_mismatch': '#ff7f0e', 
            'Beta_Space_Exp_SAC_mismatch': '#1f77b4',
            'Random_Phase': '#d62728'
        }
        
    def plot_ris_service_efficiency(self, ris_data, method_name, save_path=None):
        """
        繪製RIS服務效率圖表
        
        參數:
            ris_data: RIS服務數據 {ris_id: [user1_rate, user2_rate, user3_rate, user4_rate]}
            method_name: 方法名稱
            save_path: 保存路徑
        """
        # 計算每個RIS的平均服務速率
        ris_ids = []
        avg_rates = []
        
        for ris_id, user_rates in ris_data.items():
            ris_ids.append(f'RIS {ris_id+1}')
            avg_rates.append(np.mean(user_rates))
        
        # 創建圖表
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        bars = ax.bar(ris_ids, avg_rates, color=self.method_colors.get(method_name, '#1f77b4'), 
                     alpha=0.7, edgecolor='black', linewidth=1)
        
        # 添加數值標籤
        for bar, rate in zip(bars, avg_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{rate:.2f}', ha='center', va='bottom', fontsize=10)
        
        # 設置圖表屬性
        ax.set_xlabel('RIS編號', fontsize=12)
        ax.set_ylabel('平均訊號傳輸速率 (bps/Hz)', fontsize=12)
        ax.set_title(f'{method_name} - RIS服務效率\n' + 
                    f'每個RIS服務4個用戶的平均速率', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加統計信息
        total_avg = np.mean(avg_rates)
        max_rate = np.max(avg_rates)
        min_rate = np.min(avg_rates)
        
        stats_text = f'總平均: {total_avg:.2f}\n最高: {max_rate:.2f}\n最低: {min_rate:.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               verticalalignment='top', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"RIS服務效率圖已保存至: {save_path}")
        
        plt.show()
        return fig, ax
    
    def plot_method_comparison(self, comparison_data, save_path=None):
        """
        繪製多種方法的RIS服務效率比較
        
        參數:
            comparison_data: {method_name: {ris_id: avg_rate}}
            save_path: 保存路徑
        """
        # 準備數據
        methods = list(comparison_data.keys())
        num_ris = len(list(comparison_data.values())[0])
        ris_labels = [f'RIS {i+1}' for i in range(num_ris)]
        
        # 創建圖表
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        x = np.arange(num_ris)
        width = 0.8 / len(methods)
        
        # 繪製每種方法的柱狀圖
        for i, method in enumerate(methods):
            rates = [comparison_data[method][ris_id] for ris_id in range(num_ris)]
            bars = ax.bar(x + i * width, rates, width, 
                         label=method, color=self.method_colors.get(method, self.colors[i]),
                         alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # 添加數值標籤（僅對較高的值）
            for j, (bar, rate) in enumerate(zip(bars, rates)):
                if rate > np.mean(rates) * 0.8:  # 只標註較高的值
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{rate:.2f}', ha='center', va='bottom', 
                           fontsize=8, rotation=0)
        
        # 設置圖表屬性
        ax.set_xlabel('RIS編號', fontsize=12)
        ax.set_ylabel('平均訊號傳輸速率 (bps/Hz)', fontsize=12)
        ax.set_title('不同方法的RIS服務效率比較\n' + 
                    '每個RIS服務4個用戶的平均速率', fontsize=14)
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(ris_labels)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加總體統計
        method_totals = {}
        for method in methods:
            method_totals[method] = np.mean([comparison_data[method][ris_id] for ris_id in range(num_ris)])
        
        # 排序並顯示
        sorted_methods = sorted(method_totals.items(), key=lambda x: x[1], reverse=True)
        stats_text = '總體平均排名:\n'
        for i, (method, avg) in enumerate(sorted_methods):
            stats_text += f'{i+1}. {method}: {avg:.2f}\n'
        
        ax.text(1.02, 0.5, stats_text, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
               verticalalignment='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"方法比較圖已保存至: {save_path}")
        
        plt.show()
        return fig, ax
    
    def plot_training_curves(self, training_data, L_value, save_path=None):
        """
        繪製訓練曲線
        
        參數:
            training_data: {method_name: [reward_list]}
            L_value: RIS元素數量
            save_path: 保存路徑
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        for method, rewards in training_data.items():
            if len(rewards) > 0:
                # 計算移動平均以平滑曲線
                window_size = max(100, len(rewards) // 50)
                smoothed_rewards = self._moving_average(rewards, window_size)
                
                x_values = np.arange(len(smoothed_rewards))
                ax.plot(x_values, smoothed_rewards, 
                       label=method, color=self.method_colors.get(method, '#1f77b4'),
                       linewidth=2, alpha=0.8)
                
                # 添加原始數據的淡化版本
                if len(rewards) < 5000:  # 只對較短的序列顯示原始數據
                    ax.plot(np.arange(len(rewards)), rewards, 
                           color=self.method_colors.get(method, '#1f77b4'),
                           alpha=0.2, linewidth=0.5)
        
        # 設置圖表屬性
        ax.set_xlabel('訓練步數', fontsize=12)
        ax.set_ylabel('訊號傳輸速率 (bps/Hz)', fontsize=12)
        ax.set_title(f'訓練曲線比較 (L = {L_value})\n' + 
                    '移動平均平滑處理', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加最終性能統計
        final_stats = {}
        for method, rewards in training_data.items():
            if len(rewards) > 0:
                final_stats[method] = np.mean(rewards[-100:])  # 最後100步的平均
        
        if final_stats:
            stats_text = '最終性能 (最後100步平均):\n'
            sorted_stats = sorted(final_stats.items(), key=lambda x: x[1], reverse=True)
            for i, (method, final_perf) in enumerate(sorted_stats):
                stats_text += f'{i+1}. {method}: {final_perf:.2f}\n'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                   verticalalignment='top', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"訓練曲線圖已保存至: {save_path}")
        
        plt.show()
        return fig, ax
    
    def plot_all_training_curves(self, all_training_data, save_path=None):
        """
        繪製所有L值的訓練曲線
        
        參數:
            all_training_data: {L_value: {method_name: [reward_list]}}
            save_path: 保存路徑
        """
        L_values = sorted(all_training_data.keys())
        fig, axes = plt.subplots(1, len(L_values), figsize=(6*len(L_values), 6))
        
        if len(L_values) == 1:
            axes = [axes]
        
        for i, L_value in enumerate(L_values):
            ax = axes[i]
            training_data = all_training_data[L_value]
            
            for method, rewards in training_data.items():
                if len(rewards) > 0:
                    # 計算移動平均
                    window_size = max(100, len(rewards) // 50)
                    smoothed_rewards = self._moving_average(rewards, window_size)
                    
                    x_values = np.arange(len(smoothed_rewards))
                    ax.plot(x_values, smoothed_rewards, 
                           label=method, color=self.method_colors.get(method, '#1f77b4'),
                           linewidth=2, alpha=0.8)
            
            # 設置子圖屬性
            ax.set_xlabel('訓練步數', fontsize=10)
            ax.set_ylabel('訊號傳輸速率 (bps/Hz)', fontsize=10)
            ax.set_title(f'L = {L_value}', fontsize=12)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('不同RIS元素數量(L)的訓練曲線比較', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"全部訓練曲線圖已保存至: {save_path}")
        
        plt.show()
        return fig, axes
    
    def plot_performance_summary(self, summary_data, save_path=None):
        """
        繪製性能總結圖
        
        參數:
            summary_data: {L_value: {method_name: final_performance}}
            save_path: 保存路徑
        """
        # 準備數據
        L_values = sorted(summary_data.keys())
        methods = list(summary_data[L_values[0]].keys())
        
        # 創建數據框
        data = []
        for L in L_values:
            for method in methods:
                if method in summary_data[L]:
                    data.append({
                        'L值': f'L={L}',
                        '方法': method,
                        '性能': summary_data[L][method]
                    })
        
        df = pd.DataFrame(data)
        
        # 創建熱力圖
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 熱力圖
        pivot_df = df.pivot(index='方法', columns='L值', values='性能')
        sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='YlOrRd', 
                   ax=ax1, cbar_kws={'label': '性能 (bps/Hz)'})
        ax1.set_title('不同方法和L值的性能熱力圖', fontsize=12)
        
        # 柱狀圖比較
        x = np.arange(len(L_values))
        width = 0.8 / len(methods)
        
        for i, method in enumerate(methods):
            values = [summary_data[L].get(method, 0) for L in L_values]
            bars = ax2.bar(x + i * width, values, width, 
                          label=method, color=self.method_colors.get(method, self.colors[i]),
                          alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # 添加數值標籤
            for bar, value in zip(bars, values):
                if value > 0:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{value:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax2.set_xlabel('RIS元素數量', fontsize=12)
        ax2.set_ylabel('最終性能 (bps/Hz)', fontsize=12)
        ax2.set_title('不同L值下各方法的最終性能', fontsize=12)
        ax2.set_xticks(x + width * (len(methods) - 1) / 2)
        ax2.set_xticklabels([f'L={L}' for L in L_values])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"性能總結圖已保存至: {save_path}")
        
        plt.show()
        return fig, (ax1, ax2)
    
    def _moving_average(self, data, window_size):
        """計算移動平均"""
        if len(data) < window_size:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2 + 1)
            smoothed.append(np.mean(data[start:end]))
        
        return smoothed
    
    def save_results_summary(self, all_results, save_path):
        """保存結果總結"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'results': all_results
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"結果總結已保存至: {save_path}")

def create_sample_data():
    """創建示例數據用於測試"""
    np.random.seed(42)
    
    # 示例RIS服務數據
    ris_service_data = {}
    for method in ['SAC_golden', 'SAC_mismatch', 'Beta_Space_Exp_SAC_mismatch', 'Random_Phase']:
        ris_service_data[method] = {}
        for ris_id in range(10):
            # 模擬不同方法的性能差異
            if method == 'Beta_Space_Exp_SAC_mismatch':
                base_rate = np.random.uniform(4.5, 6.0)
            elif method == 'SAC_golden':
                base_rate = np.random.uniform(4.0, 5.5)
            elif method == 'SAC_mismatch':
                base_rate = np.random.uniform(3.5, 5.0)
            else:  # Random_Phase
                base_rate = np.random.uniform(2.0, 3.5)
            
            ris_service_data[method][ris_id] = base_rate
    
    # 示例訓練數據
    training_data = {}
    for L in [16, 32, 64]:
        training_data[L] = {}
        for method in ['SAC_golden', 'SAC_mismatch', 'Beta_Space_Exp_SAC_mismatch']:
            if method == 'Beta_Space_Exp_SAC_mismatch':
                trend = np.linspace(2, 5.5, 20000)
            elif method == 'SAC_golden':
                trend = np.linspace(2, 5.0, 20000)
            else:
                trend = np.linspace(2, 4.5, 20000)
            
            noise = np.random.normal(0, 0.2, 20000)
            training_data[L][method] = trend + noise
    
    return ris_service_data, training_data

# 測試函數
if __name__ == "__main__":
    # 創建可視化對象
    viz = RISVisualization()
    
    # 創建示例數據
    ris_data, train_data = create_sample_data()
    
    # 測試可視化功能
    print("🎨 測試可視化功能...")
    
    # 1. RIS服務效率比較
    viz.plot_method_comparison(ris_data, 'test_ris_comparison.png')
    
    # 2. 訓練曲線
    viz.plot_all_training_curves(train_data, 'test_training_curves.png')
    
    # 3. 性能總結
    summary = {}
    for L in [16, 32, 64]:
        summary[L] = {}
        for method in ['SAC_golden', 'SAC_mismatch', 'Beta_Space_Exp_SAC_mismatch']:
            summary[L][method] = np.mean(train_data[L][method][-100:])
    
    viz.plot_performance_summary(summary, 'test_performance_summary.png')
    
    print("✅ 可視化功能測試完成！") 