#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新算法批量訓練腳本
訓練6個新模型：DDPG和PPO的L=16,32,64

模型列表：
1. DDPG_L16 (β_min=0.3)
2. DDPG_L32 (β_min=0.6) 
3. DDPG_L64 (β_min=0.6)
4. PPO_L16 (β_min=0.3)
5. PPO_L32 (β_min=0.6)
6. PPO_L64 (β_min=0.6)
"""

import subprocess
import sys
import os
import time
from pathlib import Path
from datetime import datetime

def print_header(title):
    """打印標題"""
    print("=" * 80)
    print(f"🚀 {title}")
    print("=" * 80)

def print_model_info(algorithm, L, beta_min, model_num, total_models):
    """打印模型信息"""
    print(f"\n📋 模型 {model_num}/{total_models}: {algorithm}_L{L}_β{beta_min}")
    print(f"⚙️  配置: L={L}, β_min={beta_min}, 算法={algorithm}")

def run_training_command(algorithm, L, beta_min):
    """
    運行訓練命令
    
    參數:
        algorithm: 算法名稱 (DDPG/PPO)
        L: RIS元件數量
        beta_min: 最小beta值
        
    返回:
        是否成功
    """
    cmd = [
        sys.executable, str(Path(__file__).resolve().parent / 'train_new_algorithms.py'),
        '--algorithm', algorithm,
        '--num_RIS_elements', str(L),
        '--beta_min', str(beta_min),
        '--max_time_steps', '20000',
        '--save_model', 'True',
        '--seed', '0'
    ]
    
    print(f"🔧 執行命令: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, check=True, capture_output=False)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"✅ 訓練完成! 耗時: {duration/60:.1f}分鐘")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 訓練失敗! 錯誤碼: {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ 訓練失敗! 錯誤: {e}")
        return False

def main():
    """主函數"""
    print_header("新算法批量訓練開始")
    
    # 訓練配置
    training_configs = [
        # DDPG模型
        {'algorithm': 'DDPG', 'L': 16, 'beta_min': 0.3},
        {'algorithm': 'DDPG', 'L': 32, 'beta_min': 0.6},
        {'algorithm': 'DDPG', 'L': 64, 'beta_min': 0.6},
        # PPO模型
        {'algorithm': 'PPO', 'L': 16, 'beta_min': 0.3},
        {'algorithm': 'PPO', 'L': 32, 'beta_min': 0.6},
        {'algorithm': 'PPO', 'L': 64, 'beta_min': 0.6},
    ]
    
    total_models = len(training_configs)
    successful_models = 0
    failed_models = []
    
    print(f"📊 總計訓練模型數: {total_models}")
    print(f"⏱️  預估總時間: {total_models * 30:.0f}分鐘 (每模型約30分鐘)")
    print(f"🕐 開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    overall_start_time = time.time()
    
    # 逐個訓練模型
    for i, config in enumerate(training_configs, 1):
        algorithm = config['algorithm']
        L = config['L']
        beta_min = config['beta_min']
        
        print_model_info(algorithm, L, beta_min, i, total_models)
        
        # 檢查是否要跳過
        print("開始訓練...")
        
        # 執行訓練
        success = run_training_command(algorithm, L, beta_min)
        
        if success:
            successful_models += 1
            print(f"🎉 模型 {i}/{total_models} 訓練成功")
        else:
            failed_models.append(f"{algorithm}_L{L}_β{beta_min}")
            print(f"💥 模型 {i}/{total_models} 訓練失敗")
        
        # 顯示進度
        progress = i / total_models * 100
        print(f"📈 整體進度: {progress:.1f}% ({i}/{total_models})")
        
        if i < total_models:
            print("⏸️  等待3秒後繼續下一個模型...")
            time.sleep(3)
    
    # 最終總結
    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time
    
    print_header("批量訓練完成")
    print(f"🏁 結束時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  總耗時: {total_duration/60:.1f}分鐘 ({total_duration/3600:.1f}小時)")
    print(f"✅ 成功訓練: {successful_models}/{total_models} 模型")
    print(f"❌ 失敗訓練: {len(failed_models)}/{total_models} 模型")
    
    if failed_models:
        print(f"💥 失敗模型列表:")
        for model in failed_models:
            print(f"   - {model}")
        print(f"\n🔄 重試失敗模型的命令:")
        for model in failed_models:
            parts = model.split('_')
            alg = parts[0]
            L = parts[1][1:]  # 去掉'L'
            beta = parts[2][1:]  # 去掉'β' 
            print(f"   python train_new_algorithms.py --algorithm {alg} --num_RIS_elements {L} --beta_min {beta}")
    else:
        print("🎉 所有模型訓練成功!")
    
    print("=" * 80)

if __name__ == "__main__":
    main() 