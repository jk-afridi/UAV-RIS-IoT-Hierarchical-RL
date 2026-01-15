#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量訓練腳本
用於訓練9個模型配置：3種方法 × 3種L參數

方法：
1. SAC_golden
2. SAC_mismatch  
3. Beta_Space_Exp_SAC_mismatch

參數配置：
- L=16, Beta_min=0.3, K=4, M=4, P_t=30.0
- L=32, Beta_min=0.6, K=4, M=4, P_t=30.0
- L=64, Beta_min=0.6, K=4, M=4, P_t=30.0
"""

import subprocess
import sys
import os
from pathlib import Path
import json
from datetime import datetime
import time
import numpy as np

def check_model_exists(config):
    """
    檢查模型是否已存在且訓練完整
    
    參數:
        config: 配置字典
        
    返回:
        bool: 如果模型存在且完整則返回True
    """
    try:
        # 構建模型保存路徑
        save_path = f"Beta_min. = {config['beta_min']}, K = {config['num_users']}, M = {config['num_antennas']}, N = {config['num_RIS_elements']}, P_t = {float(config['power_t'])}"
        model_dir = f"./Models/{save_path}"
        result_dir = f"./Results/{save_path}"
        
        # 構建文件名（與main.py中的保存邏輯一致）
        file_name = f"{config['policy']}_{config['objective_function']}_{config['seed']}"
        
        # 檢查模型文件是否存在
        model_files = [
            f"{model_dir}/{file_name}_actor",
            f"{model_dir}/{file_name}_critic",
        ]
        
        # 檢查結果文件是否存在
        result_file = f"{result_dir}/{file_name}.npy"
        
        # 所有文件都必須存在
        for file_path in model_files:
            if not os.path.exists(file_path):
                print(f"  📋 缺少模型文件: {os.path.basename(file_path)}")
                return False
        
        if not os.path.exists(result_file):
            print(f"  📋 缺少結果文件: {os.path.basename(result_file)}")
            return False
        
        # 檢查結果文件的訓練步數
        try:
            rewards = np.load(result_file)
            expected_steps = config['max_time_steps']
            actual_steps = len(rewards)
            
            if actual_steps < expected_steps * 0.95:  # 允許5%的容錯
                print(f"  📋 訓練不完整: {actual_steps}/{expected_steps} 步")
                return False
            
            print(f"  ✅ 模型完整: {actual_steps} 步訓練")
            return True
            
        except Exception as e:
            print(f"  ❌ 無法讀取結果文件: {e}")
            return False
            
    except Exception as e:
        print(f"  ❌ 檢查模型時出錯: {e}")
        return False

def create_config():
    """創建實驗配置"""
    configs = []
    
    # 實驗參數配置
    base_params = {
        'num_antennas': 4,
        'num_users': 4, 
        'power_t': 30.0,
        'max_time_steps': 20000,
        'buffer_size': 20000,
        'batch_size': 16,
        'save_model': True,
        'actor_lr': 1e-3,
        'critic_lr': 1e-3,
        'seed': 0
    }
    
    # L參數和對應的beta_min值
    l_configs = [
        {'num_RIS_elements': 16, 'beta_min': 0.3},
        {'num_RIS_elements': 32, 'beta_min': 0.6},
        {'num_RIS_elements': 64, 'beta_min': 0.6}
    ]
    
    # 三種方法配置
    methods = [
        {
            'name': 'SAC_golden',
            'policy': 'SAC',
            'objective_function': 'golden'
        },
        {
            'name': 'SAC_mismatch', 
            'policy': 'SAC',
            'objective_function': 'mismatch'
        },
        {
            'name': 'Beta_Space_Exp_SAC_mismatch',
            'policy': 'Beta_Space_Exp_SAC', 
            'objective_function': 'mismatch'
        }
    ]
    
    # 生成所有配置組合
    for l_config in l_configs:
        for method in methods:
            config = base_params.copy()
            config.update(l_config)
            config.update(method)
            config['config_name'] = f"{method['name']}_L{l_config['num_RIS_elements']}_beta{l_config['beta_min']}"
            configs.append(config)
    
    return configs

def run_training(config):
    """運行單個訓練配置"""
    print(f"\n{'='*60}")
    print(f"開始訓練: {config['config_name']}")
    print(f"{'='*60}")
    
    # 構建命令行參數
    cmd = [
        sys.executable, str(Path(__file__).resolve().parent / 'main.py'),
        '--policy', config['policy'],
        '--objective_function', config['objective_function'],
        '--num_antennas', str(config['num_antennas']),
        '--num_RIS_elements', str(config['num_RIS_elements']),
        '--num_users', str(config['num_users']),
        '--power_t', str(config['power_t']),
        '--beta_min', str(config['beta_min']),
        '--max_time_steps', str(config['max_time_steps']),
        '--buffer_size', str(config['buffer_size']),
        '--batch_size', str(config['batch_size']),
        '--actor_lr', str(config['actor_lr']),
        '--critic_lr', str(config['critic_lr']),
        '--seed', str(config['seed']),
        '--save_model', str(config['save_model']).lower()
    ]
    
    start_time = time.time()
    
    try:
        # 運行訓練 - 即時顯示輸出
        print(f"執行命令: {' '.join(cmd)}")
        
        try:
            # 使用Popen來即時顯示輸出
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                universal_newlines=True
            )
            
            output_lines = []
            # 即時顯示輸出
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(f"    {output.strip()}")
                    output_lines.append(output)
            
            return_code = process.poll()
            full_output = ''.join(output_lines)
            
            # 創建結果對象以模擬subprocess.run的返回
            class Result:
                def __init__(self, returncode, stdout, stderr=""):
                    self.returncode = returncode
                    self.stdout = stdout
                    self.stderr = stderr
            
            result = Result(return_code, full_output)
            
        except Exception as e:
            print(f"    執行過程中出錯: {e}")
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                encoding='utf-8',
                errors='replace'
            )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ 訓練完成: {config['config_name']}")
            print(f"⏱️ 訓練時間: {training_time:.2f}秒")
            return {'success': True, 'time': training_time, 'output': result.stdout}
        else:
            print(f"❌ 訓練失敗: {config['config_name']}")
            print(f"錯誤訊息:\n{result.stderr}")
            return {'success': False, 'error': result.stderr}
            
    except Exception as e:
        print(f"❌ 執行錯誤: {config['config_name']}")
        print(f"錯誤: {str(e)}")
        return {'success': False, 'error': str(e)}

def save_training_log(configs, results):
    """保存訓練日誌"""
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'total_configs': len(configs),
        'successful_runs': sum(1 for r in results if r['success']),
        'failed_runs': sum(1 for r in results if not r['success']),
        'configs': configs,
        'results': results
    }
    
    log_file = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n📝 訓練日誌已保存: {log_file}")
    return log_file

def main():
    """主函數"""
    print("🚀 開始批量訓練實驗")
    print(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 創建配置
    configs = create_config()
    print(f"\n📊 總共需要訓練 {len(configs)} 個模型配置")
    
    # 顯示配置清單
    print("\n📋 訓練配置清單:")
    for i, config in enumerate(configs, 1):
        print(f"  {i:2d}. {config['config_name']}")
    
    # 自動開始訓練，不需要確認
    print("🚀 自動開始批量訓練...")
    
    # 執行訓練
    results = []
    total_start_time = time.time()
    failed_configs = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n🎯 進度: {i}/{len(configs)}")
        
        # 檢查模型是否已存在且完整
        if check_model_exists(config):
            print(f"✅ 模型已存在且完整，跳過: {config['config_name']}")
            results.append({'success': True, 'time': 0, 'output': 'Model already exists'})
            continue
        
        result = run_training(config)
        results.append(result)
        
        # 記錄失敗的配置
        if not result['success']:
            failed_configs.append(config['config_name'])
        
        # 每完成一個配置後稍作休息
        if i < len(configs):
            print("⏸️ 休息3秒後繼續下一個配置...")
            time.sleep(3)
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    # 統計結果
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"\n{'='*60}")
    print("🎉 批量訓練完成！")
    print(f"{'='*60}")
    print(f"📊 總體統計:")
    print(f"  ✅ 成功: {successful}/{len(configs)}")
    print(f"  ❌ 失敗: {failed}/{len(configs)}")
    print(f"  ⏱️ 總時間: {total_time:.2f}秒 ({total_time/3600:.2f}小時)")
    print(f"  📈 平均時間: {total_time/len(configs):.2f}秒/配置")
    
    # 保存日誌
    log_file = save_training_log(configs, results)
    
    if failed > 0:
        print(f"\n⚠️ 有 {failed} 個配置訓練失敗:")
        for config_name in failed_configs:
            print(f"  - {config_name}")
        print(f"📝 詳細錯誤請檢查日誌: {log_file}")
        
        # 自動重新嘗試失敗的配置（最多1次）
        if failed_configs:
            print("🔄 自動重新嘗試失敗的配置...")
            retry_configs = [config for config in configs if config['config_name'] in failed_configs]
            
            for config in retry_configs:
                print(f"\n🔄 重試: {config['config_name']}")
                retry_result = run_training(config)
                if retry_result['success']:
                    print(f"✅ 重試成功: {config['config_name']}")
                    failed_configs.remove(config['config_name'])
                else:
                    print(f"❌ 重試仍然失敗: {config['config_name']}")
    
    if len(failed_configs) == 0:
        print(f"\n🎉 所有訓練任務完成！")
    else:
        print(f"\n⚠️ 批量訓練完成，但有 {len(failed_configs)} 個配置仍然失敗")

if __name__ == "__main__":
    main() 