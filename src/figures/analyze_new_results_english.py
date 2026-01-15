#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
New Algorithm Training Results Analysis Script (English Version)
Support for SAC, DDPG, PPO and Beta_Space_Exp_SAC algorithm result analysis
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from datetime import datetime
import seaborn as sns

def setup_english_fonts():
    """Setup English fonts"""
    import matplotlib.font_manager as fm
    
    # Use standard English fonts
    english_fonts = [
        'DejaVu Sans',
        'Arial', 
        'Helvetica',
        'Times New Roman'
    ]
    
    for font_name in english_fonts:
        if font_name in [f.name for f in fm.fontManager.ttflist]:
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            return font_name
    
    print("⚠️ Using default font")
    return 'DejaVu Sans'

class NewAlgorithmAnalyzer:
    """New Algorithm Training Results Analyzer"""
    
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.font_name = setup_english_fonts()
        
        # Algorithm color mapping
        self.algorithm_colors = {
            'SAC_golden': '#2ca02c',        # Green
            'SAC_mismatch': '#ff7f0e',      # Orange
            'Beta_Space_Exp_SAC_mismatch': '#1f77b4',  # Blue
            'DDPG_mismatch': '#d62728',     # Red
            'PPO_mismatch': '#9467bd',      # Purple
        }
        
        # Algorithm label mapping
        self.algorithm_labels = {
            'SAC_golden': 'SAC (Golden)',
            'SAC_mismatch': 'SAC (Mismatch)',
            'Beta_Space_Exp_SAC_mismatch': 'Beta-Space SAC',
            'DDPG_mismatch': 'DDPG',
            'PPO_mismatch': 'PPO',
        }
        
        print(f"Font setup: {self.font_name}")
        
    def load_training_results(self):
        """Load all training results"""
        print("🔍 Loading training results...")
        
        training_data = {}
        
        # Search all result directories
        for root, dirs, files in os.walk(self.results_dir):
            for file in files:
                if file.endswith('.npy'):
                    file_path = os.path.join(root, file)
                    
                    try:
                        # Parse directory name to get parameters
                        dir_name = os.path.basename(root)
                        params = self._parse_directory_name(dir_name)
                        
                        # Parse filename to get algorithm
                        algorithm_info = self._parse_algorithm_name(file)
                        
                        if params and algorithm_info:
                            # Load data
                            rewards = np.load(file_path)
                            
                            L_value = params['L']
                            algorithm = algorithm_info['algorithm']
                            
                            # Organize data structure
                            if L_value not in training_data:
                                training_data[L_value] = {}
                            
                            training_data[L_value][algorithm] = {
                                'rewards': rewards,
                                'algorithm': algorithm,
                                'L': L_value,
                                'beta_min': params['beta_min'],
                                'steps': len(rewards)
                            }
                            
                            print(f"  ✅ Loaded: {algorithm}_L{L_value} ({len(rewards)} steps)")
                            
                    except Exception as e:
                        print(f"  ❌ Loading failed {file_path}: {e}")
        
        if not training_data:
            print("❌ No valid training results found")
        else:
            print(f"✅ Loading complete, found {sum(len(methods) for methods in training_data.values())} training results")
        
        return training_data
    
    def _parse_directory_name(self, dir_name):
        """Parse directory name to get parameters"""
        try:
            # Format: "Beta_min. = 0.6, K = 4, M = 4, N = 32, P_t = 30.0"
            parts = dir_name.split(', ')
            
            params = {}
            for part in parts:
                if 'Beta_min.' in part:
                    params['beta_min'] = float(part.split('=')[1].strip())
                elif 'N =' in part:
                    params['L'] = int(part.split('=')[1].strip())
                elif 'K =' in part:
                    params['K'] = int(part.split('=')[1].strip())
                elif 'M =' in part:
                    params['M'] = int(part.split('=')[1].strip())
                elif 'P_t =' in part:
                    params['P_t'] = float(part.split('=')[1].strip())
            
            return params if 'L' in params else None
            
        except Exception as e:
            return None
    
    def _parse_algorithm_name(self, filename):
        """Parse filename to get algorithm information"""
        try:
            # Remove .npy extension
            name = filename.replace('.npy', '')
            
            # Supported algorithm patterns
            if name.startswith('SAC_golden_'):
                return {'algorithm': 'SAC_golden'}
            elif name.startswith('SAC_mismatch_'):
                return {'algorithm': 'SAC_mismatch'}
            elif name.startswith('Beta_Space_Exp_SAC_mismatch_'):
                return {'algorithm': 'Beta_Space_Exp_SAC_mismatch'}
            elif name.startswith('DDPG_mismatch_'):
                return {'algorithm': 'DDPG_mismatch'}
            elif name.startswith('PPO_mismatch_'):
                return {'algorithm': 'PPO_mismatch'}
            else:
                return None
                
        except Exception as e:
            return None
    
    def plot_single_L_curves(self, L_data, L_value, save_dir):
        """Plot training curves for single L value"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort algorithms by performance (for legend order)
        algorithm_performance = {}
        for algorithm, data in L_data.items():
            rewards = data['rewards']
            if len(rewards) > 100:
                algorithm_performance[algorithm] = np.mean(rewards[-100:])
        
        sorted_algorithms = sorted(algorithm_performance.items(), key=lambda x: x[1], reverse=True)
        
        for algorithm, _ in sorted_algorithms:
            if algorithm in L_data:
                data = L_data[algorithm]
                rewards = data['rewards']
                
                # Calculate moving average
                window = min(100, len(rewards) // 10)
                if window > 1:
                    smooth_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    x_smooth = np.arange(window-1, len(rewards))
                else:
                    smooth_rewards = rewards
                    x_smooth = np.arange(len(rewards))
                
                # Plot raw data (transparent)
                ax.plot(rewards, alpha=0.2, color=self.algorithm_colors.get(algorithm, '#666666'))
                
                # Plot smooth curve
                label = self.algorithm_labels.get(algorithm, algorithm)
                ax.plot(x_smooth, smooth_rewards, 
                       label=label, 
                       color=self.algorithm_colors.get(algorithm, '#666666'),
                       linewidth=2.5)
        
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Signal Transmission Rate (bps/Hz)', fontsize=12)
        ax.set_title(f'Training Curves Comparison (L = {L_value})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # Add final performance statistics
        stats_text = 'Final Performance (Last 100 steps average):\n'
        for i, (algorithm, final_perf) in enumerate(sorted_algorithms[:3]):  # Show top 3 only
            stats_text += f'{i+1}. {self.algorithm_labels.get(algorithm, algorithm)}: {final_perf:.3f}\n'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
               verticalalignment='top', fontsize=9)
        
        plt.tight_layout()
        
        # Save figure
        filename = f'training_curves_L{L_value}.png'
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  💾 Saved: {filename}")
    
    def plot_all_L_curves(self, training_data, save_dir):
        """Plot comparison chart for all L values"""
        
        L_values = sorted(training_data.keys())
        n_L = len(L_values)
        
        fig, axes = plt.subplots(1, n_L, figsize=(6*n_L, 6))
        
        if n_L == 1:
            axes = [axes]
        
        for i, L_value in enumerate(L_values):
            ax = axes[i]
            L_data = training_data[L_value]
            
            for algorithm, data in L_data.items():
                rewards = data['rewards']
                
                # Calculate moving average
                window = min(100, len(rewards) // 10)
                if window > 1:
                    smooth_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    x_smooth = np.arange(window-1, len(rewards))
                else:
                    smooth_rewards = rewards
                    x_smooth = np.arange(len(rewards))
                
                # Plot smooth curve
                label = self.algorithm_labels.get(algorithm, algorithm)
                ax.plot(x_smooth, smooth_rewards,
                       label=label,
                       color=self.algorithm_colors.get(algorithm, '#666666'),
                       linewidth=2)
            
            ax.set_xlabel('Training Steps', fontsize=10)
            ax.set_ylabel('Signal Transmission Rate (bps/Hz)', fontsize=10)
            ax.set_title(f'L = {L_value}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=8, loc='lower right')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Training Curves Comparison for Different RIS Element Numbers (L)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        filename = 'training_curves_all_L.png'
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  💾 Saved: {filename}")
    
    def plot_performance_summary(self, training_data, save_dir):
        """Generate performance summary heatmap"""
        
        # Calculate final performance
        performance_matrix = {}
        L_values = sorted(training_data.keys())
        all_algorithms = set()
        
        for L_value in L_values:
            L_data = training_data[L_value]
            performance_matrix[L_value] = {}
            
            for algorithm, data in L_data.items():
                rewards = data['rewards']
                if len(rewards) > 100:
                    final_performance = np.mean(rewards[-100:])
                    performance_matrix[L_value][algorithm] = final_performance
                    all_algorithms.add(algorithm)
        
        # Create matrix
        all_algorithms = sorted(list(all_algorithms))
        matrix_data = []
        
        for algorithm in all_algorithms:
            row = []
            for L_value in L_values:
                if algorithm in performance_matrix[L_value]:
                    row.append(performance_matrix[L_value][algorithm])
                else:
                    row.append(np.nan)
            matrix_data.append(row)
        
        matrix_data = np.array(matrix_data)
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(matrix_data, cmap='RdYlGn', aspect='auto')
        
        # Set labels
        ax.set_xticks(range(len(L_values)))
        ax.set_xticklabels([f'L={L}' for L in L_values])
        ax.set_yticks(range(len(all_algorithms)))
        ax.set_yticklabels([self.algorithm_labels.get(alg, alg) for alg in all_algorithms])
        
        # Add value annotations
        for i in range(len(all_algorithms)):
            for j in range(len(L_values)):
                if not np.isnan(matrix_data[i, j]):
                    text = ax.text(j, i, f'{matrix_data[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('Algorithm Performance Summary (Final 100-step Average)', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Performance (bps/Hz)')
        
        plt.tight_layout()
        
        # Save figure
        filename = 'performance_summary.png'
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  💾 Saved: {filename}")
        
        return performance_matrix
    
    def generate_analysis_report(self, training_data, performance_matrix, save_dir):
        """Generate analysis report"""
        
        report_path = os.path.join(save_dir, 'analysis_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("New Algorithm RIS Training Results Analysis Report\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall summary
            total_algorithms = sum(len(methods) for methods in training_data.values())
            f.write(f"Analysis Overview:\n")
            f.write(f"- Total analyzed training results: {total_algorithms}\n")
            f.write(f"- Covering {len(training_data)} different L values: {sorted(training_data.keys())}\n")
            f.write(f"- Including algorithms: {sorted(set(alg for methods in training_data.values() for alg in methods.keys()))}\n\n")
            
            # Performance ranking for each L value
            f.write("Performance Ranking for Each L Value:\n")
            f.write("-" * 30 + "\n")
            
            for L_value in sorted(training_data.keys()):
                f.write(f"\nL = {L_value}:\n")
                
                if L_value in performance_matrix:
                    L_performance = performance_matrix[L_value]
                    sorted_performance = sorted(L_performance.items(), key=lambda x: x[1], reverse=True)
                    
                    for i, (algorithm, performance) in enumerate(sorted_performance):
                        f.write(f"  {i+1}. {self.algorithm_labels.get(algorithm, algorithm)}: {performance:.4f}\n")
                    
                    # Calculate performance gap
                    if len(sorted_performance) > 1:
                        best_perf = sorted_performance[0][1]
                        worst_perf = sorted_performance[-1][1]
                        gap = best_perf - worst_perf
                        f.write(f"  Performance gap: {gap:.4f} ({gap/best_perf*100:.1f}%)\n")
            
            # Overall algorithm performance
            f.write(f"\n\nOverall Algorithm Performance Analysis:\n")
            f.write("-" * 30 + "\n")
            
            algorithm_avg_performance = {}
            algorithm_count = {}
            
            for L_value, L_performance in performance_matrix.items():
                for algorithm, performance in L_performance.items():
                    if algorithm not in algorithm_avg_performance:
                        algorithm_avg_performance[algorithm] = 0
                        algorithm_count[algorithm] = 0
                    algorithm_avg_performance[algorithm] += performance
                    algorithm_count[algorithm] += 1
            
            # Calculate average performance
            for algorithm in algorithm_avg_performance:
                algorithm_avg_performance[algorithm] /= algorithm_count[algorithm]
            
            sorted_avg_performance = sorted(algorithm_avg_performance.items(), key=lambda x: x[1], reverse=True)
            
            for i, (algorithm, avg_performance) in enumerate(sorted_avg_performance):
                f.write(f"{i+1}. {self.algorithm_labels.get(algorithm, algorithm)}: {avg_performance:.4f} (average)\n")
            
            # Convergence analysis
            f.write(f"\n\nConvergence Analysis:\n")
            f.write("-" * 30 + "\n")
            
            for L_value, L_data in training_data.items():
                f.write(f"\nL = {L_value}:\n")
                
                for algorithm, data in L_data.items():
                    rewards = data['rewards']
                    if len(rewards) > 1000:
                        # Calculate convergence steps (simple method: steps to reach 90% of final performance)
                        final_perf = np.mean(rewards[-100:])
                        target_perf = final_perf * 0.9
                        
                        convergence_step = None
                        for i, reward in enumerate(rewards):
                            if reward >= target_perf:
                                convergence_step = i
                                break
                        
                        if convergence_step is not None:
                            f.write(f"  {self.algorithm_labels.get(algorithm, algorithm)}: ~{convergence_step} steps to converge\n")
                        else:
                            f.write(f"  {self.algorithm_labels.get(algorithm, algorithm)}: Not reached stable convergence\n")
        
        print(f"  📄 Analysis report generated: analysis_report.txt")
    
    def analyze_all(self, save_dir='./outputs/analysis/analysis_results_english'):
        """Execute complete analysis"""
        print("🚀 Starting new algorithm training results analysis...")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Load training results
        training_data = self.load_training_results()
        
        if not training_data:
            print("❌ No valid training results found")
            return
        
        print(f"\n📈 Found results for the following configurations:")
        for L_value, methods in training_data.items():
            print(f"  L = {L_value}: {list(methods.keys())}")
        
        print(f"\n🎨 Generating visualization charts...")
        
        # Generate separate training curves for each L value
        for L_value in sorted(training_data.keys()):
            print(f"  📈 Generating training curves for L = {L_value}")
            self.plot_single_L_curves(training_data[L_value], L_value, save_dir)
        
        # Generate comparison chart for all L values
        print(f"  📈 Generating comparison chart for all L values")
        self.plot_all_L_curves(training_data, save_dir)
        
        # Generate performance summary
        print(f"  📊 Generating performance summary heatmap")
        performance_matrix = self.plot_performance_summary(training_data, save_dir)
        
        # Generate analysis report
        print(f"  📄 Generating analysis report")
        self.generate_analysis_report(training_data, performance_matrix, save_dir)
        
        print(f"\n✅ Analysis complete! Results saved in: {save_dir}")
        print(f"📋 Generated files:")
        print(f"  - training_curves_L16.png")
        print(f"  - training_curves_L32.png") 
        print(f"  - training_curves_L64.png")
        print(f"  - training_curves_all_L.png")
        print(f"  - performance_summary.png")
        print(f"  - analysis_report.txt")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Analyze new algorithm RIS training results')
    parser.add_argument('--results_dir', default='./Results', help='Results directory')
    parser.add_argument('--save_dir', default='./outputs/analysis/analysis_results_english', help='Analysis results save directory')
    
    args = parser.parse_args()
    
    # Create analyzer and execute analysis
    analyzer = NewAlgorithmAnalyzer(args.results_dir)
    analyzer.analyze_all(args.save_dir)

if __name__ == "__main__":
    main() 