#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DBSCAN RIS Environment Visualization (Complete English Version)
Generate English-labeled charts identical to the original Chinese version
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle
from matplotlib.patches import Polygon as MplPolygon
import glob
from datetime import datetime
import math
import warnings
warnings.filterwarnings('ignore')

# Setup English fonts
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

class DBSCANVisualizationEnglishComplete:
    """Complete DBSCAN RIS Environment Visualization (English Version)"""
    
    def __init__(self, environment_data_path):
        """Initialize visualizer with environment data"""
        self.environment_data_path = environment_data_path
        self.load_environment_data()
        
        print(f"🚀 Initializing Complete DBSCAN RIS Visualization (English Version)")
        print(f"📁 Environment data: {os.path.basename(environment_data_path)}")
    
    def load_environment_data(self):
        """Load environment data from JSON file (supports multiple formats)"""
        print("\n📥 Loading environment data...")
        
        with open(self.environment_data_path, 'r', encoding='utf-8') as f:
            self.environment_data = json.load(f)
        
        # Extract data components - handle different JSON formats
        parameters = self.environment_data['parameters']
        
        # Basic environment parameters (common to both formats)
        self.city_size = parameters['city_size']
        self.bs_position = parameters['bs_position']
        self.ris_max_distance_to_bs = parameters['ris_max_distance_to_bs']
        self.ris_service_radius = parameters['ris_service_radius']
        
        # Detect JSON format and load data accordingly
        if 'deployment' in self.environment_data:
            deployment = self.environment_data['deployment']
            
            # Check if it's DBSCAN format or Enhanced format
            if 'detailed_services' in self.environment_data:
                # DBSCAN format with full data
                print("  📋 Detected DBSCAN format with full data")
                self.format_type = 'DBSCAN_Full'
                
                # Load from environment section for buildings and LOS grid
                environment = self.environment_data.get('environment', {})
                self.buildings = environment.get('buildings', [])
                self.los_grid = environment.get('los_grid', None)
                
                # Load detailed services
                detailed_services = self.environment_data['detailed_services']
                self.ris_services = detailed_services.get('ris_services', [])
                self.unconnected_devices = detailed_services.get('unconnected_devices', [])
                
                # Load devices from deployment
                self.devices = deployment.get('devices', [])
                
                # Load path planning data
                path_planning = self.environment_data.get('path_planning', {})
                self.path_info = path_planning.get('path_info', [])
                self.uav_total_distance = path_planning.get('total_distance', 0)
                
                # Convert path_info to uav_path format
                self.uav_path = []
                for path_item in self.path_info:
                    if len(path_item) >= 2:
                        self.uav_path.append(path_item[1])  # Extract coordinates
                
                # Advanced data (might not be available in this format)
                self.candidate_points = []
                self.voronoi_segments = []
                
            elif 'buildings' in deployment:
                # Original DBSCAN format
                print("  📋 Detected original DBSCAN format")
                self.format_type = 'DBSCAN'
                self.buildings = deployment.get('buildings', [])
                self.devices = deployment.get('devices', [])
                self.ris_services = deployment.get('ris_services', [])
                self.unconnected_devices = deployment.get('unconnected_devices', [])
                self.candidate_points = deployment.get('candidate_points', [])
                self.voronoi_segments = deployment.get('voronoi_segments', [])
                self.uav_path = deployment.get('uav_path', [])
                self.uav_total_distance = deployment.get('uav_total_distance', 0)
                self.los_grid = deployment.get('los_grid', None)
                
            else:
                # Enhanced format - adapt to DBSCAN-like structure
                print("  📋 Detected Enhanced format - adapting for visualization")
                self.format_type = 'Enhanced'
                
                # Get buildings from environment section
                environment = self.environment_data.get('environment', {})
                self.buildings = environment.get('buildings', [])
                self.los_grid = environment.get('los_grid', None)
                
                self.devices = deployment.get('devices', [])
                
                # Convert RIS mapping to services format
                self.ris_services = []
                ris_positions = deployment.get('ris_positions', [])
                ris_device_mapping = deployment.get('ris_device_mapping', {})
                
                for ris_id, device_indices in ris_device_mapping.items():
                    if int(ris_id) < len(ris_positions):
                        ris_pos = ris_positions[int(ris_id)]
                        served_devices = [self.devices[i] for i in device_indices if i < len(self.devices)]
                        
                        service = {
                            'ris_position': ris_pos,
                            'cluster_id': int(ris_id),
                            'devices': served_devices,
                            'service_type': 'complete_service'
                        }
                        self.ris_services.append(service)
                
                # Load path planning data
                path_planning = self.environment_data.get('path_planning', {})
                self.path_info = path_planning.get('path_info', [])
                self.uav_total_distance = path_planning.get('total_distance', 0)
                
                # Convert path_info to uav_path format
                self.uav_path = []
                for path_item in self.path_info:
                    if len(path_item) >= 2:
                        self.uav_path.append(path_item[1])  # Extract coordinates
                
                self.unconnected_devices = []
                self.candidate_points = []
                self.voronoi_segments = []
        
        # Convert LOS grid back to numpy array if available
        if self.los_grid is not None:
            self.los_grid = np.array(self.los_grid)
        
        print(f"✅ Successfully loaded environment data ({self.format_type} format):")
        print(f"   🏢 City size: {self.city_size}×{self.city_size} meters")
        print(f"   📡 Base station: {self.bs_position}")
        print(f"   🏠 Buildings: {len(self.buildings)}")
        print(f"   📱 Devices: {len(self.devices)}")
        print(f"   🛰️ RIS services: {len(self.ris_services)}")
        print(f"   ❌ Unconnected devices: {len(self.unconnected_devices)}")
        print(f"   📍 Candidate points: {len(self.candidate_points)}")
        print(f"   📐 Voronoi segments: {len(self.voronoi_segments)}")
        print(f"   🛩️ UAV path points: {len(self.uav_path)}")
        print(f"   📏 UAV total distance: {self.uav_total_distance:.1f}m")
    
    def visualize_complete_system(self, save_path):
        """Generate complete DBSCAN RIS system visualization with English labels - identical to original"""
        print("\n🎨 Generating complete system visualization (English)...")
        
        fig, ax = plt.subplots(figsize=(18, 14))
        ax.set_facecolor('#F8F8F8')
        
        # Background LOS grid (亮暗區顯示)
        if self.los_grid is not None:
            print(f"  🌐 Adding LOS grid background...")
            ax.imshow(self.los_grid, cmap='RdYlBu_r', alpha=0.25, 
                     extent=[0, self.city_size, 0, self.city_size], origin='lower')
        
        # Draw buildings
        for building in self.buildings:
            if building[0] == 'rectangle':
                x, y, w, h = building[1]
                rect = Rectangle((x, y), w, h, fill=True, 
                               facecolor='#696969', edgecolor='#2F4F4F', linewidth=1.5)
                ax.add_patch(rect)
            else:
                poly = MplPolygon(building[1], fill=True, 
                              facecolor='#696969', edgecolor='#2F4F4F', linewidth=1.5)
                ax.add_patch(poly)
        
        # Prepare color mapping for each service instance
        service_colors = plt.cm.Set3(np.linspace(0, 1, len(self.ris_services)))
        
        # Collect all actual device positions
        all_real_devices = set(tuple(device) for device in self.devices)
        
        # Draw unconnected devices
        if self.unconnected_devices:
            uncon_x, uncon_y = zip(*self.unconnected_devices)
            ax.scatter(uncon_x, uncon_y, c='red', s=120, alpha=1.0, 
                      marker='X', linewidth=3, zorder=15,
                      label=f'Unconnected Devices ({len(self.unconnected_devices)})')
        
        # Draw RIS services
        ris_positions_plotted = {}  # Record service count for each position
        
        for i, service in enumerate(self.ris_services):
            ris_pos = service['ris_position']
            cluster_id = service['cluster_id']
            devices = service['devices']
            service_type = service.get('service_type', 'complete_service')
            
            # Get color for this service
            color = service_colors[i % len(service_colors)]
            
            # Record RIS position service count
            ris_key = tuple(ris_pos)
            if ris_key not in ris_positions_plotted:
                ris_positions_plotted[ris_key] = 0
            ris_positions_plotted[ris_key] += 1
            
            # Draw actual service devices
            if devices:
                device_x, device_y = zip(*devices)
                ax.scatter(device_x, device_y, c=[color], s=100, alpha=0.9, 
                          marker='o', edgecolors='black', linewidth=1.5, zorder=12,
                          label=f'Cluster {cluster_id} Actual Devices' if i < 10 else '_nolegend_')
            
            # Service lines from RIS to actual devices
            for device in devices:
                line_style = '-' if 'complete' in service_type else '--'
                line_width = 2.5 if 'complete' in service_type else 2.0
                ax.plot([ris_pos[0], device[0]], [ris_pos[1], device[1]], 
                       color=color, linewidth=line_width, alpha=0.8, linestyle=line_style, zorder=8)
        
        # Draw all actual devices (gray background)
        if all_real_devices:
            real_device_x, real_device_y = zip(*all_real_devices)
            ax.scatter(real_device_x, real_device_y, c='lightgray', s=60, alpha=0.3, 
                      marker='o', edgecolors='gray', linewidth=0.5, zorder=5,
                      label='All Device Positions')
        
        # Draw RIS positions (size based on service count)
        for ris_pos, service_count in ris_positions_plotted.items():
            size = 200 + service_count * 50  # Base size + service count bonus
            ax.scatter(ris_pos[0], ris_pos[1], c='purple', s=size, marker='^', 
                      edgecolors='black', linewidth=2, zorder=20,
                      label='RIS Panels' if ris_pos == list(ris_positions_plotted.keys())[0] else '_nolegend_')
            
            # Mark service count
            if service_count > 1:
                ax.text(ris_pos[0], ris_pos[1] + 2, f'{service_count}', 
                       ha='center', va='bottom', fontweight='bold', fontsize=10, 
                       color='purple', zorder=21)
            
            # RIS service range circles
            service_circle = Circle(ris_pos, self.ris_service_radius, 
                                  fill=False, edgecolor='purple', linewidth=1.5, 
                                  linestyle=':', alpha=0.4, zorder=6)
            ax.add_patch(service_circle)
            
            # Base station to RIS connections
            ax.plot([self.bs_position[0], ris_pos[0]], [self.bs_position[1], ris_pos[1]], 
                   color='red', linewidth=1.5, alpha=0.6, linestyle='-', zorder=7)
        
        # Add RIS service range to legend
        ax.plot([], [], color='purple', linewidth=1.5, linestyle=':', alpha=0.4,
               label=f'RIS Service Range (R={self.ris_service_radius:.0f}m)')
        
        # Draw UAV path (Voronoi路徑規劃)
        if self.uav_path and len(self.uav_path) > 1:
            print(f"  🛩️ Adding UAV path with {len(self.uav_path)} points...")
            for i in range(len(self.uav_path) - 1):
                start = self.uav_path[i]
                end = self.uav_path[i + 1]
                ax.plot([start[0], end[0]], [start[1], end[1]], 
                       color='orange', linewidth=3, alpha=0.8, zorder=11)
            
            ax.plot([], [], color='orange', linewidth=3, alpha=0.8, 
                   label=f'UAV Path ({self.uav_total_distance:.1f}m)')
        
        # Draw base station
        ax.scatter(self.bs_position[0], self.bs_position[1], c='#4B0082', s=400, 
                  marker='s', label='Base Station', edgecolors='#301934', linewidth=3, zorder=25)
        
        # Base station coverage range
        circle = plt.Circle(self.bs_position, self.ris_max_distance_to_bs, fill=False, 
                           color='#4B0082', linestyle='--', linewidth=2, alpha=0.3)
        ax.add_patch(circle)
        
        ax.set_xlim(0, self.city_size)
        ax.set_ylim(0, self.city_size)
        ax.set_xlabel('X Coordinate (meters)', fontsize=12)
        ax.set_ylabel('Y Coordinate (meters)', fontsize=12)
        
        # Calculate statistics
        served_devices = sum(len(service['devices']) for service in self.ris_services)
        total_services = len(self.ris_services)
        unique_ris_positions = len(ris_positions_plotted)
        
        # Title - English version
        title = (f'DBSCAN Smart RIS System Deployment (Shadow Area Devices)\n'
                f'Active Service: {served_devices}/{len(self.devices)} devices, '
                f'RIS Services: {total_services}, RIS Positions: {unique_ris_positions}')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Legend (optimized display)
        handles, labels = ax.get_legend_handles_labels()
        
        # Filter duplicate legend items
        unique_handles = []
        unique_labels = []
        seen_labels = set()
        
        for handle, label in zip(handles, labels):
            if label not in seen_labels:
                unique_handles.append(handle)
                unique_labels.append(label)
                seen_labels.add(label)
        
        ax.legend(unique_handles, unique_labels, loc='center left', bbox_to_anchor=(1.02, 0.5), 
                 frameon=True, facecolor='white', edgecolor='gray', fontsize=10)
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Complete system visualization saved: {save_path}")
        print(f"   📊 Features included:")
        print(f"     - LOS grid background: {'✅' if self.los_grid is not None else '❌'}")
        print(f"     - UAV path planning: {'✅' if self.uav_path else '❌'}")
        print(f"     - Path total distance: {self.uav_total_distance:.1f}m")
        print(f"     - RIS service ranges: ✅")
        print(f"     - BS-RIS connections: ✅")
    
    def visualize_voronoi_candidates_analysis(self, save_path):
        """Generate Voronoi segments and candidate points analysis with English labels"""
        print("\n🎨 Generating Voronoi candidates analysis (English)...")
        
        # Skip if no Voronoi data available
        if not self.voronoi_segments and not self.candidate_points:
            print("  ⚠️ No Voronoi data available - skipping Voronoi analysis")
            return
        
        fig, ax = plt.subplots(figsize=(18, 14))
        ax.set_facecolor('#F8F8F8')
        
        # Background LOS grid (if available)
        if self.los_grid is not None:
            ax.imshow(self.los_grid, cmap='RdYlBu_r', alpha=0.3, 
                    extent=[0, self.city_size, 0, self.city_size], origin='lower')
        
        # Draw buildings
        for building in self.buildings:
            if building[0] == 'rectangle':
                x, y, w, h = building[1]
                rect = Rectangle((x, y), w, h, fill=True, 
                            facecolor='#696969', edgecolor='#2F4F4F', linewidth=2)
                ax.add_patch(rect)
            else:
                poly = MplPolygon(building[1], fill=True, 
                            facecolor='#696969', edgecolor='#2F4F4F', linewidth=2)
                ax.add_patch(poly)
        
        # Draw Voronoi segments network
        if self.voronoi_segments:
            print(f"Drawing {len(self.voronoi_segments)} Voronoi segments")
            for i, segment in enumerate(self.voronoi_segments):
                start, end = segment
                ax.plot([start[0], end[0]], [start[1], end[1]], 
                    'b-', linewidth=1.2, alpha=0.7, zorder=3)
            
            # Add legend item
            ax.plot([], [], 'b-', linewidth=1.2, alpha=0.7, 
                label=f'Voronoi Segments ({len(self.voronoi_segments)})')
        
        # Draw candidate points
        if self.candidate_points:
            print(f"Drawing {len(self.candidate_points)} candidate points")
            candidate_x, candidate_y = zip(*self.candidate_points)
            ax.scatter(candidate_x, candidate_y, c='lime', s=40, alpha=0.8, 
                    marker='s', edgecolors='darkgreen', linewidth=1, zorder=10,
                    label=f'RIS Candidate Points ({len(self.candidate_points)})')
        
        # Draw devices (distinguish served and unserved)
        served_devices = []
        for service in self.ris_services:
            served_devices.extend(service['devices'])
        
        # Convert to tuples for comparison
        served_device_tuples = [tuple(device) for device in served_devices]
        device_tuples = [tuple(device) for device in self.devices]
        unserved_devices = [device for device in device_tuples if device not in served_device_tuples]
        
        if served_devices:
            served_x, served_y = zip(*served_devices)
            ax.scatter(served_x, served_y, c='green', s=60, alpha=0.9, 
                    marker='o', edgecolors='darkgreen', linewidth=1.5, zorder=12,
                    label=f'Served Devices ({len(served_devices)})')
        
        if unserved_devices:
            unserved_x, unserved_y = zip(*unserved_devices)
            ax.scatter(unserved_x, unserved_y, c='red', s=80, alpha=1.0, 
                    marker='X', edgecolors='darkred', linewidth=2, zorder=15,
                    label=f'Unserved Devices ({len(unserved_devices)})')
        
        # Draw selected RIS positions
        if self.ris_services:
            selected_positions = list(set(tuple(service['ris_position']) for service in self.ris_services))
            if selected_positions:
                ris_x, ris_y = zip(*selected_positions)
                ax.scatter(ris_x, ris_y, c='purple', s=150, marker='^', 
                        edgecolors='black', linewidth=2, zorder=20,
                        label=f'Selected RIS Positions ({len(selected_positions)})')
                
                # Draw RIS service ranges
                for ris_pos in selected_positions:
                    service_circle = Circle(ris_pos, self.ris_service_radius, 
                                        fill=False, edgecolor='purple', linewidth=1.5, 
                                        linestyle=':', alpha=0.6, zorder=8)
                    ax.add_patch(service_circle)
        
        # Draw base station
        ax.scatter(self.bs_position[0], self.bs_position[1], c='#4B0082', s=400, 
                marker='s', label='Base Station', edgecolors='#301934', linewidth=3, zorder=25)
        
        # Base station coverage range
        circle = plt.Circle(self.bs_position, self.ris_max_distance_to_bs, fill=False, 
                        color='#4B0082', linestyle='--', linewidth=2, alpha=0.4)
        ax.add_patch(circle)
        
        ax.set_xlim(0, self.city_size)
        ax.set_ylim(0, self.city_size)
        ax.set_xlabel('X Coordinate (meters)', fontsize=12)
        ax.set_ylabel('Y Coordinate (meters)', fontsize=12)
        
        # Statistics
        total_segments = len(self.voronoi_segments)
        total_candidates = len(self.candidate_points)
        
        title = (f'Voronoi Segments and Candidate Points Analysis\n'
                f'Voronoi Segments: {total_segments}, Candidate Points: {total_candidates}, '
                f'Served: {len(served_devices)}/{len(self.devices)} devices')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Legend
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, 
                facecolor='white', edgecolor='gray', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Voronoi candidates analysis saved: {save_path}")
    
    def generate_all_visualizations(self, save_dir='./outputs/dbscan/dbscan_ris_analysis_english_complete'):
        """Generate all English visualizations"""
        print(f"\n🚀 Starting Complete RIS Visualization Generation (English Version)")
        print("=" * 70)
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate complete system visualization
        system_path = os.path.join(save_dir, f'dbscan_ris_system_complete_english_{timestamp}.png')
        self.visualize_complete_system(system_path)
        
        # Generate Voronoi candidates analysis (if data available)
        if self.voronoi_segments or self.candidate_points:
            voronoi_path = os.path.join(save_dir, f'voronoi_candidates_analysis_complete_english_{timestamp}.png')
            self.visualize_voronoi_candidates_analysis(voronoi_path)
        else:
            print("  ⚠️ Skipping Voronoi analysis - no Voronoi data available in this format")
            voronoi_path = None
        
        print("\n" + "=" * 70)
        print(f"✅ Complete RIS Visualization Complete (English Version, {self.format_type} format)!")
        print(f"\n📋 Generated Files:")
        print(f"🔸 Complete System: {os.path.basename(system_path)}")
        if voronoi_path:
            print(f"🔸 Voronoi Analysis: {os.path.basename(voronoi_path)}")
        print(f"\n📁 All files saved to: {save_dir}")
        
        return True

def find_latest_environment_data():
    """Find latest environment data file (prioritize DBSCAN format)"""
    # Search for environment data files (prioritize DBSCAN)
    search_patterns = [
        ("./outputs/dbscan/dbscan_ris_environments/dbscan_environment_data_*.json", "DBSCAN"),
        ("./outputs/dbscan/enhanced_ris_environments/environment_data_*.json", "Enhanced")
    ]
    
    for pattern, format_type in search_patterns:
        files = glob.glob(pattern)
        if files:
            latest_file = max(files, key=os.path.getctime)
            print(f"🔍 Found latest environment data: {os.path.basename(latest_file)} ({format_type} format)")
            return latest_file
    
    print("❌ No environment data files found")
    print("💡 Please run dbscan.py or new_test.py to generate environment data")
    return None

def main():
    """Main function"""
    print("🚀 Complete DBSCAN RIS Environment Visualization (English Version)")
    print("=" * 70)
    print("Generate English-labeled charts identical to the original Chinese version")
    print("- Complete DBSCAN RIS system deployment visualization")
    print("- Voronoi segments and candidate points analysis (if available)")
    print("- All original features preserved (LOS grid, UAV paths, distances)")
    print()
    
    # Find environment data
    environment_data_path = find_latest_environment_data()
    if not environment_data_path:
        return False
    
    # Create visualizer
    visualizer = DBSCANVisualizationEnglishComplete(environment_data_path)
    
    # Generate all visualizations
    success = visualizer.generate_all_visualizations()
    
    if success:
        print("\n🎯 Key Features:")
        print("- Used existing environment data (no regeneration)")
        print("- All labels converted to English")
        print("- Identical to original Chinese version")
        print("- Ready for paper publication")
        print("- Maintains original city layout and device positions")
        print(f"- Compatible with {visualizer.format_type} format")
        print("- Includes LOS grid (light/shadow areas)")
        print("- Includes UAV paths and total distances")
        print("- Includes RIS service ranges and all connections")
    
    return success

if __name__ == "__main__":
    main() 