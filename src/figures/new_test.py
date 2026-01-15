#!/usr/bin/env python3
"""
優化版智慧城市RIS環境生成器 - 合併版本
結合test.py的建築物生成與Voronoi功能 + new_test.py的裝置生成邏輯
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon as MplPolygon
import time
import json
import math
import os
from datetime import datetime
from shapely.geometry import Point, Polygon as ShapelyPolygon, LineString
import networkx as nx
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# 全域字型設定
plt.rcParams["font.sans-serif"] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams["axes.unicode_minus"] = False

# 環境參數
CITY_SIZE = 100  # 100x100公尺
BS_POSITION = (50, 50)  # 基地臺位置（環境中心）
BS_HEIGHT = 15  # 基地臺高度（公尺）
RIS_HEIGHT = 15  # RIS高度（公尺）
RIS_MAX_DISTANCE_TO_BS = 50  # RIS與基地臺最大距離（公尺）
RIS_SERVICE_ANGLE = 60  # RIS服務仰角（度）
RIS_COUNT = 10  # RIS數量
DEVICES_PER_RIS = 4  # 每個RIS服務的裝置數量
TOTAL_DEVICES = 40  # 總裝置數量
MAX_BUILDING_AREA = 240  # 建築物面積上限（平方公尺）

# 計算RIS服務半徑
RIS_SERVICE_RADIUS = RIS_HEIGHT * math.tan(math.radians(RIS_SERVICE_ANGLE / 2)) * 2.2

# 輸出目錄
OUTPUT_DIR = "outputs/dbscan/enhanced_ris_environments"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"合併版環境參數:")
print(f"- 城市大小: {CITY_SIZE}x{CITY_SIZE} 公尺")
print(f"- 基地臺位置: {BS_POSITION}, 高度: {BS_HEIGHT} 公尺")
print(f"- RIS服務半徑: {RIS_SERVICE_RADIUS:.1f} 公尺")
print(f"- 目標配置: {RIS_COUNT} 個RIS, {TOTAL_DEVICES} 個裝置")

def line_segments_intersect(p1, q1, p2, q2):
    """精確檢查兩條線段是否相交"""
    def on_segment(p, q, r):
        return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
    
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(val) < 1e-10:  # 共線
            return 0
        return 1 if val > 0 else 2  # 順時針或逆時針
    
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    
    # 一般情況
    if o1 != o2 and o3 != o4:
        return True
    
    # 特殊情況
    if (o1 == 0 and on_segment(p1, p2, q1)) or \
       (o2 == 0 and on_segment(p1, q2, q1)) or \
       (o3 == 0 and on_segment(p2, p1, q2)) or \
       (o4 == 0 and on_segment(p2, q1, q2)):
        return True
    
    return False

def has_line_of_sight(point1, point2, buildings):
    """檢查兩點之間是否有視距"""
    line = LineString([point1, point2])
    
    for building in buildings:
        try:
            if building[0] == 'rectangle':
                x, y, width, height = building[1]
                rect_edges = [
                    ((x, y), (x + width, y)),
                    ((x + width, y), (x + width, y + height)),
                    ((x + width, y + height), (x, y + height)),
                    ((x, y + height), (x, y))
                ]
                
                for edge in rect_edges:
                    if line_segments_intersect(point1, point2, edge[0], edge[1]):
                        return False
                        
            else:  # 多邊形建築
                polygon_points = building[1]
                n = len(polygon_points)
                for i in range(n):
                    edge_start = polygon_points[i]
                    edge_end = polygon_points[(i + 1) % n]
                    if line_segments_intersect(point1, point2, edge_start, edge_end):
                        return False
        except:
            continue
    
    return True

def generate_smaller_buildings(city_size=100, grid_size=20, building_density=0.7, max_area=240):
    """生成適中大小的建築物 - 使用test.py的建築物生成邏輯"""
    print(f"生成建築物分布（密度{building_density}，最大面積{max_area}）...")
    buildings = []
    building_centers = []
    
    # 使用固定種子確保建築物位置固定
    building_rng = np.random.RandomState(42)
    
    grid_count = city_size // grid_size
    
    for i in range(grid_count):
        for j in range(grid_count):
            # 計算網格中心
            center_x = j * grid_size + grid_size // 2
            center_y = i * grid_size + grid_size // 2
            
            # 避免在基地臺附近生成建築物
            distance_to_bs = math.sqrt((center_x - BS_POSITION[0])**2 + (center_y - BS_POSITION[1])**2)
            if distance_to_bs < 20:  # 基地台周圍的無建築區域
                continue
            
            if building_rng.random() < building_density:
                # 60%多邊形，40%矩形
                if building_rng.random() < 0.6:
                    num_sides = building_rng.randint(3, 7)
                    max_radius = math.sqrt(max_area / math.pi)
                    radius_base = building_rng.uniform(4, min(10, max_radius))
                    
                    angles = np.linspace(0, 2*np.pi, num_sides, endpoint=False)
                    angles += building_rng.uniform(-0.3, 0.3, num_sides)
                    
                    radii = radius_base + building_rng.uniform(-2, 2, num_sides)
                    radii = np.clip(radii, 3, max_radius)
                    
                    polygon_x = center_x + radii * np.cos(angles)
                    polygon_y = center_y + radii * np.sin(angles)
                    
                    polygon_x = np.clip(polygon_x, 2, city_size - 2)
                    polygon_y = np.clip(polygon_y, 2, city_size - 2)
                    
                    polygon_points = list(zip(polygon_x, polygon_y))
                    
                    try:
                        poly_area = ShapelyPolygon(polygon_points).area
                        if poly_area <= max_area:
                            buildings.append(('polygon', polygon_points))
                            building_centers.append((center_x, center_y))
                    except:
                        continue
                        
                else:  # 矩形建築
                    max_side = int(math.sqrt(max_area))
                    width = building_rng.randint(6, min(14, max_side))
                    height = building_rng.randint(6, min(max_area // width, max_side))
                    
                    if width * height <= max_area:
                        x = center_x - width // 2 + building_rng.randint(-3, 4)
                        y = center_y - height // 2 + building_rng.randint(-3, 4)
                        
                        x = max(1, min(x, city_size - width - 1))
                        y = max(1, min(y, city_size - height - 1))
                        
                        buildings.append(('rectangle', (x, y, width, height)))
                        building_centers.append((center_x, center_y))
    
    print(f"生成了 {len(buildings)} 個建築物")
    return buildings, building_centers

def calculate_precise_los_grid(city_size, bs_x, bs_y, buildings):
    """計算視距網格"""
    print("計算視距網格...")
    
    los_grid = np.ones((city_size, city_size), dtype=np.int8)
    
    for i in range(city_size):
        for j in range(city_size):
            if j == bs_x and i == bs_y:
                los_grid[i, j] = 1
                continue
                
            if not has_line_of_sight((j, i), (bs_x, bs_y), buildings):
                los_grid[i, j] = 0
    
    # 統計暗區
    total_points = city_size * city_size
    los_points = np.sum(los_grid)
    nlos_points = total_points - los_points
    nlos_ratio = nlos_points / total_points
    
    print(f"視距計算完成: 亮區 {los_points} 點, 暗區 {nlos_points} 點 ({nlos_ratio:.1%})")
    
    return los_grid

def is_point_in_building(x, y, buildings):
    """檢查點是否在建築物內"""
    for building in buildings:
        if building[0] == 'rectangle':
            rx, ry, rw, rh = building[1]
            if rx <= x < rx + rw and ry <= y < ry + rh:
                return True
        else:
            try:
                if ShapelyPolygon(building[1]).contains(Point(x, y)):
                    return True
            except:
                continue
    return False

class MergedRISEnvironmentGenerator:
    """合併版RIS環境生成器 - 結合兩個檔案的優點"""
    
    def __init__(self):
        self.buildings = None
        self.building_centers = None
        self.los_grid = None
        self.devices = None
        self.ris_positions = None
        self.ris_device_mapping = None
        self.voronoi_diagram = None
        self.voronoi_segments = None
        self.voronoi_segments_full = None
        self.boundary_segments = None
        self.candidate_points = None
        self.uav_path = None
        self.uav_total_distance = None
        self.uav_path_voronoi = None
        self.uav_total_distance_voronoi = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def generate_base_environment(self):
        """建立基礎環境 - 使用test.py的建築物生成"""
        print("=== 步驟1: 建立基礎城市環境 ===")
        
        # 使用test.py的建築物生成
        self.buildings, self.building_centers = generate_smaller_buildings(
            city_size=CITY_SIZE, 
            grid_size=20, 
            building_density=0.7,  # 提高密度（原本17個+2個 ≈ 19個）
            max_area=MAX_BUILDING_AREA
        )
        
        # 計算視距網格
        self.los_grid = calculate_precise_los_grid(CITY_SIZE, BS_POSITION[0], BS_POSITION[1], self.buildings)
        
        print("✅ 基礎環境建立完成")
        return True
    
    def find_optimal_ris_positions(self):
        """尋找最佳RIS位置 - 使用new_test.py的優化邏輯"""
        print("=== 步驟2: 尋找最佳RIS位置 ===")
        
        bs_x, bs_y = BS_POSITION
        
        # 在基地臺範圍內尋找候選位置
        print("搜索RIS候選位置...")
        candidate_positions = []
        
        for angle in np.linspace(0, 2*np.pi, 72):  # 每5度一個方向
            for distance in np.arange(15, RIS_MAX_DISTANCE_TO_BS + 1, 2):  # 每2公尺一個距離
                x = bs_x + distance * np.cos(angle)
                y = bs_y + distance * np.sin(angle)
                
                # 檢查是否在邊界內
                if 5 <= x < CITY_SIZE - 5 and 5 <= y < CITY_SIZE - 5:
                    x, y = int(round(x)), int(round(y))
                    
                    # 檢查RIS位置的有效性
                    if (self.los_grid[y, x] and  # 在亮區
                        not is_point_in_building(x, y, self.buildings) and  # 不在建築物內
                        has_line_of_sight((x, y), BS_POSITION, self.buildings)):  # 與基地臺有視距
                        candidate_positions.append((x, y))
        
        print(f"找到 {len(candidate_positions)} 個有效的RIS候選位置")
        
        if len(candidate_positions) < RIS_COUNT:
            print(f"警告：候選位置不足，僅找到 {len(candidate_positions)} 個位置")
            return candidate_positions[:RIS_COUNT], {}
        
        # 評估每個候選位置的適用性
        print("評估候選位置並選擇最佳組合...")
        position_scores = []
        
        for pos in candidate_positions:
            score = self._evaluate_ris_position(pos)
            position_scores.append((pos, score))
        
        # 按分數排序並選擇最佳位置
        position_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 使用貪心算法選擇分散的最佳位置
        selected_positions = []
        min_distance_between_ris = 15  # RIS之間的最小距離
        
        for pos, score in position_scores:
            if len(selected_positions) >= RIS_COUNT:
                break
                
            # 檢查與已選位置的距離
            too_close = False
            for selected_pos in selected_positions:
                distance = math.sqrt((pos[0] - selected_pos[0])**2 + (pos[1] - selected_pos[1])**2)
                if distance < min_distance_between_ris:
                    too_close = True
                    break
            
            if not too_close:
                selected_positions.append(pos)
        
        print(f"選定 {len(selected_positions)} 個最佳RIS位置")
        
        # 為每個RIS生成裝置
        ris_device_mapping = {}
        all_devices = []
        
        for i, ris_pos in enumerate(selected_positions):
            devices = self._generate_devices_for_ris(ris_pos, DEVICES_PER_RIS)
            if devices:
                ris_device_mapping[i] = devices
                all_devices.extend(devices)
                print(f"RIS-{i+1} at {ris_pos}: 生成 {len(devices)} 個裝置")
            else:
                print(f"RIS-{i+1} at {ris_pos}: 警告，無法生成足夠裝置")
        
        # 如果總裝置數不足40個，嘗試為現有RIS增加更多裝置
        if len(all_devices) < TOTAL_DEVICES:
            print(f"當前裝置總數 {len(all_devices)}，需要補充到 {TOTAL_DEVICES} 個")
            needed_devices = TOTAL_DEVICES - len(all_devices)
            
            # 為每個RIS嘗試添加更多裝置
            for i, ris_pos in enumerate(selected_positions):
                if needed_devices <= 0:
                    break
                    
                additional_devices = self._generate_additional_devices_for_ris(ris_pos, needed_devices)
                if additional_devices:
                    if i in ris_device_mapping:
                        ris_device_mapping[i].extend(additional_devices)
                    else:
                        ris_device_mapping[i] = additional_devices
                    all_devices.extend(additional_devices)
                    needed_devices -= len(additional_devices)
                    print(f"RIS-{i+1}: 補充 {len(additional_devices)} 個裝置")
        
        self.ris_positions = selected_positions
        self.ris_device_mapping = ris_device_mapping
        self.devices = all_devices
        
        print(f"最終生成 {len(all_devices)} 個裝置")
        
        return selected_positions, ris_device_mapping
    
    def _evaluate_ris_position(self, ris_pos):
        """評估RIS位置的適用性分數"""
        score = 0
        
        # 1. 與基地臺的距離（適中距離得分更高）
        distance_to_bs = math.sqrt((ris_pos[0] - BS_POSITION[0])**2 + (ris_pos[1] - BS_POSITION[1])**2)
        optimal_distance = 25  # 最佳距離
        distance_score = max(0, 100 - abs(distance_to_bs - optimal_distance) * 2)
        score += distance_score
        
        # 2. 服務範圍內的暗區面積
        dark_area_score = 0
        service_radius = int(RIS_SERVICE_RADIUS)
        
        for dx in range(-service_radius, service_radius + 1):
            for dy in range(-service_radius, service_radius + 1):
                if dx*dx + dy*dy <= service_radius*service_radius:
                    check_x = ris_pos[0] + dx
                    check_y = ris_pos[1] + dy
                    
                    if (0 <= check_x < CITY_SIZE and 0 <= check_y < CITY_SIZE):
                        # 如果該點在暗區且不在建築物內
                        if (not self.los_grid[check_y, check_x] and 
                            not is_point_in_building(check_x, check_y, self.buildings)):
                            dark_area_score += 1
        
        score += dark_area_score * 2
        
        # 3. 與城市邊界的距離（避免太靠近邊界）
        edge_distance = min(ris_pos[0], ris_pos[1], CITY_SIZE - ris_pos[0], CITY_SIZE - ris_pos[1])
        edge_score = min(edge_distance * 3, 50)
        score += edge_score
        
        return score
    
    def _generate_devices_for_ris(self, ris_position, device_count):
        """為指定RIS生成服務範圍內的裝置"""
        devices = []
        ris_x, ris_y = ris_position
        
        # 創建服務範圍內的候選位置列表
        candidate_positions = []
        service_radius = int(RIS_SERVICE_RADIUS)
        
        # 搜索服務範圍內的暗區位置
        for dx in range(-service_radius, service_radius + 1):
            for dy in range(-service_radius, service_radius + 1):
                distance = math.sqrt(dx*dx + dy*dy)
                if 8 <= distance <= service_radius:  # 最少距離RIS 8公尺
                    check_x = ris_x + dx
                    check_y = ris_y + dy
                    
                    if (3 <= check_x < CITY_SIZE - 3 and 3 <= check_y < CITY_SIZE - 3):
                        # 檢查所有約束條件
                        if (not self.los_grid[check_y, check_x] and  # 在暗區
                            not is_point_in_building(check_x, check_y, self.buildings) and  # 不在建築物內
                            has_line_of_sight(ris_position, (check_x, check_y), self.buildings)):  # 與RIS有視距
                            candidate_positions.append((check_x, check_y))
        
        print(f"  RIS位置 {ris_position} 找到 {len(candidate_positions)} 個候選位置")
        
        if len(candidate_positions) < device_count:
            print(f"  警告：候選位置不足，嘗試放寬約束條件...")
            # 放寬約束條件
            for dx in range(-service_radius - 3, service_radius + 4):
                for dy in range(-service_radius - 3, service_radius + 4):
                    distance = math.sqrt(dx*dx + dy*dy)
                    if 5 <= distance <= service_radius + 3:
                        check_x = ris_x + dx
                        check_y = ris_y + dy
                        
                        if (2 <= check_x < CITY_SIZE - 2 and 2 <= check_y < CITY_SIZE - 2):
                            pos = (check_x, check_y)
                            if pos not in candidate_positions:
                                if (not self.los_grid[check_y, check_x] and
                                    not is_point_in_building(check_x, check_y, self.buildings) and
                                    has_line_of_sight(ris_position, pos, self.buildings)):
                                    candidate_positions.append(pos)
        
        if len(candidate_positions) < device_count:
            print(f"  最終找到 {len(candidate_positions)} 個候選位置，少於所需的 {device_count} 個")
            return candidate_positions
        
        # 隨機選擇分散的位置
        np.random.shuffle(candidate_positions)
        selected_devices = []
        min_device_distance = 4
        
        for pos in candidate_positions:
            if len(selected_devices) >= device_count:
                break
            
            # 檢查與已選裝置的距離
            too_close = False
            for selected_device in selected_devices:
                distance = math.sqrt((pos[0] - selected_device[0])**2 + (pos[1] - selected_device[1])**2)
                if distance < min_device_distance:
                    too_close = True
                    break
            
            if not too_close:
                selected_devices.append(pos)
        
        # 如果還是不夠，直接添加剩餘的候選位置
        if len(selected_devices) < device_count:
            remaining_candidates = [pos for pos in candidate_positions if pos not in selected_devices]
            needed = device_count - len(selected_devices)
            selected_devices.extend(remaining_candidates[:needed])
        
        return selected_devices
    
    def _generate_additional_devices_for_ris(self, ris_position, max_additional):
        """為RIS生成額外的裝置"""
        additional_devices = []
        ris_x, ris_y = ris_position
        
        # 擴大搜索範圍
        extended_radius = int(RIS_SERVICE_RADIUS * 1.3)
        candidate_positions = []
        
        for dx in range(-extended_radius, extended_radius + 1):
            for dy in range(-extended_radius, extended_radius + 1):
                distance = math.sqrt(dx*dx + dy*dy)
                if 5 <= distance <= extended_radius:
                    check_x = ris_x + dx
                    check_y = ris_y + dy
                    
                    if (1 <= check_x < CITY_SIZE - 1 and 1 <= check_y < CITY_SIZE - 1):
                        # 檢查是否已被其他RIS佔用
                        pos = (check_x, check_y)
                        already_used = False
                        if self.ris_device_mapping:
                            for devices_list in self.ris_device_mapping.values():
                                if pos in devices_list:
                                    already_used = True
                                    break
                        
                        if not already_used:
                            if (not self.los_grid[check_y, check_x] and
                                not is_point_in_building(check_x, check_y, self.buildings) and
                                has_line_of_sight(ris_position, pos, self.buildings)):
                                candidate_positions.append(pos)
        
        # 隨機選擇額外的裝置位置
        np.random.shuffle(candidate_positions)
        for pos in candidate_positions[:max_additional]:
            additional_devices.append(pos)
        
        return additional_devices
    
    def generate_voronoi_diagram(self):
        """生成Voronoi圖 - 使用test.py的邏輯"""
        print("=== 步驟3: 生成Voronoi圖 ===")
        
        # 收集所有種子點
        seeds = []
        
        # 建築物中心點
        seeds.extend(self.building_centers)
        
        # 基地台位置
        seeds.append(BS_POSITION)
        
        # 城市邊界的四個角落
        seeds.extend([
            (0, 0),
            (CITY_SIZE, 0), 
            (CITY_SIZE, CITY_SIZE),
            (0, CITY_SIZE)
        ])
        
        # 邊界中點
        seeds.extend([
            (CITY_SIZE/2, 0),
            (CITY_SIZE, CITY_SIZE/2),
            (CITY_SIZE/2, CITY_SIZE),
            (0, CITY_SIZE/2)
        ])
        
        # 外擴邊界的種子點
        margin = 20
        extended_seeds = []
        for i in range(-1, 3):
            for j in range(-1, 3):
                x = i * (CITY_SIZE + margin) / 2
                y = j * (CITY_SIZE + margin) / 2
                if x < -margin/2 or x > CITY_SIZE + margin/2 or y < -margin/2 or y > CITY_SIZE + margin/2:
                    extended_seeds.append((x, y))
        
        # 合併所有種子點
        all_seeds = seeds + extended_seeds
        
        print(f"種子點數量: {len(seeds)} (顯示) + {len(extended_seeds)} (輔助)")
        
        # 生成Voronoi圖
        self.voronoi = Voronoi(np.array(all_seeds))
        
        # 提取線段
        self._extract_voronoi_segments()
        
        print("✅ Voronoi圖生成完成")
        return True
    
    def _extract_voronoi_segments(self):
        """提取Voronoi線段"""
        if not hasattr(self, 'voronoi'):
            return
        
        full_map_segments = []
        candidate_segments = []
        
        def clip_segment_to_city(start, end):
            """將線段裁剪到城市邊界內"""
            x1, y1 = start
            x2, y2 = end
            
            # 裁剪到 [0, CITY_SIZE] 範圍
            if x1 < 0 or x1 > CITY_SIZE or y1 < 0 or y1 > CITY_SIZE or \
               x2 < 0 or x2 > CITY_SIZE or y2 < 0 or y2 > CITY_SIZE:
                
                clipped_start = (max(0, min(CITY_SIZE, x1)), max(0, min(CITY_SIZE, y1)))
                clipped_end = (max(0, min(CITY_SIZE, x2)), max(0, min(CITY_SIZE, y2)))
                
                if math.sqrt((clipped_end[0] - clipped_start[0])**2 + (clipped_end[1] - clipped_start[1])**2) < 1:
                    return None
                    
                return clipped_start, clipped_end
            
            return start, end
        
        # 處理Voronoi脊線
        for ridge_vertices in self.voronoi.ridge_vertices:
            if -1 in ridge_vertices:
                continue
            
            if len(ridge_vertices) >= 2:
                v0 = self.voronoi.vertices[ridge_vertices[0]]
                v1 = self.voronoi.vertices[ridge_vertices[1]]
                
                clipped = clip_segment_to_city(v0, v1)
                if clipped is None:
                    continue
                
                start, end = clipped
                
                # 檢查線段是否被建築物阻擋
                if not self._segment_intersects_buildings(start, end):
                    full_map_segments.append((start, end))
                    
                    # 檢查是否在基地台服務範圍內
                    bs_distance_start = math.sqrt((start[0] - BS_POSITION[0])**2 + (start[1] - BS_POSITION[1])**2)
                    bs_distance_end = math.sqrt((end[0] - BS_POSITION[0])**2 + (end[1] - BS_POSITION[1])**2)
                    
                    if bs_distance_start <= RIS_MAX_DISTANCE_TO_BS or bs_distance_end <= RIS_MAX_DISTANCE_TO_BS:
                        candidate_segments.append((start, end))
        
        self.voronoi_segments_full = full_map_segments
        self.candidate_segments = candidate_segments
        
        print(f"全地圖線段: {len(self.voronoi_segments_full)} 條")
        print(f"基地台範圍線段: {len(self.candidate_segments)} 條")
    
    def _segment_intersects_buildings(self, v0, v1):
        """檢查線段是否與建築物相交"""
        line = LineString([v0, v1])
        
        for building in self.buildings:
            try:
                if building[0] == 'rectangle':
                    x, y, width, height = building[1]
                    building_poly = ShapelyPolygon([(x, y), (x + width, y), 
                                                  (x + width, y + height), (x, y + height)])
                else:
                    building_poly = ShapelyPolygon(building[1])
                
                if line.intersects(building_poly):
                    return True
            except:
                continue
        
        return False
    
    def plan_direct_service_path(self):
        """規劃直線服務路徑"""
        print("=== 步驟4a: 規劃直線服務路徑 ===")
        
        if not self.ris_positions:
            return None, 0
        
        # 使用最近鄰居算法
        start_point = BS_POSITION
        service_points = list(self.ris_positions)
        
        # 計算距離矩陣
        all_points = [start_point] + service_points
        n = len(all_points)
        
        # 最近鄰居算法
        unvisited = set(range(1, n))
        current = 0
        path = [0]
        total_distance = 0
        
        while unvisited:
            distances = []
            for next_point in unvisited:
                p1 = all_points[current]
                p2 = all_points[next_point]
                dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                distances.append((next_point, dist))
            
            nearest = min(distances, key=lambda x: x[1])
            nearest_idx, nearest_dist = nearest
            
            total_distance += nearest_dist
            path.append(nearest_idx)
            unvisited.remove(nearest_idx)
            current = nearest_idx
        
        # 構建路徑
        path_points = [all_points[i] for i in path]
        
        self.uav_path = path_points
        self.uav_total_distance = total_distance
        
        print(f"直線路徑規劃完成: 總距離 {total_distance:.2f} 公尺")
        return path_points, total_distance
    
    def plan_voronoi_service_path(self):
        """基於Voronoi線段規劃服務路徑"""
        print("=== 步驟4b: 規劃Voronoi線段服務路徑 ===")
        
        if not self.ris_positions or not self.voronoi_segments_full:
            print("錯誤：缺少RIS位置或Voronoi線段數據，使用直線路徑")
            return self.plan_direct_service_path()
        
        # 構建Voronoi線段網絡圖
        G = nx.Graph()
        
        # 添加所有線段端點作為節點
        points = set()
        for segment in self.voronoi_segments_full:
            if len(segment) >= 2:
                p1, p2 = tuple(segment[0]), tuple(segment[1])
                points.add(p1)
                points.add(p2)
                distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                G.add_edge(p1, p2, weight=distance)
        
        print(f"Voronoi網絡: {len(points)} 個節點, {len(G.edges)} 條邊")
        
        if len(points) < 2:
            print("Voronoi線段不足，使用直線路徑")
            return self.plan_direct_service_path()
        
        # 添加基地台和RIS位置到圖中
        bs_node = tuple(BS_POSITION)
        G.add_node(bs_node)
        
        all_voronoi_points = list(points)
        
        # 連接基地台到最近的Voronoi點
        if all_voronoi_points:
            closest_to_bs = min(all_voronoi_points, 
                               key=lambda p: math.sqrt((p[0] - bs_node[0])**2 + (p[1] - bs_node[1])**2))
            distance = math.sqrt((closest_to_bs[0] - bs_node[0])**2 + (closest_to_bs[1] - bs_node[1])**2)
            G.add_edge(bs_node, closest_to_bs, weight=distance)
            print(f"基地台連接到最近Voronoi點: 距離 {distance:.1f}公尺")
        
        # 連接每個RIS到最近的Voronoi點
        ris_nodes = []
        for ris_pos in self.ris_positions:
            ris_node = tuple(ris_pos)
            ris_nodes.append(ris_node)
            G.add_node(ris_node)
            
            if all_voronoi_points:
                closest_to_ris = min(all_voronoi_points,
                                    key=lambda p: math.sqrt((p[0] - ris_node[0])**2 + (p[1] - ris_node[1])**2))
                distance = math.sqrt((closest_to_ris[0] - ris_node[0])**2 + (closest_to_ris[1] - ris_node[1])**2)
                G.add_edge(ris_node, closest_to_ris, weight=distance)
        
        # 使用最短路徑連接所有RIS
        try:
            unvisited = set(ris_nodes)
            current_pos = bs_node
            path_segments = []
            total_distance = 0
            
            while unvisited:
                nearest_ris = None
                min_path_length = float('inf')
                
                for ris_node in unvisited:
                    try:
                        if nx.has_path(G, current_pos, ris_node):
                            path_length = nx.dijkstra_path_length(G, current_pos, ris_node, weight='weight')
                            if path_length < min_path_length:
                                min_path_length = path_length
                                nearest_ris = ris_node
                    except nx.NetworkXNoPath:
                        continue
                
                if nearest_ris:
                    path = nx.dijkstra_path(G, current_pos, nearest_ris, weight='weight')
                    path_segments.extend(path)
                    total_distance += min_path_length
                    
                    unvisited.remove(nearest_ris)
                    current_pos = nearest_ris
                else:
                    if unvisited:
                        nearest_ris = min(unvisited, 
                                         key=lambda p: math.sqrt((current_pos[0] - p[0])**2 + (current_pos[1] - p[1])**2))
                        distance = math.sqrt((current_pos[0] - nearest_ris[0])**2 + (current_pos[1] - nearest_ris[1])**2)
                        path_segments.extend([current_pos, nearest_ris])
                        total_distance += distance
                        unvisited.remove(nearest_ris)
                        current_pos = nearest_ris
            
            # 去重連續的路徑點
            unique_path = []
            for point in path_segments:
                if not unique_path or point != unique_path[-1]:
                    unique_path.append(point)
            
            self.uav_path_voronoi = unique_path
            self.uav_total_distance_voronoi = total_distance
            
            print(f"Voronoi路徑規劃完成: 總距離 {total_distance:.2f} 公尺")
            return unique_path, total_distance
                
        except Exception as e:
            print(f"Voronoi路徑規劃失敗: {e}，使用直線路徑")
            return self.plan_direct_service_path()
    
    def visualize_voronoi_system(self, save_path):
        """可視化Voronoi線段網絡系統"""
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_facecolor('#F5F5F5')
        
        # 背景 LOS 網格
        ax.imshow(self.los_grid, cmap='RdYlBu_r', alpha=0.3, 
                 extent=[0, CITY_SIZE, 0, CITY_SIZE], origin='lower')
        
        # 繪製建築物
        for building in self.buildings:
            if building[0] == 'rectangle':
                x, y, w, h = building[1]
                rect = Rectangle((x, y), w, h, fill=True, 
                               facecolor='#8B4513', edgecolor='#654321', linewidth=1.5)
                ax.add_patch(rect)
            else:
                poly = MplPolygon(building[1], fill=True, 
                             facecolor='#8B4513', edgecolor='#654321', linewidth=1.5)
                ax.add_patch(poly)
        
        # 繪製Voronoi線段網絡
        if hasattr(self, 'voronoi_segments_full'):
            print(f"繪製Voronoi圖: {len(self.voronoi_segments_full)} 條線段")
            for segment in self.voronoi_segments_full:
                start, end = segment
                ax.plot([start[0], end[0]], [start[1], end[1]], 
                       'b-', linewidth=1.0, alpha=0.6, label='_nolegend_')
            
            if self.voronoi_segments_full:
                ax.plot([], [], 'b-', linewidth=1.0, alpha=0.6, label='Voronoi線段')
        
        # 繪製基地台
        ax.scatter(BS_POSITION[0], BS_POSITION[1], c='purple', s=200, 
                  marker='s', label='基地台', edgecolors='indigo', linewidth=3, zorder=15)
        
        # 基地台覆蓋範圍
        circle = plt.Circle(BS_POSITION, RIS_MAX_DISTANCE_TO_BS, 
                           fill=False, color='purple', linestyle='--', 
                           linewidth=2, alpha=0.6, label='基地台覆蓋範圍')
        ax.add_patch(circle)
        
        ax.set_xlim(0, CITY_SIZE)
        ax.set_ylim(0, CITY_SIZE)
        ax.set_xlabel('X 座標 (公尺)', fontsize=12)
        ax.set_ylabel('Y 座標 (公尺)', fontsize=12)
        ax.set_title('Voronoi線段網絡系統', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Voronoi線段網絡可視化已保存: {save_path}")
    
    def visualize_complete_system(self, save_path, path_type="direct"):
        """完整系統可視化 - 包含服務範圍和路徑規劃"""
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_facecolor('#F8F8F8')
        
        # 背景LOS網格
        ax.imshow(self.los_grid, cmap='RdYlBu_r', alpha=0.25, 
                 extent=[0, CITY_SIZE, 0, CITY_SIZE], origin='lower')
        
        # 繪製建築物
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
        
        # 繪製裝置
        if self.devices:
            device_x, device_y = zip(*self.devices)
            ax.scatter(device_x, device_y, c='#32CD32', s=60, alpha=0.9, 
                      marker='o', label=f'IoT裝置 ({len(self.devices)})', 
                      edgecolors='#228B22', linewidth=1.5, zorder=10)
        
        # 繪製RIS面板和服務範圍
        if self.ris_positions and self.ris_device_mapping:
            colors = plt.cm.Set3(np.linspace(0, 1, len(self.ris_positions)))
            
            for i, (ris_pos, color) in enumerate(zip(self.ris_positions, colors)):
                # RIS服務範圍圓圈
                service_circle = Circle(ris_pos, RIS_SERVICE_RADIUS, 
                                      fill=False, edgecolor=color, linewidth=2, 
                                      linestyle=':', alpha=0.8, zorder=5, 
                                      label='RIS服務範圍' if i == 0 else '_nolegend_')
                ax.add_patch(service_circle)
                
                # RIS位置
                ax.scatter(ris_pos[0], ris_pos[1], c=[color], s=150, marker='^', 
                          label='RIS面板' if i == 0 else '_nolegend_', 
                          edgecolors='black', linewidth=2, zorder=15)
                
                # 基地臺到RIS的視距線
                ax.plot([BS_POSITION[0], ris_pos[0]], [BS_POSITION[1], ris_pos[1]], 
                       color='red', linewidth=1, alpha=0.5, linestyle='-', zorder=6,
                       label='基地台-RIS連線' if i == 0 else '_nolegend_')
                
                # 該RIS服務的裝置和連線
                if i in self.ris_device_mapping:
                    device_positions = self.ris_device_mapping[i]
                    if device_positions:
                        # 繪製RIS到裝置的服務連線
                        for j, device_pos in enumerate(device_positions):
                            ax.plot([ris_pos[0], device_pos[0]], [ris_pos[1], device_pos[1]], 
                                   color=color, linewidth=1.5, alpha=0.7, zorder=7,
                                   label='RIS服務連線' if i == 0 and j == 0 else '_nolegend_')
        
        # 繪製路徑（根據path_type）
        if path_type == "voronoi" and hasattr(self, 'uav_path_voronoi') and self.uav_path_voronoi:
            # Voronoi路徑
            for i in range(len(self.uav_path_voronoi) - 1):
                start = self.uav_path_voronoi[i]
                end = self.uav_path_voronoi[i + 1]
                ax.plot([start[0], end[0]], [start[1], end[1]], 
                       color='purple', linewidth=3, alpha=0.8, zorder=12)
            
            ax.plot([], [], color='purple', linewidth=3, alpha=0.8, 
                   label=f'UAV Voronoi路徑 ({self.uav_total_distance_voronoi:.1f}m)')
        
        elif hasattr(self, 'uav_path') and self.uav_path:
            # 直線路徑
            for i in range(len(self.uav_path) - 1):
                start = self.uav_path[i]
                end = self.uav_path[i + 1]
                ax.plot([start[0], end[0]], [start[1], end[1]], 
                       color='purple', linewidth=3, alpha=0.8, zorder=12)
            
            ax.plot([], [], color='purple', linewidth=3, alpha=0.8, 
                   label=f'UAV直線路徑 ({self.uav_total_distance:.1f}m)')
        
        # 繪製基地台
        ax.scatter(BS_POSITION[0], BS_POSITION[1], c='#4B0082', s=300, 
                  marker='s', label='基地台', edgecolors='#301934', linewidth=3, zorder=20)
        
        # 基地台覆蓋範圍
        circle = plt.Circle(BS_POSITION, RIS_MAX_DISTANCE_TO_BS, fill=False, 
                           color='#4B0082', linestyle='--', linewidth=2, alpha=0.5)
        ax.add_patch(circle)
        
        ax.set_xlim(0, CITY_SIZE)
        ax.set_ylim(0, CITY_SIZE)
        ax.set_xlabel('X 座標 (公尺)', fontsize=12)
        ax.set_ylabel('Y 座標 (公尺)', fontsize=12)
        
        # 計算統計信息
        total_devices = len(self.devices) if self.devices else 0
        ris_count = len(self.ris_positions) if self.ris_positions else 0
        
        path_suffix = " (Voronoi路徑)" if path_type == "voronoi" else " (直線路徑)"
        
        title = (f'合併版RIS系統部署{path_suffix}\n'
                f'RIS面板: {ris_count}個, IoT裝置: {total_devices}個')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 圖例
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, 
                 facecolor='white', edgecolor='gray', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"完整系統可視化已保存: {save_path}")
    
    def plan_service_path_for_json(self):
        """規劃服務路徑 - 為JSON格式準備（與new_test.py兼容）"""
        if not self.ris_positions:
            return None, None, 0
        
        # 收集所有需要服務的點（所有RIS位置）
        service_points = list(self.ris_positions)
        
        # 使用直線路徑的距離
        total_distance = self.uav_total_distance if hasattr(self, 'uav_total_distance') else 0
        
        # 構建詳細路徑信息（與new_test.py格式一致）
        path_info = [("基地臺", BS_POSITION)]
        if hasattr(self, 'uav_path') and self.uav_path:
            for i, point in enumerate(self.uav_path[1:], 1):  # 跳過起點
                # 找到對應的RIS索引
                ris_idx = -1
                for idx, ris_pos in enumerate(self.ris_positions):
                    if abs(point[0] - ris_pos[0]) < 1 and abs(point[1] - ris_pos[1]) < 1:
                        ris_idx = idx
                        break
                
                if ris_idx >= 0:
                    served_devices = len(self.ris_device_mapping.get(ris_idx, []))
                    path_info.append((f"RIS-{ris_idx+1} (服務{served_devices}個裝置)", point))
                else:
                    path_info.append((f"RIS-{i} (服務未知個裝置)", point))
        
        return path_info, service_points, total_distance
    
    def save_environment_data(self, save_path):
        """儲存環境數據 - 與new_test.py格式一致"""
        print(f"儲存環境數據: {save_path}")
        
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, tuple):
                return list(obj)
            elif isinstance(obj, dict):
                return {str(k): convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        # 計算統計信息
        total_devices = len(self.devices) if self.devices else 0
        devices_in_nlos = sum(1 for d in self.devices if not self.los_grid[d[1]][d[0]]) if self.devices else 0
        nlos_ratio = devices_in_nlos / total_devices if total_devices > 0 else 0
        
        # 規劃服務路徑（為JSON格式準備）
        path_info, service_points, total_distance = self.plan_service_path_for_json()
        
        # 構建數據結構（與new_test.py一致）
        environment_data = {
            "metadata": {
                "generation_time": datetime.now().isoformat(),
                "generator_version": "合併版 3.0",
                "description": "合併版智慧城市RIS環境數據"
            },
            "parameters": {
                "city_size": CITY_SIZE,
                "bs_position": convert_to_serializable(BS_POSITION),
                "bs_height": BS_HEIGHT,
                "ris_height": RIS_HEIGHT,
                "ris_max_distance_to_bs": RIS_MAX_DISTANCE_TO_BS,
                "ris_service_radius": RIS_SERVICE_RADIUS,
                "ris_service_angle": RIS_SERVICE_ANGLE,
                "ris_count": RIS_COUNT,
                "devices_per_ris": DEVICES_PER_RIS,
                "target_total_devices": TOTAL_DEVICES
            },
            "environment": {
                "buildings": convert_to_serializable(self.buildings),
                "los_grid": convert_to_serializable(self.los_grid),
            },
            "deployment": {
                "ris_positions": convert_to_serializable(self.ris_positions) if self.ris_positions else [],
                "devices": convert_to_serializable(self.devices) if self.devices else [],
                "ris_device_mapping": convert_to_serializable(self.ris_device_mapping) if self.ris_device_mapping else {}
            },
            "path_planning": {
                "path_info": [(info, list(pos)) for info, pos in path_info] if path_info else [],
                "total_distance": total_distance,
                "service_points_count": len(service_points) if service_points else 0
            },
            "statistics": {
                "total_buildings": len(self.buildings) if self.buildings else 0,
                "total_ris": len(self.ris_positions) if self.ris_positions else 0,
                "total_devices": total_devices,
                "devices_in_nlos": devices_in_nlos,
                "nlos_ratio": nlos_ratio,
                "average_devices_per_ris": total_devices / len(self.ris_positions) if self.ris_positions else 0,
                "coverage_efficiency": total_devices / TOTAL_DEVICES if TOTAL_DEVICES > 0 else 0
            }
        }
        
        # 儲存JSON檔案
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(environment_data, f, indent=2, ensure_ascii=False)
        
        return environment_data
    
    def run_complete_generation(self):
        """執行完整的環境生成流程"""
        try:
            print("=== 開始合併版環境生成流程 ===")
            
            # 步驟1: 基礎環境
            if not self.generate_base_environment():
                print("❌ 步驟1失敗：基礎環境生成")
                return False
            
            # 步驟2: RIS位置優化
            if not self.find_optimal_ris_positions():
                print("❌ 步驟2失敗：RIS位置優化")
                return False
            
            # 步驟3: Voronoi圖生成
            if not self.generate_voronoi_diagram():
                print("❌ 步驟3失敗：Voronoi圖生成")
                return False
            
            # 步驟4: 路徑規劃
            self.plan_direct_service_path()
            self.plan_voronoi_service_path()
            
            print("=== 合併版環境生成完成 ===")
            return True
            
        except Exception as e:
            print(f"❌ 生成過程中發生錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """主程式"""
    print("=" * 60)
    print("合併版智慧城市RIS環境生成器")
    print("結合test.py建築物生成 + new_test.py裝置生成邏輯")
    print("=" * 60)
    
    generator = MergedRISEnvironmentGenerator()
    timestamp = generator.timestamp
    
    # 定義輸出檔案路徑
    voronoi_ris_system_path = os.path.join(OUTPUT_DIR, f"voronoi_ris_system_{timestamp}.png")
    enhanced_ris_system_direct_path = os.path.join(OUTPUT_DIR, f"enhanced_ris_system_direct_{timestamp}.png")
    enhanced_ris_system_voronoi_path = os.path.join(OUTPUT_DIR, f"enhanced_ris_system_voronoi_{timestamp}.png")
    environment_data_path = os.path.join(OUTPUT_DIR, f"environment_data_{timestamp}.json")
    
    # 執行完整生成
    if not generator.run_complete_generation():
        print("❌ 環境生成失敗")
        return False
    
    # 生成所有需要的可視化圖片
    print("\n=== 生成可視化圖片 ===")
    
    # 1. Voronoi線段網絡圖
    generator.visualize_voronoi_system(voronoi_ris_system_path)
    
    # 2. 完整系統圖 (直線路徑)
    generator.visualize_complete_system(enhanced_ris_system_direct_path, path_type="direct")
    
    # 3. 完整系統圖 (Voronoi線段路徑)
    generator.visualize_complete_system(enhanced_ris_system_voronoi_path, path_type="voronoi")
    
    # 4. 生成環境數據JSON檔案（與new_test.py格式一致）
    print("\n=== 生成環境數據JSON檔案 ===")
    environment_data = generator.save_environment_data(environment_data_path)
    
    # 輸出結果
    print()
    print("=" * 60)
    print("合併版結果總結")
    print("=" * 60)
    
    print(f"✅ 建築物數量: {len(generator.buildings)}")
    print(f"✅ RIS數量: {len(generator.ris_positions) if generator.ris_positions else 0} 個")
    print(f"✅ 總裝置數量: {len(generator.devices) if generator.devices else 0} 個")
    print(f"✅ Voronoi線段: {len(generator.voronoi_segments_full) if hasattr(generator, 'voronoi_segments_full') else 0} 條")
    
    if hasattr(generator, 'uav_total_distance') and generator.uav_total_distance:
        print(f"✅ 直線路徑距離: {generator.uav_total_distance:.1f} 公尺")
    
    if hasattr(generator, 'uav_total_distance_voronoi') and generator.uav_total_distance_voronoi:
        print(f"✅ Voronoi路徑距離: {generator.uav_total_distance_voronoi:.1f} 公尺")
    
    print()
    print("📁 輸出檔案:")
    print(f"  - voronoi_ris_system: {voronoi_ris_system_path}")
    print(f"  - enhanced_ris_system_direct: {enhanced_ris_system_direct_path}")
    print(f"  - enhanced_ris_system_voronoi: {enhanced_ris_system_voronoi_path}")
    print(f"  - environment_data: {environment_data_path}")
    
    # 顯示JSON檔案的關鍵信息
    if environment_data:
        print(f"\n📋 JSON檔案統計:")
        print(f"  - 建築物數量: {environment_data['statistics']['total_buildings']}")
        print(f"  - RIS面板數量: {environment_data['statistics']['total_ris']}")
        print(f"  - IoT裝置數量: {environment_data['statistics']['total_devices']}")
        print(f"  - 暗區比例: {environment_data['statistics']['nlos_ratio']:.1%}")
        print(f"  - 覆蓋效率: {environment_data['statistics']['coverage_efficiency']:.1%}")
    
    return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "="*60)
        print("✅ 合併版環境生成器執行成功！")
        print("所有功能已合併且包含路徑規劃和服務範圍。")
        print("="*60)
    else:
        print("\n❌ 環境生成失敗，請檢查錯誤訊息。")
