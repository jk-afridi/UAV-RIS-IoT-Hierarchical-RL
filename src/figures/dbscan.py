#!/usr/bin/env python3
"""
DBSCAN群體化RIS服務系統
基於原有環境架構，使用DBSCAN對80個裝置進行分群，
並根據群體分配RIS服務，支援虛擬裝置補齊機制
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
from sklearn.cluster import DBSCAN
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
TOTAL_DEVICES = 80  # 總裝置數量（增加到80個）
MAX_BUILDING_AREA = 240  # 建築物面積上限（平方公尺）

# DBSCAN參數
DBSCAN_EPS = 8  # 鄰域半徑（公尺）
DBSCAN_MIN_SAMPLES = 1  # 最小樣本數
MAX_CLUSTER_SIZE = 4  # 每群最大裝置數

# 計算RIS服務半徑
RIS_SERVICE_RADIUS = RIS_HEIGHT * math.tan(math.radians(RIS_SERVICE_ANGLE / 2)) * 2.2

# 輸出目錄
OUTPUT_DIR = "outputs/dbscan/dbscan_ris_environments"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"DBSCAN RIS系統參數:")
print(f"- 城市大小: {CITY_SIZE}x{CITY_SIZE} 公尺")
print(f"- 基地臺位置: {BS_POSITION}, 高度: {BS_HEIGHT} 公尺")
print(f"- RIS服務半徑: {RIS_SERVICE_RADIUS:.1f} 公尺")
print(f"- 目標裝置數量: {TOTAL_DEVICES} 個")
print(f"- DBSCAN參數: eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES}")

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
    """生成適中大小的建築物"""
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

class DBSCANRISEnvironmentGenerator:
    """DBSCAN群體化RIS環境生成器"""
    
    def __init__(self):
        self.buildings = None
        self.building_centers = None
        self.los_grid = None
        self.devices = None
        self.device_clusters = None
        self.cluster_groups = None
        self.ris_services = []  # 儲存所有RIS服務實例
        self.candidate_points = None
        self.voronoi_diagram = None
        self.voronoi_segments_full = None
        self.unconnected_devices = []
        self.virtual_devices = []  # 記錄虛擬裝置
        self.uav_path = None
        self.uav_total_distance = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def generate_base_environment(self):
        """建立基礎環境"""
        print("=== 步驟1: 建立基礎城市環境 ===")
        
        # 建築物生成
        self.buildings, self.building_centers = generate_smaller_buildings(
            city_size=CITY_SIZE, 
            grid_size=20, 
            building_density=0.7,
            max_area=MAX_BUILDING_AREA
        )
        
        # 計算視距網格
        self.los_grid = calculate_precise_los_grid(CITY_SIZE, BS_POSITION[0], BS_POSITION[1], self.buildings)
        
        print("✅ 基礎環境建立完成")
        return True
    
    def generate_devices(self):
        """生成80個IoT裝置 - 只在暗區生成（NLOS）"""
        print("=== 步驟2: 生成IoT裝置（僅暗區） ===")
        
        devices = []
        max_attempts = 50000  # 增加嘗試次數
        attempts = 0
        min_distance_to_building = 1.5  # 與建築物的最小距離（公尺）
        min_device_distance = 1.5  # 裝置間最小距離
        
        # 使用固定種子確保可重現性
        device_rng = np.random.RandomState(123)
        
        def distance_to_nearest_building(x, y):
            """計算點到最近建築物的距離"""
            min_dist = float('inf')
            
            for building in self.buildings:
                if building[0] == 'rectangle':
                    bx, by, bw, bh = building[1]
                    # 計算點到矩形的最短距離
                    dx = max(bx - x, 0, x - (bx + bw))
                    dy = max(by - y, 0, y - (by + bh))
                    dist = math.sqrt(dx*dx + dy*dy)
                else:
                    # 多邊形建築 - 計算到邊界的距離
                    try:
                        building_poly = ShapelyPolygon(building[1])
                        point = Point(x, y)
                        if building_poly.contains(point):
                            dist = 0  # 在建築物內部
                        else:
                            dist = point.distance(building_poly)
                    except:
                        dist = 0  # 如果計算失敗，保守起見設為0
                
                min_dist = min(min_dist, dist)
            
            return min_dist
        
        print(f"目標裝置數: {TOTAL_DEVICES} 個（僅暗區）")
        
        # 只在暗區生成裝置 - 漸進放寬約束策略
        distance_thresholds = [min_distance_to_building, min_distance_to_building * 0.7, min_distance_to_building * 0.5]
        device_distance_thresholds = [min_device_distance, min_device_distance * 0.7, min_device_distance * 0.5]
        bs_distance_thresholds = [12, 10, 8]
        
        for round_num, (building_dist, device_dist, bs_dist) in enumerate(zip(distance_thresholds, device_distance_thresholds, bs_distance_thresholds)):
            round_attempts = 0
            max_round_attempts = max_attempts // len(distance_thresholds)
            
            print(f"  回合 {round_num + 1}: 建築物距離≥{building_dist:.1f}m, 裝置間距≥{device_dist:.1f}m, 基地台距離>{bs_dist}m")
            
            while len(devices) < TOTAL_DEVICES and round_attempts < max_round_attempts:
                x = device_rng.randint(2, CITY_SIZE - 2)
                y = device_rng.randint(2, CITY_SIZE - 2)
                
                # ⭐ 嚴格約束：只在暗區生成
                if (not self.los_grid[y, x] and  # ⭐ 必須在暗區（NLOS）
                    not is_point_in_building(x, y, self.buildings) and  # 不在建築物內
                    distance_to_nearest_building(x, y) >= building_dist and  # 與建築物保持距離
                    math.sqrt((x - BS_POSITION[0])**2 + (y - BS_POSITION[1])**2) > bs_dist):  # 距離基地台
                    
                    # 確保與已存在裝置的最小距離
                    too_close = False
                    for existing_device in devices:
                        if math.sqrt((x - existing_device[0])**2 + (y - existing_device[1])**2) < device_dist:
                            too_close = True
                            break
                    
                    if not too_close:
                        devices.append((x, y))
                        if len(devices) % 10 == 0:  # 每10個裝置顯示進度
                            print(f"    已生成 {len(devices)}/{TOTAL_DEVICES} 個裝置...")
                
                round_attempts += 1
                attempts += 1
            
            print(f"  回合 {round_num + 1} 完成: {len(devices)} 個裝置")
            
            if len(devices) >= TOTAL_DEVICES:
                break
        
        self.devices = devices
        print(f"✅ 在暗區生成了 {len(devices)} 個IoT裝置")
        
        if len(devices) < TOTAL_DEVICES:
            print(f"⚠️ 警告：僅生成了 {len(devices)}/{TOTAL_DEVICES} 個裝置")
            print(f"   原因：暗區可用空間限制（建築物密度、距離約束）")
            print(f"   建議：調整建築物密度或放寬約束條件")
        
        # 驗證所有裝置都在暗區
        nlos_count = sum(1 for x, y in devices if not self.los_grid[y, x])
        los_count = len(devices) - nlos_count
        
        print("裝置位置驗證:")
        print(f"  暗區裝置: {nlos_count} 個 ({nlos_count/len(devices):.1%})")
        if los_count > 0:
            print(f"  ❌ 錯誤：發現 {los_count} 個裝置在亮區")
            # 移除亮區的裝置
            self.devices = [(x, y) for x, y in devices if not self.los_grid[y, x]]
            print(f"  ✅ 已移除亮區裝置，剩餘 {len(self.devices)} 個裝置")
        else:
            print(f"  ✅ 所有裝置都在暗區")
        
        return True

    def perform_dbscan_clustering(self):
        """使用DBSCAN對裝置進行分群"""
        print("=== 步驟3: DBSCAN裝置分群 ===")
        
        if not self.devices:
            print("❌ 沒有裝置可以分群")
            return False
        
        # 準備數據
        device_array = np.array(self.devices)
        
        # 執行DBSCAN
        dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
        cluster_labels = dbscan.fit_predict(device_array)
        
        # 分析結果
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"DBSCAN結果: {n_clusters} 個群體, {n_noise} 個噪音點")
        
        # 組織群體
        self.device_clusters = {}
        noise_devices = []
        
        for i, label in enumerate(cluster_labels):
            if label == -1:
                noise_devices.append(self.devices[i])
            else:
                if label not in self.device_clusters:
                    self.device_clusters[label] = []
                self.device_clusters[label].append(self.devices[i])
        
        # 分割過大的群體（超過4個裝置的群體）
        self._split_large_clusters()
        
        # 將噪音點作為單獨群體
        for noise_device in noise_devices:
            # 為噪音點分配新的群體ID
            new_cluster_id = max(self.device_clusters.keys()) + 1 if self.device_clusters else 0
            self.device_clusters[new_cluster_id] = [noise_device]
        
        # ⭐ 添加驗證步驟
        if not self._validate_clusters():
            print("❌ 群體驗證失敗")
            return False
        
        print(f"最終群體數量: {len(self.device_clusters)}")
        for cluster_id, devices in self.device_clusters.items():
            print(f"  群體 {cluster_id}: {len(devices)} 個裝置")
        
        return True

    def _validate_clusters(self):
        """驗證所有群體都不超過最大裝置數"""
        print("驗證群體大小...")
        invalid_clusters = []
        
        for cluster_id, devices in self.device_clusters.items():
            if len(devices) > MAX_CLUSTER_SIZE:
                invalid_clusters.append((cluster_id, len(devices)))
        
        if invalid_clusters:
            print(f"⚠️ 發現 {len(invalid_clusters)} 個超大群體:")
            for cluster_id, size in invalid_clusters:
                print(f"  群體 {cluster_id}: {size} 個裝置")
            return False
        
        print("✅ 所有群體大小都符合限制")
        return True
    
    def _split_large_clusters(self):
        """分割超過4個裝置的群體"""
        clusters_to_split = []
        
        for cluster_id, devices in self.device_clusters.items():
            if len(devices) > MAX_CLUSTER_SIZE:
                clusters_to_split.append(cluster_id)
        
        for cluster_id in clusters_to_split:
            devices = self.device_clusters[cluster_id]
            print(f"分割群體 {cluster_id} ({len(devices)} 個裝置)")
            
            # 刪除原群體
            del self.device_clusters[cluster_id]
            
            # 計算需要的子群體數量
            n_sub_clusters = math.ceil(len(devices) / MAX_CLUSTER_SIZE)
            
            if n_sub_clusters > 1:
                # 使用K-means進一步分割
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_sub_clusters, random_state=42, n_init=10)
                sub_labels = kmeans.fit_predict(np.array(devices))
                
                # 創建新的子群體
                for sub_label in range(n_sub_clusters):
                    sub_devices = [devices[i] for i in range(len(devices)) if sub_labels[i] == sub_label]
                    if sub_devices:
                        # 確保子群體不超過4個裝置
                        while len(sub_devices) > MAX_CLUSTER_SIZE:
                            # 如果還是太大，再次分割
                            split_point = MAX_CLUSTER_SIZE
                            first_part = sub_devices[:split_point]
                            second_part = sub_devices[split_point:]
                            
                            # 添加第一部分
                            new_cluster_id = max(self.device_clusters.keys()) + 1 if self.device_clusters else 0
                            self.device_clusters[new_cluster_id] = first_part
                            print(f"  創建子群體 {new_cluster_id}: {len(first_part)} 個裝置")
                            
                            # 繼續處理剩餘部分
                            sub_devices = second_part
                        
                        # 添加最後剩餘的部分
                        if sub_devices:
                            new_cluster_id = max(self.device_clusters.keys()) + 1 if self.device_clusters else 0
                            self.device_clusters[new_cluster_id] = sub_devices
                            print(f"  創建子群體 {new_cluster_id}: {len(sub_devices)} 個裝置")
            else:
                # 如果只需要一個群體但還是太大，直接按順序分割
                while devices:
                    chunk = devices[:MAX_CLUSTER_SIZE]
                    devices = devices[MAX_CLUSTER_SIZE:]
                    
                    new_cluster_id = max(self.device_clusters.keys()) + 1 if self.device_clusters else 0
                    self.device_clusters[new_cluster_id] = chunk
                    print(f"  創建子群體 {new_cluster_id}: {len(chunk)} 個裝置")

    def generate_voronoi_diagram(self):
        """生成Voronoi圖"""
        print("=== 步驟4: 生成Voronoi圖 ===")
        
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
        
        self.voronoi_segments_full = full_map_segments
        print(f"全地圖線段: {len(self.voronoi_segments_full)} 條")
    
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
    
    def generate_candidate_points(self):
        """生成RIS候選點 - 包含Voronoi端點和亮暗區轉折點"""
        print("=== 步驟5: 生成RIS候選點 ===")
        
        candidate_points = set()
        
        # 1. Voronoi線段端點
        if hasattr(self, 'voronoi_segments_full'):
            for segment in self.voronoi_segments_full:
                start, end = segment
                candidate_points.add((int(round(start[0])), int(round(start[1]))))
                candidate_points.add((int(round(end[0])), int(round(end[1]))))
        
        # 2. 亮暗區邊界轉折點
        boundary_points = self._find_los_boundary_points()
        candidate_points.update(boundary_points)
        
        # 3. 原有亮區採樣點
        sampling_points = self._generate_los_sampling_points()
        candidate_points.update(sampling_points)
        
        # 過濾候選點
        filtered_candidates = []
        for point in candidate_points:
            x, y = point
            if (5 <= x < CITY_SIZE - 5 and 5 <= y < CITY_SIZE - 5 and  # 邊界檢查
                self.los_grid[y, x] and  # 在亮區
                not is_point_in_building(x, y, self.buildings) and  # 不在建築物內
                has_line_of_sight((x, y), BS_POSITION, self.buildings) and  # 與基地臺有視距
                math.sqrt((x - BS_POSITION[0])**2 + (y - BS_POSITION[1])**2) <= RIS_MAX_DISTANCE_TO_BS):  # 在基地臺服務範圍內
                filtered_candidates.append((x, y))
        
        self.candidate_points = filtered_candidates
        print(f"✅ 生成了 {len(filtered_candidates)} 個有效候選點")
        return True
    
    def _find_los_boundary_points(self):
        """找到亮暗區邊界的轉折點"""
        boundary_points = set()
        
        # 檢查每個亮區點周圍是否有暗區點（邊界檢測）
        for y in range(1, CITY_SIZE - 1):
            for x in range(1, CITY_SIZE - 1):
                if self.los_grid[y, x]:  # 亮區點
                    # 檢查8個方向的鄰居
                    neighbors = [
                        (x-1, y-1), (x, y-1), (x+1, y-1),
                        (x-1, y), (x+1, y),
                        (x-1, y+1), (x, y+1), (x+1, y+1)
                    ]
                    
                    has_nlos_neighbor = False
                    for nx, ny in neighbors:
                        if 0 <= nx < CITY_SIZE and 0 <= ny < CITY_SIZE:
                            if not self.los_grid[ny, nx]:  # 有暗區鄰居
                                has_nlos_neighbor = True
                                break
                    
                    if has_nlos_neighbor:  # 這是邊界點
                        boundary_points.add((x, y))
        
        print(f"找到 {len(boundary_points)} 個亮暗區邊界點")
        return boundary_points
    
    def _generate_los_sampling_points(self):
        """在亮區中生成採樣點"""
        sampling_points = set()
        
        # 每5x5網格取一個點
        for y in range(5, CITY_SIZE - 5, 5):
            for x in range(5, CITY_SIZE - 5, 5):
                if self.los_grid[y, x]:  # 在亮區
                    sampling_points.add((x, y))
        
        print(f"生成 {len(sampling_points)} 個亮區採樣點")
        return sampling_points
    
    def assign_ris_services(self):
        """智能RIS服務分配 - 優先完整群體，支持位置重疊，虛擬裝置補齊"""
        print("=== 步驟6: 智能RIS服務分配 ===")
        
        if not self.device_clusters or not self.candidate_points:
            print("❌ 缺少裝置群體或候選點")
            return False
        
        self.ris_services = []
        self.unconnected_devices = []
        self.virtual_devices = []
        assigned_clusters = set()
        
        # 第一階段：優先處理能完全包含4個裝置的群體
        print("\n🎯 第一階段：處理完整群體（4個裝置）")
        complete_clusters = {k: v for k, v in self.device_clusters.items() if len(v) == 4}
        
        for cluster_id, cluster_devices in complete_clusters.items():
            print(f"\n處理完整群體 {cluster_id} ({len(cluster_devices)} 個裝置)...")
            
            # 計算群體中心
            cluster_center = self._calculate_cluster_center(cluster_devices)
            best_candidate = self._find_best_ris_candidate(cluster_devices, cluster_center)
            
            if best_candidate:
                ris_pos, devices_in_range = best_candidate
                
                if len(devices_in_range) == 4:
                    # 完美服務
                    service_instance = {
                        'ris_position': ris_pos,
                        'cluster_id': cluster_id,
                        'devices': devices_in_range,
                        'virtual_devices': [],
                        'service_type': 'complete',
                        'priority': 'high'
                    }
                    self.ris_services.append(service_instance)
                    assigned_clusters.add(cluster_id)
                    print(f"  ✅ 群體 {cluster_id} 完全服務於RIS位置 {ris_pos}")
                else:
                    # 即使是4個裝置的群體，也可能無法全部被一個RIS覆蓋
                    print(f"  ⚠️ 群體 {cluster_id} 只有 {len(devices_in_range)}/4 個裝置在服務範圍內")
        
        # 第二階段：處理剩餘群體（包括不完整群體和未分配的4裝置群體）
        print("\n🔄 第二階段：處理剩餘群體（包括不完整群體）")
        remaining_clusters = {k: v for k, v in self.device_clusters.items() 
                             if k not in assigned_clusters}
        
        for cluster_id, cluster_devices in remaining_clusters.items():
            print(f"\n處理剩餘群體 {cluster_id} ({len(cluster_devices)} 個裝置)...")
            
            cluster_center = self._calculate_cluster_center(cluster_devices)
            best_candidate = self._find_best_ris_candidate(cluster_devices, cluster_center)
            
            if best_candidate:
                ris_pos, devices_in_range = best_candidate
                
                if len(devices_in_range) >= 1:
                    # 需要虛擬裝置補齊到4個
                    virtual_devices = self._generate_virtual_devices(devices_in_range, ris_pos, 
                                                                    4 - len(devices_in_range))
                    
                    service_instance = {
                        'ris_position': ris_pos,
                        'cluster_id': cluster_id,
                        'devices': devices_in_range,
                        'virtual_devices': virtual_devices,
                        'service_type': 'partial' if len(devices_in_range) < len(cluster_devices) else 'complete_with_virtual',
                        'priority': 'normal'
                    }
                    self.ris_services.append(service_instance)
                    assigned_clusters.add(cluster_id)
                    
                    # 記錄虛擬裝置
                    self.virtual_devices.extend(virtual_devices)
                    
                    # 將未在範圍內的裝置標記為未連接
                    for device in cluster_devices:
                        if device not in devices_in_range:
                            self.unconnected_devices.append(device)
                    
                    print(f"  ✅ 群體 {cluster_id} 服務於RIS位置 {ris_pos}")
                    print(f"     實際裝置: {len(devices_in_range)}, 虛擬裝置: {len(virtual_devices)}")
                    print(f"     未連接: {len(cluster_devices) - len(devices_in_range)}")
            else:
                # 完全無法服務的群體
                self.unconnected_devices.extend(cluster_devices)
                print(f"  ❌ 群體 {cluster_id} 無法被任何RIS服務")
        
        # 第三階段：處理孤立的未連接裝置
        print("\n🔍 第三階段：嘗試服務孤立裝置")
        self._handle_isolated_devices()
        
        # 統計最終結果
        self._print_service_statistics()
        
        return True
    
    def _find_best_ris_candidate(self, cluster_devices, cluster_center):
        """為群體找到最佳RIS候選點"""
        suitable_candidates = []
        
        for candidate_pos in self.candidate_points:
            devices_in_range = []
            
            # 檢查有多少個群體裝置在這個候選點的服務範圍內
            for device in cluster_devices:
                distance = math.sqrt((device[0] - candidate_pos[0])**2 + (device[1] - candidate_pos[1])**2)
                if (distance <= RIS_SERVICE_RADIUS and 
                    has_line_of_sight(candidate_pos, device, self.buildings)):
                    devices_in_range.append(device)
            
            if devices_in_range:
                # 計算到群體中心的距離
                distance_to_center = math.sqrt(
                    (candidate_pos[0] - cluster_center[0])**2 + 
                    (candidate_pos[1] - cluster_center[1])**2
                )
                
                # 評分：優先考慮能服務更多裝置的候選點，然後考慮距離
                score = len(devices_in_range) * 1000 - distance_to_center  # 裝置數權重遠高於距離
                suitable_candidates.append((candidate_pos, devices_in_range, score))
        
        if suitable_candidates:
            # 選擇評分最高的候選點
            suitable_candidates.sort(key=lambda x: x[2], reverse=True)
            best_candidate = suitable_candidates[0]
            return best_candidate[0], best_candidate[1]  # 返回位置和服務裝置列表
        
        return None
    
    def _generate_virtual_devices(self, real_devices, ris_position, num_virtual_needed):
        """生成虛擬裝置來補齊到4個裝置"""
        virtual_devices = []
        
        if not real_devices or num_virtual_needed <= 0:
            return virtual_devices
        
        # 策略：使用距離RIS最近的實際裝置位置作為虛擬裝置位置
        devices_with_distance = []
        for device in real_devices:
            distance = math.sqrt((device[0] - ris_position[0])**2 + (device[1] - ris_position[1])**2)
            devices_with_distance.append((device, distance))
        
        devices_with_distance.sort(key=lambda x: x[1])  # 按距離排序
        
        # 重複使用最近的裝置位置生成虛擬裝置
        for i in range(num_virtual_needed):
            # 使用循環方式選擇實際裝置位置
            source_device = devices_with_distance[i % len(devices_with_distance)][0]
            
            # 標記為虛擬裝置
            virtual_device = {
                'position': (source_device[0], source_device[1]),
                'source_real_device': source_device,
                'is_virtual': True,
                'virtual_id': f"v_{len(self.virtual_devices) + i + 1}"
            }
            virtual_devices.append(virtual_device)
        
        return virtual_devices
    
    def _handle_isolated_devices(self):
        """處理孤立的未連接裝置"""
        if not self.unconnected_devices:
            return
        
        print(f"處理 {len(self.unconnected_devices)} 個孤立裝置...")
        
        remaining_unconnected = []
        
        for device in self.unconnected_devices:
            # 為每個孤立裝置尋找候選RIS位置
            best_candidate = self._find_best_ris_candidate([device], device)
            
            if best_candidate:
                ris_pos, devices_in_range = best_candidate
                
                if devices_in_range:  # 能夠服務
                    # 生成3個虛擬裝置補齊到4個
                    virtual_devices = self._generate_virtual_devices(devices_in_range, ris_pos, 3)
                    
                    # 創建新的服務實例，使用獨特的cluster_id
                    new_cluster_id = f"isolated_{len(self.ris_services)}"
                    
                    service_instance = {
                        'ris_position': ris_pos,
                        'cluster_id': new_cluster_id,
                        'devices': devices_in_range,
                        'virtual_devices': virtual_devices,
                        'service_type': 'isolated_with_virtual',
                        'priority': 'low'
                    }
                    self.ris_services.append(service_instance)
                    self.virtual_devices.extend(virtual_devices)
                    
                    print(f"  ✅ 孤立裝置 {device} 已分配服務 (3個虛擬裝置)")
                else:
                    remaining_unconnected.append(device)
            else:
                remaining_unconnected.append(device)
        
        # 更新未連接裝置列表
        self.unconnected_devices = remaining_unconnected
        print(f"  最終未連接裝置: {len(remaining_unconnected)} 個")
    
    def _print_service_statistics(self):
        """輸出詳細的服務統計信息"""
        total_served_devices = sum(len(service['devices']) for service in self.ris_services)
        total_virtual_devices = len(self.virtual_devices)
        total_unconnected = len(self.unconnected_devices)
        unique_ris_positions = len(set(service['ris_position'] for service in self.ris_services))
        
        print(f"\n" + "="*50)
        print("RIS服務分配最終統計")
        print("="*50)
        print(f"📊 總裝置數: {len(self.devices)} 個")
        print(f"📊 總群體數: {len(self.device_clusters)} 個")
        print(f"📊 RIS服務實例: {len(self.ris_services)} 個")
        print(f"📊 RIS物理位置: {unique_ris_positions} 個")
        print(f"📊 實際服務裝置: {total_served_devices} 個")
        print(f"📊 虛擬裝置: {total_virtual_devices} 個")
        print(f"📊 未連接裝置: {total_unconnected} 個")
        print(f"📊 服務效率: {total_served_devices/len(self.devices):.1%}")
        print(f"📊 位置重疊率: {len(self.ris_services)/unique_ris_positions:.2f} 服務/位置")
        
        # 服務類型分析
        service_types = {}
        for service in self.ris_services:
            service_type = service['service_type']
            service_types[service_type] = service_types.get(service_type, 0) + 1
        
        print(f"\n🎯 服務類型分布:")
        for service_type, count in service_types.items():
            print(f"  {service_type}: {count} 個服務實例")
        
        # 驗證總數
        verification_total = total_served_devices + total_unconnected
        print(f"\n✅ 數量驗證: {verification_total} = {len(self.devices)} ({'✓' if verification_total == len(self.devices) else '✗'})")
        
        return True
    
    def _calculate_cluster_center(self, cluster_devices):
        """計算群體中心點"""
        if not cluster_devices:
            return (0, 0)
        
        center_x = sum(device[0] for device in cluster_devices) / len(cluster_devices)
        center_y = sum(device[1] for device in cluster_devices) / len(cluster_devices)
        return (center_x, center_y)
    
    def plan_voronoi_service_path(self):
        """基於Voronoi線段規劃UAV服務路徑"""
        print("=== 步驟7: 規劃Voronoi服務路徑 ===")
        
        if not self.ris_services or not self.voronoi_segments_full:
            print("錯誤：缺少RIS服務或Voronoi線段數據")
            return False
        
        # 收集所有需要服務的RIS位置（可能有重複位置但不同服務）
        service_positions = [service['ris_position'] for service in self.ris_services]
        unique_positions = list(set(service_positions))  # 去重獲得實際停留點
        
        print(f"需要服務的位置: {len(service_positions)} 個服務實例")
        print(f"實際停留位置: {len(unique_positions)} 個")
        
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
            return self._plan_direct_service_path()
        
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
        
        # 連接每個RIS位置到最近的Voronoi點
        ris_nodes = []
        for ris_pos in unique_positions:
            ris_node = tuple(ris_pos)
            ris_nodes.append(ris_node)
            G.add_node(ris_node)
            
            if all_voronoi_points:
                closest_to_ris = min(all_voronoi_points,
                                    key=lambda p: math.sqrt((p[0] - ris_node[0])**2 + (p[1] - ris_node[1])**2))
                distance = math.sqrt((closest_to_ris[0] - ris_node[0])**2 + (closest_to_ris[1] - ris_node[1])**2)
                G.add_edge(ris_node, closest_to_ris, weight=distance)
        
        # 使用最短路徑連接所有RIS位置
        try:
            unvisited = set(ris_nodes)
            current_pos = bs_node
            path_segments = [current_pos]
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
                    
                    # 添加路徑到總路徑（避免重複點）
                    for point in path[1:]:  # 跳過起點避免重複
                        path_segments.append(point)
                    
                    total_distance += min_path_length
                    unvisited.remove(nearest_ris)
                    current_pos = nearest_ris
                else:
                    # 如果無法通過Voronoi網絡到達，使用直線連接
                    if unvisited:
                        nearest_ris = min(unvisited, 
                                         key=lambda p: math.sqrt((current_pos[0] - p[0])**2 + (current_pos[1] - p[1])**2))
                        distance = math.sqrt((current_pos[0] - nearest_ris[0])**2 + (current_pos[1] - nearest_ris[1])**2)
                        path_segments.extend([nearest_ris])
                        total_distance += distance
                        unvisited.remove(nearest_ris)
                        current_pos = nearest_ris
            
            self.uav_path = path_segments
            self.uav_total_distance = total_distance
            
            print(f"✅ Voronoi路徑規劃完成: 總距離 {total_distance:.2f} 公尺")
            print(f"路徑節點數: {len(path_segments)}")
            return True
                
        except Exception as e:
            print(f"Voronoi路徑規劃失敗: {e}，使用直線路徑")
            return self._plan_direct_service_path()
    
    def _plan_direct_service_path(self):
        """規劃直線服務路徑作為備選"""
        if not self.ris_services:
            return False
        
        # 收集所有需要服務的位置
        service_positions = [service['ris_position'] for service in self.ris_services]
        unique_positions = list(set(service_positions))
        
        # 使用最近鄰居算法
        start_point = BS_POSITION
        all_points = [start_point] + unique_positions
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
        
        print(f"✅ 直線路徑規劃完成: 總距離 {total_distance:.2f} 公尺")
        return True
    
    def visualize_complete_system(self, save_path):
        """完整系統可視化 - 支持虛擬裝置和位置重疊服務"""
        fig, ax = plt.subplots(figsize=(18, 14))
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
        
        # 準備顏色映射 - 為每個服務實例分配顏色
        service_colors = plt.cm.Set3(np.linspace(0, 1, len(self.ris_services)))
        
        # 收集所有實際裝置位置（用於繪製）
        all_real_devices = set(self.devices)
        
        # 繪製未連接裝置
        if self.unconnected_devices:
            uncon_x, uncon_y = zip(*self.unconnected_devices)
            ax.scatter(uncon_x, uncon_y, c='red', s=120, alpha=1.0, 
                      marker='X', linewidth=3, zorder=15,
                      label=f'未連接裝置 ({len(self.unconnected_devices)}個)')
        
        # 繪製RIS服務
        ris_positions_plotted = {}  # 記錄每個位置的服務數量
        
        for i, service in enumerate(self.ris_services):
            ris_pos = service['ris_position']
            cluster_id = service['cluster_id']
            devices = service['devices']
            service_type = service['service_type']
            
            # 獲取此服務的顏色
            color = service_colors[i % len(service_colors)]
            
            # 記錄RIS位置的服務數量
            ris_key = tuple(ris_pos)
            if ris_key not in ris_positions_plotted:
                ris_positions_plotted[ris_key] = 0
            ris_positions_plotted[ris_key] += 1
            
            # 繪製實際服務裝置
            if devices:
                device_x, device_y = zip(*devices)
                ax.scatter(device_x, device_y, c=[color], s=100, alpha=0.9, 
                          marker='o', edgecolors='black', linewidth=1.5, zorder=12,
                          label=f'群體 {cluster_id} 實際裝置' if i < 10 else '_nolegend_')
            
            # RIS到實際裝置的服務連線
            for device in devices:
                line_style = '-' if 'complete' in service_type else '--'
                line_width = 2.5 if 'complete' in service_type else 2.0
                ax.plot([ris_pos[0], device[0]], [ris_pos[1], device[1]], 
                       color=color, linewidth=line_width, alpha=0.8, linestyle=line_style, zorder=8)
        
        # 繪製所有實際裝置（灰色背景）
        real_device_x, real_device_y = zip(*all_real_devices)
        ax.scatter(real_device_x, real_device_y, c='lightgray', s=60, alpha=0.3, 
                  marker='o', edgecolors='gray', linewidth=0.5, zorder=5,
                  label='所有裝置位置')
        
        # 繪製RIS位置（根據服務數量調整大小）
        for ris_pos, service_count in ris_positions_plotted.items():
            size = 200 + service_count * 50  # 基礎大小 + 服務數量加成
            ax.scatter(ris_pos[0], ris_pos[1], c='purple', s=size, marker='^', 
                      edgecolors='black', linewidth=2, zorder=20,
                      label='RIS面板' if ris_pos == list(ris_positions_plotted.keys())[0] else '_nolegend_')
            
            # 標記服務數量
            if service_count > 1:
                ax.text(ris_pos[0], ris_pos[1] + 2, f'{service_count}', 
                       ha='center', va='bottom', fontweight='bold', fontsize=10, 
                       color='purple', zorder=21)
            
            # RIS服務範圍圓圈
            service_circle = Circle(ris_pos, RIS_SERVICE_RADIUS, 
                                  fill=False, edgecolor='purple', linewidth=1.5, 
                                  linestyle=':', alpha=0.4, zorder=6)
            ax.add_patch(service_circle)
            
            # 基地臺到RIS的連線
            ax.plot([BS_POSITION[0], ris_pos[0]], [BS_POSITION[1], ris_pos[1]], 
                   color='red', linewidth=1.5, alpha=0.6, linestyle='-', zorder=7)
        
        # 添加RIS服務範圍到圖例
        ax.plot([], [], color='purple', linewidth=1.5, linestyle=':', alpha=0.4,
               label=f'RIS服務範圍 (R={RIS_SERVICE_RADIUS:.0f}m)')
        
        # 繪製UAV路徑
        if self.uav_path:
            for i in range(len(self.uav_path) - 1):
                start = self.uav_path[i]
                end = self.uav_path[i + 1]
                ax.plot([start[0], end[0]], [start[1], end[1]], 
                       color='orange', linewidth=3, alpha=0.8, zorder=11)
            
            ax.plot([], [], color='orange', linewidth=3, alpha=0.8, 
                   label=f'UAV路徑 ({self.uav_total_distance:.1f}m)')
        
        # 繪製基地台
        ax.scatter(BS_POSITION[0], BS_POSITION[1], c='#4B0082', s=400, 
                  marker='s', label='基地台', edgecolors='#301934', linewidth=3, zorder=25)
        
        # 基地台覆蓋範圍
        circle = plt.Circle(BS_POSITION, RIS_MAX_DISTANCE_TO_BS, fill=False, 
                           color='#4B0082', linestyle='--', linewidth=2, alpha=0.3)
        ax.add_patch(circle)
        
        ax.set_xlim(0, CITY_SIZE)
        ax.set_ylim(0, CITY_SIZE)
        ax.set_xlabel('X 座標 (公尺)', fontsize=12)
        ax.set_ylabel('Y 座標 (公尺)', fontsize=12)
        
        # 計算統計信息
        served_devices = sum(len(service['devices']) for service in self.ris_services)
        total_services = len(self.ris_services)
        unique_ris_positions = len(ris_positions_plotted)
        
        title = (f'DBSCAN智能RIS系統部署（暗區裝置）\n'
                f'實際服務: {served_devices}/{len(self.devices)}個裝置, '
                f'RIS服務: {total_services}個, RIS位置: {unique_ris_positions}個')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 圖例（優化顯示）
        handles, labels = ax.get_legend_handles_labels()
        
        # 過濾重複的圖例項目
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
        print(f"完整系統可視化已保存: {save_path}")
    def save_environment_data(self, save_path):
        """儲存環境數據 - 兼容simplified_ris_location_analysis.py格式"""
        print(f"儲存環境數據: {save_path}")
        
        def convert_to_serializable(obj):
            import numpy as np
            
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            elif isinstance(obj, tuple):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {str(k): convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                # 對於任何其他類型，嘗試轉換
                try:
                    if hasattr(obj, 'tolist'):
                        return obj.tolist()
                    elif hasattr(obj, 'item'):
                        return obj.item()
                    else:
                        return str(obj)
                except:
                    return str(type(obj).__name__)
        
        # 統計信息（只計算實際裝置）
        actual_served_devices = sum(len(service['devices']) for service in self.ris_services)
        virtual_device_count = len([vd for service in self.ris_services for vd in service.get('virtual_devices', [])])
        unconnected_count = len(self.unconnected_devices)
        
        # 計算唯一RIS位置
        unique_ris_positions = {}
        for service in self.ris_services:
            pos_key = tuple(service['ris_position'])
            if pos_key not in unique_ris_positions:
                unique_ris_positions[pos_key] = []
            unique_ris_positions[pos_key].append(service)
        
        # 構建兼容simplified_ris_location_analysis.py的數據結構
        ris_positions_list = []
        ris_device_mapping = {}
        
        for i, (ris_pos, services) in enumerate(unique_ris_positions.items()):
            ris_positions_list.append([int(ris_pos[0]), int(ris_pos[1])])
            
            # 收集該位置服務的所有實際裝置（排除虛擬裝置）
            served_devices_at_position = []
            for service in services:
                served_devices_at_position.extend(service['devices'])
            
            # 轉換為設備ID列表格式（假設設備ID就是其在devices列表中的索引）
            device_ids = []
            for device in served_devices_at_position:
                try:
                    device_id = self.devices.index(device)
                    device_ids.append(device_id)
                except ValueError:
                    # 如果找不到設備，可能是數據不一致，跳過
                    pass
            
            ris_device_mapping[i] = device_ids
        
        # 構建所有設備列表（只包含實際設備）
        devices_list = [[int(d[0]), int(d[1])] for d in self.devices]
        
        # 構建路徑信息
        path_info = []
        if self.uav_path:
            path_info.append(["基地臺", [int(BS_POSITION[0]), int(BS_POSITION[1])]])
            for i, point in enumerate(self.uav_path[1:], 1):
                # 檢查這個點是否是RIS位置
                is_ris_position = False
                for ris_idx, ris_pos in enumerate(ris_positions_list):
                    if (abs(point[0] - ris_pos[0]) < 1 and abs(point[1] - ris_pos[1]) < 1):
                        services_count = len(unique_ris_positions[tuple(ris_pos)])
                        path_info.append([f"RIS位置-{ris_idx} (服務{services_count}個群體)", [int(point[0]), int(point[1])]])
                        is_ris_position = True
                        break
                
                if not is_ris_position:
                    path_info.append([f"路徑點-{i}", [int(point[0]), int(point[1])]])
        
        # 構建詳細服務記錄（用於調試和分析）
        detailed_ris_services = []
        for i, service in enumerate(self.ris_services):
            # 處理虛擬裝置信息
            virtual_devices_info = []
            for vd in service.get('virtual_devices', []):
                if isinstance(vd, dict):
                    virtual_devices_info.append({
                        'position': [int(vd['position'][0]), int(vd['position'][1])],
                        'virtual_id': vd.get('virtual_id', f'unknown_{i}'),
                        'is_virtual': True
                    })
                else:
                    # 兼容舊格式
                    virtual_devices_info.append({
                        'position': [int(vd[0]), int(vd[1])],
                        'is_virtual': True
                    })
            
            service_data = {
                'service_id': int(i),
                'ris_position': [int(service['ris_position'][0]), int(service['ris_position'][1])],
                'cluster_id': str(service['cluster_id']),
                'devices': [[int(d[0]), int(d[1])] for d in service['devices']],
                'virtual_devices': virtual_devices_info,
                'service_type': str(service['service_type']),
                'priority': str(service.get('priority', 'normal')),
                'actual_device_count': int(len(service['devices'])),
                'virtual_device_count': int(len(service.get('virtual_devices', [])))
            }
            detailed_ris_services.append(service_data)
        
        # 構建主要數據結構
        environment_data = {
            "metadata": {
                "generation_time": datetime.now().isoformat(),
                "generator_version": "DBSCAN智能RIS系統 2.0",
                "description": "基於DBSCAN群體化的智慧城市RIS環境數據（兼容分析框架）",
                "compatible_with": "simplified_ris_location_analysis.py"
            },
            "parameters": {
                "city_size": int(CITY_SIZE),
                "bs_position": [int(BS_POSITION[0]), int(BS_POSITION[1])],
                "bs_height": int(BS_HEIGHT),
                "ris_height": int(RIS_HEIGHT),
                "ris_max_distance_to_bs": int(RIS_MAX_DISTANCE_TO_BS),
                "ris_service_radius": float(RIS_SERVICE_RADIUS),
                "ris_service_angle": int(RIS_SERVICE_ANGLE),
                "total_devices": int(TOTAL_DEVICES),
                "dbscan_eps": int(DBSCAN_EPS),
                "dbscan_min_samples": int(DBSCAN_MIN_SAMPLES),
                "max_cluster_size": int(MAX_CLUSTER_SIZE),
                "theta_bar": 0.0,
                "kappa_bar": 1.5,
                "AWGN_var": 1e-2,
                "channel_noise_var": 1e-2,
                "seed": 0
            },
            "deployment": {
                "ris_positions": ris_positions_list,
                "devices": devices_list,
                "ris_device_mapping": ris_device_mapping
            },
            "environment": {
                "buildings": convert_to_serializable(self.buildings),
                "los_grid": convert_to_serializable(self.los_grid),
                "candidate_points_count": int(len(self.candidate_points)) if self.candidate_points else 0
            },
            "clustering": {
                "device_clusters": convert_to_serializable(self.device_clusters),
                "cluster_count": int(len(self.device_clusters)) if self.device_clusters else 0,
                "dbscan_parameters": {
                    "eps": int(DBSCAN_EPS),
                    "min_samples": int(DBSCAN_MIN_SAMPLES),
                    "max_cluster_size": int(MAX_CLUSTER_SIZE)
                }
            },
            "detailed_services": {
                "ris_services": detailed_ris_services,
                "total_service_instances": int(len(self.ris_services)),
                "unique_ris_positions": int(len(unique_ris_positions)),
                "unconnected_devices": [[int(d[0]), int(d[1])] for d in self.unconnected_devices],
                "virtual_devices_summary": {
                    "total_virtual_devices": int(virtual_device_count),
                    "note": "虛擬裝置僅用於模型計算，不計入實際服務統計"
                }
            },
            "path_planning": {
                "path_info": path_info,
                "total_distance": float(self.uav_total_distance) if self.uav_total_distance else 0.0,
                "path_points_count": int(len(self.uav_path)) if self.uav_path else 0
            },
            "statistics": {
                "total_buildings": int(len(self.buildings)) if self.buildings else 0,
                "total_devices": int(len(self.devices)) if self.devices else 0,
                "actual_served_devices": int(actual_served_devices),
                "virtual_devices": int(virtual_device_count),
                "unconnected_devices": int(unconnected_count),
                "service_efficiency": float(actual_served_devices / len(self.devices)) if self.devices else 0.0,
                "cluster_count": int(len(self.device_clusters)) if self.device_clusters else 0,
                "average_devices_per_cluster": float(len(self.devices) / len(self.device_clusters)) if self.device_clusters else 0.0,
                "ris_service_instances": int(len(self.ris_services)),
                "unique_ris_positions": int(len(unique_ris_positions)),
                "average_services_per_position": float(len(self.ris_services) / len(unique_ris_positions)) if len(unique_ris_positions) > 0 else 0.0,
                "position_overlap_rate": float(len(self.ris_services) / len(unique_ris_positions)) if len(unique_ris_positions) > 0 else 0.0
            }
        }
        
        # 儲存JSON檔案
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(environment_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 環境數據已儲存")
        print(f"   格式兼容: simplified_ris_location_analysis.py")
        print(f"   實際服務裝置: {actual_served_devices}/{len(self.devices)}")
        print(f"   虛擬裝置: {virtual_device_count} 個（不計入統計）")
        print(f"   RIS位置: {len(unique_ris_positions)} 個")
        print(f"   服務實例: {len(self.ris_services)} 個")
        
        return environment_data

    def run_complete_generation(self):
        """執行完整的DBSCAN RIS環境生成流程"""
        try:
            print("=== 開始DBSCAN RIS環境生成流程 ===")
            
            # 步驟1: 基礎環境
            if not self.generate_base_environment():
                print("❌ 步驟1失敗：基礎環境生成")
                return False
            
            # 步驟2: 生成裝置
            if not self.generate_devices():
                print("❌ 步驟2失敗：裝置生成")
                return False
            
            # 步驟3: DBSCAN分群
            if not self.perform_dbscan_clustering():
                print("❌ 步驟3失敗：DBSCAN分群")
                return False
            
            # 步驟4: Voronoi圖生成
            if not self.generate_voronoi_diagram():
                print("❌ 步驟4失敗：Voronoi圖生成")
                return False
            
            # 步驟5: 候選點生成
            if not self.generate_candidate_points():
                print("❌ 步驟5失敗：候選點生成")
                return False
            
            # 步驟6: RIS服務分配
            if not self.assign_ris_services():
                print("❌ 步驟6失敗：RIS服務分配")
                return False
            
            # 步驟7: 路徑規劃
            if not self.plan_voronoi_service_path():
                print("❌ 步驟7失敗：路徑規劃")
                return False
            
            print("=== DBSCAN RIS環境生成完成 ===")
            return True
            
        except Exception as e:
            print(f"❌ 生成過程中發生錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    def visualize_voronoi_candidates_analysis(self, save_path):
        """可視化Voronoi線段和候選點分析圖"""
        fig, ax = plt.subplots(figsize=(18, 14))
        ax.set_facecolor('#F8F8F8')
        
        # 背景LOS網格
        ax.imshow(self.los_grid, cmap='RdYlBu_r', alpha=0.3, 
                extent=[0, CITY_SIZE, 0, CITY_SIZE], origin='lower')
        
        # 繪製建築物
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
        
        # 繪製Voronoi線段網絡
        if hasattr(self, 'voronoi_segments_full') and self.voronoi_segments_full:
            print(f"繪製 {len(self.voronoi_segments_full)} 條Voronoi線段")
            for i, segment in enumerate(self.voronoi_segments_full):
                start, end = segment
                ax.plot([start[0], end[0]], [start[1], end[1]], 
                    'b-', linewidth=1.2, alpha=0.7, zorder=3)
            
            # 添加圖例項目
            ax.plot([], [], 'b-', linewidth=1.2, alpha=0.7, 
                label=f'Voronoi線段 ({len(self.voronoi_segments_full)}條)')
        
        # 繪製候選點
        if self.candidate_points:
            print(f"繪製 {len(self.candidate_points)} 個候選點")
            candidate_x, candidate_y = zip(*self.candidate_points)
            ax.scatter(candidate_x, candidate_y, c='lime', s=40, alpha=0.8, 
                    marker='s', edgecolors='darkgreen', linewidth=1, zorder=10,
                    label=f'RIS候選點 ({len(self.candidate_points)}個)')
        
        # 繪製裝置（區分已服務和未服務）
        served_devices = []
        for service in self.ris_services:
            served_devices.extend(service['devices'])
        
        unserved_devices = [device for device in self.devices if device not in served_devices]
        
        if served_devices:
            served_x, served_y = zip(*served_devices)
            ax.scatter(served_x, served_y, c='green', s=60, alpha=0.9, 
                    marker='o', edgecolors='darkgreen', linewidth=1.5, zorder=12,
                    label=f'已服務裝置 ({len(served_devices)}個)')
        
        if unserved_devices:
            unserved_x, unserved_y = zip(*unserved_devices)
            ax.scatter(unserved_x, unserved_y, c='red', s=80, alpha=1.0, 
                    marker='X', edgecolors='darkred', linewidth=2, zorder=15,
                    label=f'未服務裝置 ({len(unserved_devices)}個)')
        
        # 繪製已選擇的RIS位置
        if self.ris_services:
            selected_positions = list(set(service['ris_position'] for service in self.ris_services))
            if selected_positions:
                ris_x, ris_y = zip(*selected_positions)
                ax.scatter(ris_x, ris_y, c='purple', s=150, marker='^', 
                        edgecolors='black', linewidth=2, zorder=20,
                        label=f'已選RIS位置 ({len(selected_positions)}個)')
                
                # 繪製RIS服務範圍
                for ris_pos in selected_positions:
                    service_circle = Circle(ris_pos, RIS_SERVICE_RADIUS, 
                                        fill=False, edgecolor='purple', linewidth=1.5, 
                                        linestyle=':', alpha=0.6, zorder=8)
                    ax.add_patch(service_circle)
        
        # 分析未服務裝置周圍的候選點
        if unserved_devices and self.candidate_points:
            print("\n=== 未服務裝置分析 ===")
            for i, unserved_device in enumerate(unserved_devices):
                print(f"未服務裝置 {i+1}: {unserved_device}")
                
                # 找到距離該裝置最近的候選點
                nearest_candidates = []
                for candidate in self.candidate_points:
                    distance = math.sqrt((unserved_device[0] - candidate[0])**2 + 
                                    (unserved_device[1] - candidate[1])**2)
                    if distance <= RIS_SERVICE_RADIUS * 1.2:  # 稍微放寬範圍
                        # 檢查是否有視距
                        has_los = has_line_of_sight(candidate, unserved_device, self.buildings)
                        nearest_candidates.append((candidate, distance, has_los))
                
                nearest_candidates.sort(key=lambda x: x[1])  # 按距離排序
                
                if nearest_candidates:
                    print(f"  附近候選點數量: {len(nearest_candidates)}")
                    for j, (candidate, dist, has_los) in enumerate(nearest_candidates[:3]):  # 顯示最近的3個
                        los_status = "有視距" if has_los else "無視距"
                        print(f"    候選點 {j+1}: {candidate}, 距離: {dist:.1f}m, {los_status}")
                        
                        # 在圖上標記這些關聯
                        ax.plot([unserved_device[0], candidate[0]], [unserved_device[1], candidate[1]], 
                            'orange', linewidth=1, alpha=0.6, linestyle='--', zorder=5)
                else:
                    print(f"  ❌ 服務範圍內無候選點")
        
        # 繪製基地台
        ax.scatter(BS_POSITION[0], BS_POSITION[1], c='#4B0082', s=400, 
                marker='s', label='基地台', edgecolors='#301934', linewidth=3, zorder=25)
        
        # 基地台覆蓋範圍
        circle = plt.Circle(BS_POSITION, RIS_MAX_DISTANCE_TO_BS, fill=False, 
                        color='#4B0082', linestyle='--', linewidth=2, alpha=0.4)
        ax.add_patch(circle)
        
        ax.set_xlim(0, CITY_SIZE)
        ax.set_ylim(0, CITY_SIZE)
        ax.set_xlabel('X 座標 (公尺)', fontsize=12)
        ax.set_ylabel('Y 座標 (公尺)', fontsize=12)
        
        # 統計信息
        total_segments = len(self.voronoi_segments_full) if hasattr(self, 'voronoi_segments_full') else 0
        total_candidates = len(self.candidate_points) if self.candidate_points else 0
        
        title = (f'Voronoi線段與候選點分析\n'
                f'Voronoi線段: {total_segments}條, 候選點: {total_candidates}個, '
                f'已服務: {len(served_devices)}/{len(self.devices)}個裝置')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 圖例
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, 
                facecolor='white', edgecolor='gray', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Voronoi候選點分析圖已保存: {save_path}")

    def print_detailed_candidate_analysis(self):
        """輸出詳細的候選點分析"""
        if not self.candidate_points:
            print("❌ 沒有候選點數據")
            return
        
        print("\n" + "="*60)
        print("詳細候選點分析")
        print("="*60)
        
        # 候選點來源分析
        voronoi_points = set()
        if hasattr(self, 'voronoi_segments_full'):
            for segment in self.voronoi_segments_full:
                start, end = segment
                voronoi_points.add((int(round(start[0])), int(round(start[1]))))
                voronoi_points.add((int(round(end[0])), int(round(end[1]))))
        
        boundary_points = self._find_los_boundary_points()
        sampling_points = self._generate_los_sampling_points()
        
        print(f"候選點來源統計:")
        print(f"  - Voronoi端點: {len(voronoi_points)} 個")
        print(f"  - 亮暗區邊界點: {len(boundary_points)} 個")
        print(f"  - 亮區採樣點: {len(sampling_points)} 個")
        print(f"  - 過濾後有效候選點: {len(self.candidate_points)} 個")
        
        # 候選點分布分析
        candidate_distances = []
        for candidate in self.candidate_points:
            distance = math.sqrt((candidate[0] - BS_POSITION[0])**2 + (candidate[1] - BS_POSITION[1])**2)
            candidate_distances.append(distance)
        
        if candidate_distances:
            print(f"\n候選點距離基地台分析:")
            print(f"  - 最近距離: {min(candidate_distances):.1f} 公尺")
            print(f"  - 最遠距離: {max(candidate_distances):.1f} 公尺")
            print(f"  - 平均距離: {np.mean(candidate_distances):.1f} 公尺")
        
        # 未服務裝置與候選點關係分析
        served_devices = []
        for service in self.ris_services:
            served_devices.extend(service['devices'])
        
        unserved_devices = [device for device in self.devices if device not in served_devices]
        
        if unserved_devices:
            print(f"\n未服務裝置詳細分析:")
            for i, device in enumerate(unserved_devices):
                print(f"\n未服務裝置 {i+1}: 位置 {device}")
                
                # 計算到所有候選點的距離和視距情況
                candidate_analysis = []
                for candidate in self.candidate_points:
                    distance = math.sqrt((device[0] - candidate[0])**2 + (device[1] - candidate[1])**2)
                    has_los = has_line_of_sight(candidate, device, self.buildings)
                    in_range = distance <= RIS_SERVICE_RADIUS
                    candidate_analysis.append((candidate, distance, has_los, in_range))
                
                # 排序並顯示最相關的候選點
                candidate_analysis.sort(key=lambda x: x[1])  # 按距離排序
                
                valid_candidates = [c for c in candidate_analysis if c[2] and c[3]]  # 有視距且在範圍內
                print(f"  有效候選點數量: {len(valid_candidates)}")
                
                if not valid_candidates:
                    print(f"  問題原因分析:")
                    nearby_candidates = [c for c in candidate_analysis[:5]]  # 最近的5個
                    for j, (cand, dist, has_los, in_range) in enumerate(nearby_candidates):
                        status = []
                        if not in_range:
                            status.append(f"距離{dist:.1f}m > 服務半徑{RIS_SERVICE_RADIUS:.1f}m")
                        if not has_los:
                            status.append("無視距")
                        
                        print(f"    候選點 {j+1}: {cand}, 問題: {', '.join(status)}")
def main():
    """主程式"""
    print("=" * 70)
    print("DBSCAN群體化智慧城市RIS環境生成器")
    print("基於DBSCAN對80個IoT裝置進行分群並優化RIS服務分配")
    print("=" * 70)
    
    generator = DBSCANRISEnvironmentGenerator()
    timestamp = generator.timestamp
    
    # 定義輸出檔案路徑
    system_visualization_path = os.path.join(OUTPUT_DIR, f"dbscan_ris_system_{timestamp}.png")
    voronoi_analysis_path = os.path.join(OUTPUT_DIR, f"voronoi_candidates_analysis_{timestamp}.png")  # 新增
    environment_data_path = os.path.join(OUTPUT_DIR, f"dbscan_environment_data_{timestamp}.json")

    # 執行完整生成
    if not generator.run_complete_generation():
        print("❌ 環境生成失敗")
        return False

    # 生成可視化圖片
    print("\n=== 生成可視化圖片 ===")
    generator.visualize_complete_system(system_visualization_path)

    # 新增：生成Voronoi候選點分析圖
    generator.visualize_voronoi_candidates_analysis(voronoi_analysis_path)

    # 新增：輸出詳細候選點分析
    generator.print_detailed_candidate_analysis()
    
    # 生成環境數據JSON檔案
    print("\n=== 生成環境數據JSON檔案 ===")
    environment_data = generator.save_environment_data(environment_data_path)
    
    # 輸出詳細結果
    print()
    print("=" * 70)
    print("DBSCAN RIS系統結果總結")
    print("=" * 70)
    
    # 基礎統計
    print(f"🏢 建築物數量: {len(generator.buildings)}")
    print(f"📱 總裝置數量: {len(generator.devices)} 個")
    print(f"🎯 裝置群體數量: {len(generator.device_clusters)} 個")
    
    # 服務統計
    served_devices = sum(len(service['devices']) for service in generator.ris_services)
    total_services = len(generator.ris_services)
    unique_positions = len(set(service['ris_position'] for service in generator.ris_services))
    virtual_count = len(generator.virtual_devices)
    unconnected_count = len(generator.unconnected_devices)
    
    print(f"📡 RIS服務實例: {total_services} 個")
    print(f"📍 RIS物理位置: {unique_positions} 個")
    print(f"✅ 服務裝置: {served_devices} 個")
    print(f"🔄 虛擬裝置: {virtual_count} 個")
    print(f"❌ 未連接裝置: {unconnected_count} 個")
    
    # 效率統計
    service_efficiency = served_devices / len(generator.devices) if generator.devices else 0
    position_utilization = total_services / unique_positions if unique_positions > 0 else 0
    
    print(f"📊 服務效率: {service_efficiency:.1%}")
    print(f"📊 位置利用率: {position_utilization:.2f} 服務/位置")
    
    # 路徑統計
    if generator.uav_total_distance:
        print(f"✈️ UAV路徑距離: {generator.uav_total_distance:.1f} 公尺")
        print(f"✈️ 路徑節點數: {len(generator.uav_path)} 個")
    
    # 群體分析
    print(f"\n📋 群體分析:")
    cluster_sizes = [len(devices) for devices in generator.device_clusters.values()]
    if cluster_sizes:
        print(f"  - 平均群體大小: {np.mean(cluster_sizes):.1f} 個裝置")
        print(f"  - 最大群體大小: {max(cluster_sizes)} 個裝置")
        print(f"  - 最小群體大小: {min(cluster_sizes)} 個裝置")
        
        # 群體大小分布
        size_distribution = {}
        for size in cluster_sizes:
            size_distribution[size] = size_distribution.get(size, 0) + 1
        
        print(f"  - 群體大小分布:")
        for size in sorted(size_distribution.keys()):
            count = size_distribution[size]
            print(f"    {size}個裝置: {count} 個群體")
    
    # 服務類型統計
    complete_services = sum(1 for service in generator.ris_services if service['service_type'] == 'complete')
    partial_services = sum(1 for service in generator.ris_services if service['service_type'] == 'partial')
    
    print(f"\n🎯 服務類型分析:")
    print(f"  - 完全服務群體: {complete_services} 個")
    print(f"  - 部分服務群體: {partial_services} 個")
    
    print()
    print("📁 輸出檔案:")
    print(f"  - 系統可視化: {system_visualization_path}")
    print(f"  - 環境數據: {environment_data_path}")
    
    # 顯示JSON檔案的關鍵信息
    if environment_data:
        print(f"\n📋 JSON檔案統計:")
        stats = environment_data['statistics']
        print(f"  - 總裝置數: {stats['total_devices']}")
        print(f"  - 實際服務裝置數: {stats['actual_served_devices']}")
        print(f"  - 服務效率: {stats['service_efficiency']:.1%}")
        print(f"  - 群體數量: {stats['cluster_count']}")
        print(f"  - RIS服務實例: {stats['ris_service_instances']}")
        print(f"  - 實際RIS位置: {stats['unique_ris_positions']}")
        print(f"  - 虛擬裝置: {stats['virtual_devices']}")
        print(f"  - 未連接裝置: {stats['unconnected_devices']}")
        print(f"  - 位置重疊率: {stats['position_overlap_rate']:.2f}")
    
    return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "="*70)
        print("✅ DBSCAN RIS環境生成器執行成功！")
        print("已完成裝置分群、RIS服務分配與路徑規劃。")
        print("支援虛擬裝置補齊機制與重複位置多群體服務。")
        print("="*70)
    else:
        print("\n❌ 環境生成失敗，請檢查錯誤訊息。")