# UAV‑RIS DRL 實驗專案（整理版）

本專案為 WCNC 2026 論文與中文論文之實驗重整版本，只保留 **模型訓練** 與 **產生論文圖片** 相關程式。評委回覆所需新增實驗與報告也已整理在固定位置。

## 目錄結構

```
.
├─ src/                      # 主要程式（訓練與產圖）
│  ├─ algorithms/            # SAC / DDPG / PPO / Beta‑Space 等演算法
│  ├─ core/                  # 環境與共用工具
│  ├─ train/                 # 訓練入口
│  └─ figures/               # 產圖與評估腳本
├─ Models/                   # 模型權重（Git 會忽略檔案內容）
├─ Results/                  # 訓練結果 .npy
├─ outputs/                  # 產圖與分析輸出
│  ├─ analysis/              # 訓練曲線等分析
│  ├─ ris_analysis/          # RIS 位置/效能比較圖
│  ├─ evaluation/            # 評估結果
│  ├─ dbscan/                # DBSCAN/Voronoi 環境輸出
│  └─ review/                # 評委回覆新增實驗
├─ experiments/review/       # 評委回覆新增實驗腳本
├─ reports/reviewer_response # 評委回覆 HTML/PDF 與圖片
├─ docs/papers               # 論文 PDF
└─ requirements.txt
```

## 環境安裝

建議使用 Python 3.10+。

1. 建立虛擬環境（選用）
2. 安裝基本依賴：

```bash
pip install -r requirements.txt
```

3. 產圖與環境生成常用套件（如尚未安裝）：

```bash
pip install matplotlib seaborn tqdm scipy scikit-learn shapely networkx
```

## 訓練模型

請在專案根目錄執行：

### SAC / Beta‑Space SAC 訓練

```bash
python src/train/main.py --policy Beta_Space_Exp_SAC --num_RIS_elements 64 --beta_min 0.6
```

### DDPG / PPO 訓練

```bash
python src/train/train_new_algorithms.py --algorithm DDPG --num_RIS_elements 64 --beta_min 0.6
```

### 批次訓練

```bash
python src/train/batch_training.py
python src/train/train_all_new_models.py
```

訓練輸出：
- 權重：`Models/`
- 訓練曲線數據：`Results/`

## 產生論文圖片（重點腳本）

### 訓練收斂曲線

```bash
python src/figures/analyze_new_results_english.py --results_dir ./Results
```

輸出：`outputs/analysis/analysis_results_english/`

### DBSCAN + Voronoi 環境與可視化

```bash
python src/figures/dbscan.py
python src/figures/dbscan_visualization_english_complete.py
```

輸出：`outputs/dbscan/`

### RIS 位置效能比較（L=16/32/64）

```bash
python src/figures/simplified_ris_location_analysis_english.py
```

輸出：`outputs/ris_analysis/modified_ris_analysis_english/`

## 評委回覆新增實驗

### 虛擬用戶比例影響

```bash
python experiments/review/virtual_user_ratio_experiment.py --model_path "Models/..."
```

輸出：`outputs/review/virtual_user_ratio/`

### 可擴展性延遲（CPU）

```bash
python experiments/review/scalability_timing.py
```

輸出：`outputs/review/scalability/`

## 評委回覆報告

HTML / PDF 報告位置：
- `reports/reviewer_response/reviewer_response_cn.html`
- `reports/reviewer_response/reviewer_response_cn.pdf`

若要重新輸出 PDF（需 Playwright）：

```bash
pip install playwright
python -m playwright install chromium
```

## 注意事項

- 執行腳本請**從專案根目錄**啟動，以確保相對路徑正確。
- `Models/` 內權重已被 `.gitignore` 忽略，但資料夾會保留於 Git。
- 論文 PDF 位置：`docs/papers/中文完成.pdf`、`docs/papers/paper_en_journal.pdf`。
