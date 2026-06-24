# 病房模擬到真實 (Hospital Ward Sim‑to‑Real)

針對**病房物件偵測／實例分割**的 sim‑to‑real（模擬到真實）流程。
目標：在 **Isaac Sim** 中渲染合成病房，將渲染結果轉換成**符合特定真實病房外觀的擬真影像**，
並在這份「已風格化但仍保有標註」的資料上訓練偵測器，使其能轉移到真實手持網路攝影機拍攝的照片。

風格轉換使用 **NVIDIA Cosmos‑Transfer2.5**（world‑foundation model）。此外，本專案也發展出一條
**領域無關（domain‑agnostic）偵測器訓練**路線（見「領域無關訓練器」一節），直接同時用模擬與少量
**有標註的真實資料**聯合訓練——目前最成功、實際把領域落差（domain gap）收斂的做法。

```
Isaac Sim 渲染 ──► RGB + GT 深度 + 實例分割 + COCO/YOLO 標註   (src/sim/replicator.py)
        │
        ├─ seg / GT 深度 / edge ─► Cosmos controls（控制訊號）
        ├─ guided foreground mask（錨定有標註的物件）
        └─ style ref = 依物件清單比對出最相近的真實照片
                          ▼
            Cosmos‑Transfer2.5 ──► 擬真、符合本病房風格、且標註對齊的影格
                          │            (src/cosmos/jobs.py → cosmos_jobs/run_all.sh)
                          ▼
            訓練偵測器（YOLO）— 兩種路線：
              (a) 在風格化的模擬資料上訓練
              (b) 領域無關：聯合 模擬 + 有標註真實(real_dev) 訓練   (src/training/train_yolo.py)
                          ▼
            在**真實 holdout** 上評估；領域落差以 DINOv2 / MMD 特徵空間量測
```

## 專案結構

所有 Python 程式碼集中於 `src/`，依職責分成子模組。進入點腳本（renderer、trainer、CLI）會
在啟動時把 `src/` 加入 `sys.path`，因此可直接以檔案路徑執行（`python src/<pkg>/<mod>.py`），
無需安裝。

```
src/
  layout/      佈局生成：物件擺放系統（見「佈局生成」一節）
    affordances.py   每個物件類別「附著在什麼上」的明確宣告 + 表面等價類（host equivalence）
    slots.py         以資料驅動的 slot 規格 populate / dump（slot_layout）
    engine.py        幾何基元 + 約束情境 build_ctx + 接地檢查 + 舊版抖動取樣器（原 placement_dr）
    solver.py        程序化約束求解器（zones + 硬約束 + 模擬退火）— 目前的擺放方法
  sim/         Isaac Sim 渲染與 GT
    replicator.py    主渲染器（原 replicator_dataset.py）
    labels.py        修正 COCO 標註中誤標的 Isaac 資產
    gt_overlays.py / raw_overlays.py   GT 遮罩疊圖
    inspect_semantics.py / inspect_stage.py / debug_campose.py   場景語意與相機檢查
  cosmos/      Cosmos‑Transfer2.5
    jobs.py          產生每張影格的 Cosmos 工作設定（原 gen_cosmos_jobs.py）
  training/    模型訓練與推論
    build_dataset.py  渲染 → COCO/YOLO 一條龍
    train_yolo.py / predict_yolo.py        YOLO 偵測／分割（含領域無關訓練）
    train_seg_detr.py / predict_seg_detr.py  Mask2Former 分割器
  eval/
    domain_gap.py    量測 sim→real 分佈落差（原 measure_domain_gap.py）
  common/
    categories.py    43 類病房物件的 FIXED_CATEGORIES + SUPERCATEGORY_TREE（原 fixed_categories.py）
tests/
  test_layout.py     佈局求解器的純 Python 可信度測試（無需 Isaac）
```

`ROS2_bridge/src/fixed_categories.py` 現為相容 shim，轉接到 `src/common/categories.py`。

## 佈局生成（`src/layout/`）

把合成病房的「擺放」從手寫啟發式，改為**程序化約束擺放**（Infinigen 風格，但以純 Python 在
Isaac USD 上自行實作、不依賴 Blender）。核心觀念：

- **明確 affordance（`affordances.py`）**：每個類別宣告它如何被支撐——
  `floor`（放地上）/ `wall`（貼牆）/ `surface`（放在某表面上）/ `fixed`（固定件），
  以及 `provides_surface`（其頂面能否放東西）。並有**表面等價類**：能放在床上的小物也能放在
  床頭桌或跨床桌上（`HOST_GROUPS`）。取代了易出錯的 z 門檻推論。
- **Zones＝自由度**：每個物件被約束在一個可行區域內——牆面物件沿其**真實牆面**1D 滑動（貼齊、
  限制在所屬房間內）；地面物件在房間地板上 2D 移動（以牆面圍束）；表面小物在 host 頂面 2D；
  固定件不動。**附著於表面，沿表面滑動**。
- **硬約束＝拒絕**：重疊、穿牆、跨牆（跑到別的房間）一律不可——任何造成違規的擺放或移動直接被
  拒絕，因此構型**永遠可行**（物件不會互相重疊）。
- **程序化逐一擺放**：先放**支撐表面**（床、床頭桌、檯面），因為小物要放在它們上面，再逐一加入
  其餘家具到空位；之後再把小物放到已定位的 host 上。
- **模擬退火**只負責**軟性偏好**（床貼牆、衛星物件靠近床），在可行集合內微調。

求解器（`solver.py`）是 `engine.generate` 的 drop‑in 替代（相同的 `{path: [(x,y,z)]}` 介面）。
舊版手寫抖動取樣器仍保留在 `engine.py`（亦提供 `build_ctx` / `validate_grounding` / 幾何基元）。

**離線測試（無需 GPU／Isaac）**：
```bash
# 從授權場景產生一份對齊真實座標的 slot 規格（dump 同時擷取真實牆面與資產 USD/姿態）
~/isaac-sim/python.sh src/sim/replicator.py --dump-spec-template ward_layout.json

# 以純 Python 跑 N 個設定，檢查接地／重疊／貼牆／房間圍束，並輸出 2D + 3D 圖
python3 tests/test_layout.py ward_layout.json --frames 10 --plots /tmp/ward_plots
```

## 試過的方法（哪些有效）

| 方法 | 結果 |
|---|---|
| **CUT / CycleGAN**（含深度條件、+CLIP‑MMD loss） | ✗ 停滯——小型 GAN 沒有真實影像先驗；DINOv2 落差／探針始終未收斂。 |
| **ControlNet‑SD**（depth/seg 控制 + 在真實資料上 LoRA + IP‑Adapter） | ~ 部分——擬真但只縮小約 28% 的 MMD 落差（像通用 SD，不像*我們*的病房）。 |
| **Cosmos‑Transfer2.5**（2B，多模態控制） | ✓ 擬真**且**符合本病房；結構／標註由 controls + guided mask 保留。 |
| **領域無關訓練器**（`train_yolo.py`：聯合監督 + DANN/MMD） | ✓ 在 sim 與 real 兩個領域都維持 mAP50 ≈ 0.90+，sim↔real 落差約 1%。 |

GAN 與 ControlNet 的腳本在 Cosmos 可行後即移除（見 git 歷史）。負面結果的紀錄在
`docs/sim2real-translation-findings.md`。

## 領域無關訓練器（`src/training/train_yolo.py`）

重點觀念：這裡的領域適應**不是**把模擬「扭曲成」真實，而是學出對 sim/real **領域無關**的特徵，
讓**同一個模型在兩個領域都表現好**。做法是單一階段（single‑phase），共四個要素：

1. **聯合監督訓練**：在「一個」混合 dataloader 中，同時用「有標註的模擬」（`train`）與
   「有標註的真實」（`real_dev`）。`real_dev` 會被**過取樣**（`--real-oversample`，預設 8，
   約佔混合資料的 27%），讓領域分支每個 batch 都看到兩個領域。偵測／分割頭因此**同時從兩個
   領域的標註學習**——這正是先前 MMD‑only 做法丟掉的訊號（它把 real_dev 當成無標註）。

2. **領域不變性**：完全從**同一個混合 batch**計算，每張影像的領域標籤直接從**檔案路徑**判讀
   （路徑含 `/real_dev/` → 真實，否則 → 模擬）：
   - `--dann`：**梯度反轉領域分類頭**接在某層 backbone 特徵上（`--align-layer`，YOLO11 預設
     第 10 層 C2PSA）。backbone 一邊下降任務 loss、一邊**上升**領域 loss → 特徵變得無法區分領域。
   - `--mmd`：對該 batch 內的模擬與真實影像做**無偏 per‑location MMD**。可與 DANN 併用。
     （用 per‑location 而非全域池化，原因見 `docs/rkhs-mmd-domain-adaptation.md`。）

3. **`--cls-prior`**：用各類別出現頻率初始化偵測頭的 class bias，穩定 dense detector 初期 loss。

4. **最後在兩個領域量測領域無關程度**：訓練結束後 `best.pt` 同時在模擬 `valid` 與**真實
   holdout** 上評估，寫入 `domain_report.json`。落差小 = 真的領域無關，而非只是過擬合模擬。

**資料紀律**：`real_holdout` **絕不**參與訓練、也**絕不**用於選 checkpoint（選 model 用模擬
`valid`），是誠實的跨領域測試集。`real_dev` 的標註則**確實被使用**——當成監督訓練目標。

**指令**：
```bash
.venv/bin/python src/training/train_yolo.py --data ward_data/ward_dataset_v4 \
    --model yolo11s-seg.pt --epochs 60 --imgsz 1024 --batch 16 --workers 8 \
    --name v4_domain_agnostic --dann --mmd --cls-prior --real-oversample 8
```
切分名稱可調：`--sim-train train --real-train real_dev --sim-val valid --real-holdout real_holdout`（預設）。

**結果（v3 資料集，取自 `domain_report.json`）**：

| 領域 | box mAP50 | box mAP50‑95 | mask mAP50 | mask mAP50‑95 |
|------|-----------|--------------|------------|----------------|
| sim（valid）   | 0.957 | 0.900 | 0.931 | 0.745 |
| real（holdout）| 0.943 | 0.855 | 0.939 | 0.760 |

兩個領域都維持 **mAP50 ≈ 0.90+**（box 與 mask），sim↔real 落差約 1%——模型是真正領域無關，
而非偏向某一邊。相較純模擬基線（真實 mask mAP50‑95 ≈ 0.12 且持續下降）是大幅躍進。

## 環境

- **`.venv`** → 連結到 `/home/edge-host/Documents/.venv`，主要 ML 環境（torch 2.12+cu130、
  transformers、diffusers、pycocotools、ultralytics、cv2）。除 Cosmos 外都用它。
- **`~/cosmos-transfer2.5/.venv`** — Cosmos 環境（torch 2.9+cu130），以 `uv sync --extra=cu130` 建立。
- **Isaac Sim** 在 `~/isaac-sim` — 以 `~/isaac-sim/python.sh` 執行渲染器。

硬體：NVIDIA **GB10**（DGX Spark，aarch64，CUDA 13）。

## 資料配置（gitignore——僅追蹤程式碼）

```
ward_data/ward_dataset_v4/    （build_dataset.py 產生）
  _train_render/_raw/   Isaac BasicWriter 輸出：rgb_*、distance_to_camera_*.npy、instance_segmentation_*、...
  train/ valid/         模擬影格（COCO RLE 遮罩）+ depth/
  test/                 728 張真實病房照片 + COCO 標註（真實領域／風格參考）
  real_dev/             由 test 切出、有標註，用於聯合訓練（real 端）
  real_holdout/         同樣切自 test，僅供最終測試——絕不訓練、不選 model
cosmos_jobs_v4/
  configs/ seg/ depth/ fgmask/   （每影格一份 Cosmos JSON 與控制輸入）
  outputs/  風格化結果   run_all.sh  manifest.csv
```

類別：43 個病房物件類別（+ `ward_object` 背景超類別 id 0），定義於 `src/common/categories.py`。

### 類別階層（supercategory 樹）

COCO 的 `supercategory` 欄位表達兩層階層：43 個葉類別歸入 8 個 supercategory。樹定義於
`src/common/categories.py` 的 `SUPERCATEGORY_TREE`（`supercategory_of(name)` 為查詢函式），
`build_dataset.py` 與 `replicator.py` 產生 COCO 標註時即套用。

| supercategory | 葉類別 |
|---|---|
| **furniture** | hospital_bed, bedside_table, overbed_table, companion_chair, stool, cabinet |
| **medical_equipment** | bedside_monitor, oxygen_flowmeter, gas_manifold, iv_pole, suction_jar, suction_knob, scale, weight_scale, stethoscope, ear_thermometer |
| **consumable** | alcohol_spray_bottle, sanitizer, gauze, medical_gloves, medical_package, syringe, paperbox, tissue_dispenser |
| **waste_container** | waste_bin, medical_waste_container, soiled_linen_bin |
| **bathroom_fixture** | toilet, toilet_handle, sink, shower |
| **structure_fixture** | door, door_handle, window, mirror, light_switch, air_vent, hook |
| **textile** | curtain, bed_curtain |
| **electronics** | TV, telephone, remote_control |

## 流程

### 1. 渲染模擬（Isaac Sim）
```bash
~/isaac-sim/python.sh src/sim/replicator.py --stage data/Collected_Ward0524/Ward0524.usd \
    --out <out> --frames 3000 --extra-channels --headless
```
- 語意標註直接取自場景中**手工標記的 USD semantics**（session 內橋接到 `class` taxonomy）。
- 光照：保留場景真正的**頂層 `/World` 房間燈**，靜音殘留的逐物件 env_light 與 studio DistantLight。
- `--randomize-materials` 就地擾動各物件原始材質（**排除鏡子**以保留反射）。
- `--randomize-placement` 啟用佈局隨機化；`--layout-spec ward_layout.json` 改以 slot 規格生成家具
  （見「佈局生成」一節）。

### 2. 建立有標註資料集（主 `.venv`）
```bash
.venv/bin/python src/training/build_dataset.py --total-images 10000 \
    --out ward_data/ward_dataset_v4 --stage data/Collected_Ward0524/Ward0524.usd \
    --prune-disable --oversample 1.1 --keep-intermediates --randomize-materials --render-depth
```
一條龍：渲染 train/valid、轉成 COCO（RLE 遮罩）、重映射真實 test、輸出每張 `depth/`、
並做標註直方圖檢查與抽樣 GT 疊圖（`_gt_check/`）。

### 3. 量測 sim→real 落差（隨時）
```bash
.venv/bin/python src/eval/domain_gap.py ...   # DINOv2/CLIP 空間的無偏 MMD² + 雙樣本探針
```

### 4. 用 Cosmos‑Transfer2.5 做 sim → real 風格轉換
```bash
.venv/bin/python src/cosmos/jobs.py --sim-dir ward_data/ward_dataset_v4/train/images \
    --test-dir ward_data/ward_dataset_v4/test --out cosmos_jobs_v4 --vary-style
nohup bash cosmos_jobs_v4/run_all.sh > ~/cosmos_batch.log 2>&1 &   # 可續跑，跳過已完成影格
```
**配方**（`src/cosmos/jobs.py` 預設）：`seg 0.8` + `depth 0.8`（GT 幾何）+ `edge 1.0`；
guided foreground mask 錨定有標註物件（`guided_generation_step_threshold ≈ 10`）；prompt 為場景層級、
**刻意不列出物件類別**（避免在未標註背景畫進同類物件）；style ref 取物件清單最相近（Jaccard）的真實照片。
詳見 `docs/cosmos-transfer-sim2real.md`。

### 5. 訓練 + 評估偵測器
- **領域無關路線（建議）**：見上方（`train_yolo.py --dann --mmd --cls-prior`）。
- **風格化模擬路線**：在 Cosmos 輸出組成的資料集上訓練，於**真實 holdout** 評估。

```bash
.venv/bin/python src/training/predict_yolo.py --weights <best.pt> \
    --data ward_data/ward_dataset_v4 --split test --eval --save-viz --device 0
```

## 其他腳本
- `src/training/train_seg_detr.py`、`predict_seg_detr.py` — Mask2Former 分割器
  （swin/dinov2/gfn/lejepa backbone，含 `--align-real` MMD 特徵對齊）與推論。
- `src/sim/gt_overlays.py`、`raw_overlays.py`、`inspect_semantics.py` — GT 疊圖與場景語意檢查。
- `docs/` — **[cosmos-transfer-sim2real.md](docs/cosmos-transfer-sim2real.md)**、
  **[sim2real-data-efficiency-methods.md](docs/sim2real-data-efficiency-methods.md)**、
  **[rkhs-mmd-domain-adaptation.md](docs/rkhs-mmd-domain-adaptation.md)**、Cosmos/Replicator 設定、
  global‑first 架構、sim2real 負面結果。

## 現況
- **佈局生成**：程序化約束求解器（`src/layout/solver.py`）已驗證——逐一擺放、硬性非重疊、
  貼齊真實牆面、接地正確；在真實病房 spec 上 0 違規（離線測試），取代手寫抖動取樣器。
- 領域無關訓練器（`train_yolo.py`）已驗證：sim 與 real 兩端皆達 mAP50 ≈ 0.90+，領域落差收斂。
- Cosmos‑Transfer2.5 安裝並端到端驗證完成。
- `ward_dataset_v4` 由 `Collected_Ward0524` 場景建立（已修正光照變暗與鏡面反射問題）。
