# 病房模擬到真實 (Hospital Ward Sim‑to‑Real)

針對**病房物件偵測／實例分割**的 sim‑to‑real（模擬到真實）流程。
目標：在 **Isaac Sim** 中渲染合成病房，將渲染結果轉換成**符合特定真實病房外觀的擬真影像**，
並在這份「已風格化但仍保有標註」的資料上訓練偵測器，使其能轉移到真實手持網路攝影機拍攝的照片。

風格轉換使用 **NVIDIA Cosmos‑Transfer2.5**（world‑foundation model）。這是本專案中第一個
能同時產生擬真、且符合「我們這間病房」外觀，又能保持合成標註有效的方法。

此外，本專案後來發展出一條**領域無關（domain‑agnostic）偵測器訓練**路線（見下方
「領域無關訓練器」一節），直接同時用模擬與少量**有標註的真實資料**聯合訓練，
這是目前最成功、實際把領域落差（domain gap）收斂的做法。

```
Isaac Sim 渲染 ──► RGB + GT 深度 + 實例分割 + COCO/YOLO 標註   (replicator_dataset.py)
        │
        ├─ seg（類別 id 圖）─┐
        ├─ GT 深度 ──────────┤ controls（控制訊號）
        ├─ edge（即時計算）──┘
        ├─ guided foreground mask（錨定有標註的物件）
        ├─ prompt = 該影格實際的物件類別（取自標註）
        └─ style ref = 依物件清單比對出最相近的真實照片
                          │
                          ▼
            Cosmos‑Transfer2.5 ──► 擬真、符合本病房風格、且標註對齊的影格
                          │            (gen_cosmos_jobs.py → cosmos_jobs/run_all.sh)
                          ▼
            訓練偵測器（YOLO）— 兩種路線：
              (a) 在風格化的模擬資料上訓練
              (b) 領域無關：聯合 模擬 + 有標註真實(real_dev) 訓練   (train_yolo.py)
                          ▼
            在**真實 holdout** 上評估；領域落差以 DINOv2 / MMD 特徵空間量測
```

## 試過的方法（哪些有效）

| 方法 | 結果 |
|---|---|
| **CUT / CycleGAN**（含深度條件、+CLIP‑MMD loss） | ✗ 停滯——小型 GAN 沒有真實影像先驗；DINOv2 落差／探針始終未收斂。 |
| **ControlNet‑SD**（depth/seg 控制 + 在真實資料上 LoRA + IP‑Adapter） | ~ 部分——擬真但只縮小約 28% 的 MMD 落差（像通用 SD，不像*我們*的病房）。 |
| **Cosmos‑Transfer2.5**（2B，多模態控制） | ✓ 擬真**且**符合本病房；結構／標註由 controls + guided mask 保留。 |
| **領域無關訓練器**（train_yolo.py：聯合監督 + DANN/MMD） | ✓ 在 sim 與 real 兩個領域都維持 mAP50 ≈ 0.90+，sim↔real 落差約 1%。 |

GAN 與 ControlNet 的腳本在 Cosmos 可行後即移除（見 git 歷史）。負面結果的紀錄在
`docs/sim2real-translation-findings.md`。

## 領域無關訓練器（`train_yolo.py`）

重點觀念：這裡的領域適應**不是**把模擬「扭曲成」真實，而是學出對 sim/real **領域無關**的特徵，
讓**同一個模型在兩個領域都表現好**。做法是單一階段（single‑phase），共四個要素：

1. **聯合監督訓練**：在「一個」混合 dataloader 中，同時用「有標註的模擬」（`train`）與
   「有標註的真實」（`real_dev`）。`real_dev` 會被**過取樣**（`--real-oversample`，預設 8，
   約佔混合資料的 27%），以免被 8,500 張模擬影像淹沒，也讓領域分支每個 batch 都看到兩個領域。
   偵測／分割頭因此**同時從兩個領域的標註學習**——這正是先前 MMD‑only 做法丟掉的訊號
   （它把 real_dev 當成無標註）。

2. **領域不變性（domain‑invariance）**：完全從**同一個混合 batch**計算，不需要第二個
   dataloader、也不需要額外 forward；每張影像的領域標籤直接從**檔案路徑**判讀
   （路徑含 `/real_dev/` → 真實，否則 → 模擬）：
   - `--dann`：**梯度反轉（gradient‑reversal）領域分類頭**接在某層 backbone 特徵上
     （`--align-layer`，YOLO11 預設第 10 層 C2PSA）。它做 sim‑vs‑real 的 BCE，λ 由
     `--dann-ramp` 漸增。backbone 一邊下降任務 loss、一邊**上升**領域 loss → 特徵變得無法
     區分領域。表格會多一欄 `dann`。
   - `--mmd`：對該 batch 內的模擬與真實影像做**無偏 per‑location MMD**（依領域切開後比對）。
     表格多一欄 `mmd`，可與 DANN 併用。（用 per‑location 而非全域池化：池化後的卷積特徵在高維
     會塌縮成退化的 ~0 MMD，見 `docs/rkhs-mmd-domain-adaptation.md`。）

3. **`--cls-prior`**：用（模擬 train 量到的）各類別出現頻率初始化偵測頭的 class bias，
   穩定 dense detector 初期的 loss（即 RetinaNet/YOLO `bias_init` 的客製化版本）。

4. **最後在兩個領域量測領域無關程度**：訓練結束後，`best.pt` 會同時在模擬 `valid` 與
   **真實 holdout** 上評估，印出並寫入 `domain_report.json`（sim 與 real 的 box/mask mAP
   及 sim→real 落差）。落差小 = 真的領域無關，而非只是過擬合模擬。

**資料紀律**：`real_holdout` **絕不**參與訓練、也**絕不**用於選 checkpoint（選 model 用模擬
`valid`），因此它是一個誠實的跨領域測試集。`real_dev` 的標註則**確實被使用**——當成監督訓練的目標
（已驗證：2,386 筆真實標註，全部含分割遮罩）。

**工程細節**：模型／trainer 類別定義在 module 層級（可 pickle）；`save_model` 會剝除領域適應的
hook/狀態，使存出的 checkpoint 能以單純的 `SegmentationModel` 重新載入。DANN 頭在建立 optimizer
**之前**就先建好（用一次極小的 dummy forward 讀出特徵通道數），確保其參數會被最佳化。領域 loss 項在
eval 時會補一個 0，讓 `loss_items` 長度與擴充後的 `loss_names` 一致（validator 也會呼叫
`model.loss`）。COCO→YOLO 轉換是**每個 COCO 標註對應一個 YOLO 實例**（遮罩被遮擋而斷成多塊時，
用 `merge_multi_segment` 縫成單一多邊形），避免把單一物件碎裂成多個假實例。

**指令**：
```bash
.venv/bin/python train_yolo.py --data ward_data/ward_dataset_v4 --model yolo11s-seg.pt \
    --epochs 60 --imgsz 1024 --batch 16 --workers 8 --name v4_domain_agnostic \
    --dann --mmd --cls-prior --real-oversample 8
```
切分名稱可調：`--sim-train train --real-train real_dev --sim-val valid --real-holdout real_holdout`
（此為預設）。訓練時可看 `dann`/`mmd` 欄與 `real_holdout_metrics.csv`；最終看 `domain_report.json`。

**結果（v3 資料集）**：最終模型在兩個領域的評估（取自 `domain_report.json`）：

| 領域 | box mAP50 | box mAP50‑95 | mask mAP50 | mask mAP50‑95 |
|------|-----------|--------------|------------|----------------|
| sim（valid）   | 0.957 | 0.900 | 0.931 | 0.745 |
| real（holdout）| 0.943 | 0.855 | 0.939 | 0.760 |

兩個領域都維持 **mAP50 ≈ 0.90+**（box 與 mask），sim↔real 落差約 1%（mask mAP50 幾乎相等：
0.931 vs 0.939）——模型是真正領域無關，而非偏向某一邊。相較純模擬基線（真實 mask
mAP50‑95 ≈ 0.12 且持續下降）是大幅躍進。

## 環境

兩個 Python 環境（皆未納入版控）：
- **`.venv`** → 連結到 `/home/edge-host/Documents/.venv`，主要 ML 環境（torch 2.12+cu130、
  transformers、diffusers、pycocotools、ultralytics、cv2）。除了 Cosmos 之外都用它。
- **`~/cosmos-transfer2.5/.venv`** — Cosmos 環境（torch 2.9+cu130），以 `uv sync --extra=cu130` 建立。
- **Isaac Sim** 在 `~/isaac-sim` — 用 `~/isaac-sim/python.sh` 執行渲染器。

硬體：NVIDIA **GB10**（DGX Spark，aarch64，CUDA 13）。Cosmos‑Transfer2.5 官方支援 DGX Spark（cu130）。

## 資料配置（gitignore——僅追蹤程式碼）

```
ward_data/ward_dataset_v4/    （build_dataset.py 產生；舊版為 ward_v3）
  _train_render/_raw/   Isaac BasicWriter 輸出：rgb_*、distance_to_camera_*.npy、instance_segmentation_*、...
  train/                模擬影格：img_*.png（+轉檔後的 images/ labels/）depth/ _annotations.coco.json
  valid/                模擬影格（結構相同）
  test/                 728 張真實病房照片 + COCO 標註（真實領域／風格參考）
  real_dev/             由 test 以 crc32(stem)%2 切出、有標註，用於聯合訓練（real 端）
  real_holdout/         同樣切自 test，僅供最終測試——絕不訓練、不選 model
cosmos_jobs_v3/
  configs/  seg/  depth/  fgmask/   （每個模擬影格一份 Cosmos JSON 與控制輸入）
  outputs/  風格化結果 <stem>.jpg    run_all.sh  manifest.csv
```

類別：44 個病房類別，定義於 `ROS2_bridge/src/fixed_categories.py`。

## 流程

### 1. 渲染模擬（Isaac Sim）
```bash
~/isaac-sim/python.sh replicator_dataset.py --stage <ward.usd> --out <out> \
    --frames 3000 --extra-channels --headless        # RGB + GT 深度 + 實例分割 + 標註
```
- 語意標註直接取自場景中**手工標記的 USD semantics**（在 session 內橋接到 `class` taxonomy；
  不再用以 prim 名稱的 regex 規則）。
- 光照：保留場景中真正的**頂層 `/World` 房間燈**（不同場景版本命名不同，採命名無關規則），
  靜音殘留的逐物件 env_light 與 studio DistantLight。
- 材質領域隨機化（`--randomize-materials`）會就地擾動各物件的原始材質，但**排除鏡子材質**
  以保留反射。

### 2. 建立有標註資料集（主 `.venv`）
```bash
.venv/bin/python build_dataset.py --total-images 10000 --out ward_data/ward_dataset_v4 \
    --stage data/Collected_Ward0524/Ward0524.usd --prune-disable --oversample 1.1 \
    --keep-intermediates --randomize-materials --render-depth
```
一條龍：渲染 train/valid、轉成 COCO（RLE 遮罩）、由 `Ward_dataset0518` 複製並重映射真實 test、
輸出每張的 `depth/`（給 Cosmos depth 控制）、並做標註直方圖檢查與抽樣 GT 疊圖（`_gt_check/`）。

### 3. 量測 sim→real 落差（隨時）
DINOv2/CLIP 空間的無偏 MMD² 與平衡分類器雙樣本探針（見 `docs/rkhs-mmd-domain-adaptation.md`）。

### 4. 用 Cosmos‑Transfer2.5 做 sim → real 風格轉換
```bash
.venv/bin/python gen_cosmos_jobs.py --sim-dir ward_data/ward_dataset_v3/train/images \
    --test-dir ward_data/ward_dataset_v3/test --out cosmos_jobs_v3 --vary-style
nohup bash cosmos_jobs_v3/run_all.sh > ~/cosmos_batch.log 2>&1 &   # 可續跑，跳過已完成影格
```
**配方**（`gen_cosmos_jobs.py` 預設）：
- **Controls：** `seg 0.8`（類別 id 圖→物件區域；由 0.6 提高以更嚴格貼合標註圖、抑制背景亂長物件）
  + `depth 0.8`（GT 幾何）+ `edge 1.0`（輪廓，即時）。不用 `vis`（vis 會保留模擬的顏色/材質）。
- **Guided generation：** 前景遮罩（物件遮罩聯集）錨定有標註物件；
  `guided_generation_step_threshold ≈ 10`（整數步數，約 35 步中的前 10 步先錨定結構，之後放開讓它擬真重繪）。
- **Prompt：** 場景層級框架 + 共用材質，**刻意不列出物件類別**（seg 控制已指定有哪些物件與其位置；
  在 prompt 中列出類別會誘使模型把更多同類物件畫進未標註的背景）。風格參考仍用物件清單比對。
- **Style ref（`image_context_path`）：** 物件清單與該影格最相近（Jaccard）的真實照片；
  `--vary-style` 會在前幾名中抽樣並變化 seed 與光照。
- **`guidance` 3**（適中，避免壓過結構）。
- **空前景影格會被略過**：相機若沒拍到任何有標註物件，guided mask 全為 0 會使 Cosmos 崩潰，
  且這種影格也沒東西可風格化。

限制：Cosmos **沒有逐區域的類別→外觀綁定**（seg 是空間性的，prompt 是場景層級）；物件*身分*由 prompt
偏導、由 guided mask 錨定，並非硬綁定。

### 5. 訓練 + 評估偵測器
- **領域無關路線（建議）：** 見上方「領域無關訓練器」一節（`train_yolo.py --dann --mmd --cls-prior`）。
- **風格化模擬路線：** 在 Cosmos 輸出組成的資料集上訓練，於**真實 holdout** 評估。

預測與視覺化：
```bash
.venv/bin/python predict_yolo.py --weights <best.pt> --data ward_data/ward_dataset_v4 \
    --split test --eval --save-viz --device 0          # COCO mAP + 疊圖
# 省略 --max-viz → 全部影像；--predict-batch 控制每塊張數（會在塊間釋放 GPU 記憶體，避免 OOM）
```

## 其他腳本
- `train_seg_detr.py`、`predict_seg_detr.py` — Mask2Former 分割器（swin/dinov2/gfn/lejepa backbone，
  含 `--align-real` MMD 特徵對齊）與推論。
- `render_gt_overlays.py`、`overlay_raw_gt.py`、`inspect_semantics.py` — GT 疊圖與場景語意檢查。
- `docs/` — **[cosmos-transfer-sim2real.md](docs/cosmos-transfer-sim2real.md)**（目前 Cosmos 做法與配方）、
  **[sim2real-data-efficiency-methods.md](docs/sim2real-data-efficiency-methods.md)**（領域無關配方與結果）、
  **[rkhs-mmd-domain-adaptation.md](docs/rkhs-mmd-domain-adaptation.md)**（MMD/RKHS 理論）、
  Cosmos/Replicator 設定、global‑first 架構、sim2real 負面結果。

## 現況
- 領域無關訓練器（`train_yolo.py`）已驗證：在 sim 與 real 兩端皆達 mAP50 ≈ 0.90+，領域落差收斂。
- Cosmos‑Transfer2.5 安裝並端到端驗證完成；`cosmos_jobs_v3/` 持有設定檔，`run_all.sh` 為長時間批次步驟。
- `ward_dataset_v4` 由新的 `Collected_Ward0524` 場景建立（已修正光照變暗與鏡面反射問題）。
