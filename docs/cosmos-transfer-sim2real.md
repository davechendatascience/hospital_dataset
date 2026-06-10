# Cosmos‑Transfer2.5 病房 Sim‑to‑Real — 方法與突破

這份筆記記錄我們如何使用 **NVIDIA Cosmos‑Transfer2.5‑2B**，把 Isaac‑Sim 合成的病房算圖
轉換成擬真、且符合「我們這間病房」風格的影像，同時保留合成資料的標註（labels）有效；
也記錄了讓它真正能用的一連串修正與洞見。

實作：`gen_cosmos_jobs.py`（為每一張 sim 影格產生一個 Cosmos job）→
`cosmos_jobs/run_all.sh`（可續跑的批次）。Cosmos 程式庫與環境位於 `~/cosmos-transfer2.5`。

---

## 1. 為什麼選 Cosmos（在 GAN 與 ControlNet 失敗之後）

| 方法 | 結果 | 原因 |
|---|---|---|
| CUT / CycleGAN（含 depth、+CLIP‑MMD loss） | ✗ 差距始終無法收斂 | 小型 generator + 約 728 張影像的 discriminator，缺乏真實影像先驗 |
| ControlNet‑SD（depth/seg control + LoRA‑on‑real + IP‑Adapter） | ~ 只縮小約 28% 的 DINOv2‑MMD 差距 | 擬真，但是「通用 SD」的樣貌 —— 不是我們這間病房 |
| **Cosmos‑Transfer2.5‑2B** | ✓ 既擬真又像我們的病房 | world‑foundation 先驗 + 多模態空間控制 + 真實風格參考影像 |

Cosmos 是第一個同時做到「看起來真實」且「符合我們病房」的方法，因為結構由 control maps
釘住，而外觀來自一張真實參考影像加上十億級的生成式先驗 —— 反面結果見
`sim2real-translation-findings.md`。

## 2. Cosmos‑Transfer2.5 的運作方式（我們用到的控制桿）

執行模式：**image‑to‑image**，每個 job 一張影格（`max_frames: 1`、
`num_video_frames_per_chunk: 1`），透過 `examples/inference.py -i <cfg>.json -o <out>` 執行。

- **`video_path`** —— sim 的 RGB 影格（被重新上樣式的對象）。
- **Control 分支**（每個為 `{control_path?, control_weight}`；空間性的、ControlNet 風格）：
  - `seg` —— 由 sim COCO 點陣化出的 class‑id 顏色圖 → 物件**區域**。
  - `depth` —— Isaac 的 ground‑truth 深度 → **幾何**。
  - `edge` —— 輪廓（即時計算）→ 只給形狀、不給顏色。
  - `vis`（blur）—— 保留**輸入影像的顏色/紋理**；我們刻意**不使用**它（會把 sim 的顏色帶進來）。
- **`image_context_path`** —— 一張**真實病房照片**當作風格參考（類似 IP‑Adapter 的外觀驅動：
  顏色、材質、光線）。
- **`prompt`** —— 描述場景/物件的文字。由 **Cosmos‑Reason1‑7B** 編碼（它在這條推論路徑中
  是當作*文字編碼器*，並不是會自動改寫 prompt 的場景推理器 —— 此路徑沒有 prompt 上採樣/重寫機制）。
- **`guidance`** —— classifier‑free guidance（預設 3；維持適中，避免文字蓋過結構）。
- **Guided generation**（選用、且很重要 —— 見 §4）：`guided_generation_mask`（前景遮罩）
  在去噪早期把遮罩區域錨定到 sim；`guided_generation_step_threshold` 控制錨定持續多久。

**關鍵限制：** Cosmos **沒有逐區域的 class→外觀綁定**。seg 圖純粹是*空間性*的（沒有
顏色→「overbed table」的對照表），而 prompt 是*場景層級*的。因此物件**身分（identity）**只能
被*引導*（透過 prompt）與在*結構上被錨定*（透過 guided mask），永遠不會被硬性綁定到某個區域。
要做到真正的逐類別控制，需要**對 seg control 分支做 post‑training / LoRA**。

## 3. 突破與修正（依時間順序）

### 3.1 Depth control 必須是 3 通道
Cosmos 的 control reader 會執行 `einops.rearrange(x, "t h w c -> t c h w")`，也就是需要 **HWC**。
我們的 GT depth PNG 是灰階（`mode L`，2 維）→ 直接讓整個批次崩潰
（`Wrong shape: expected 4 dims … (1,1080,1920)`）。**修正：** 把 depth control 另存成 3 通道
RGB（`Image.open(d).convert("RGB")`）。seg 圖原本就是 RGB，所以只有 depth 需要處理。

### 3.2 以標註驅動的 prompt（正確的物件*類型*）
通用場景模板會讓 Cosmos 在標註寫著 **overbed table** 的地方畫出*櫃子* (cabinet)
（因為 prompt 字面就寫了 "bedside cabinet"）。**修正：** 每張影格的 prompt 改成由它**自己的
COCO 類別**組成（`overbed table, IV pole, telephone, …`）。點名真實存在的物件，會引導 Cosmos
畫出正確的物件。

### 3.3 Guided generation = 錨定有標註的前景
為了阻止模型自行重新詮釋物件，我們把**前景遮罩**（該影格所有 instance mask 的聯集）餵給
`guided_generation_mask`。踩過的坑：
- 格式必須是 **`.npz`、key 為 `arr_0`、shape 為 `(T,H,W)`**（單通道；`foreground_labels=None`
  代表任何非零值都算前景）。PNG 會被拒絕（"not a mp4 or npz file"）。
- **`guided_generation_step_threshold` 是整數的步數**（預設 10，總去噪步數約 35）——
  *不是* 0–1 的比例。（外部建議講的「0.3」在這裡約等於 10 步。）

### 3.4 Guided 與擬真之間的拉扯
Guided generation 會把遮罩區域錨定**到 sim 輸入**，所以它保留了結構/身分，卻把**前景外觀
往 sim 拉**（在那些區域蓋過真實風格參考）。在一張病房影格上量測：
- **steps 25（強）：** 身分鎖得很死，但偏 sim 味（桌面偏綠、床偏深藍）。
- **steps 12（中）：** 平衡良好 —— 明顯是一張桌子，且大致擬真。
- **steps 0（關）：** 最擬真，但物件身分可能漂移。

所以強度是一個在*身分*（高）與*擬真*（低）之間的旋鈕。

### 3.5 為什麼我們仍然需要 guided mask（context 會誤導）
沒有 guidance 時，**`image_context_path`（風格參考）可能會主導**，讓 Cosmos
*從參考影像的內容去生成*，而不是依 sim 的實際場景 → 物件/場景錯誤。guided 前景遮罩會錨定
*這一張*影格的結構，使參考影像只負責**重新上樣式**。結論：guided 保持**開啟**，但用**中等強度**
（steps ≈ 10）—— 早期錨定結構、後期釋放以做擬真的重新上樣式。

### 3.6 拿掉我們自己的場景分類器 —— control 已經定義了場景
我們原本有一個（依物件存在與否的）ward/corridor/bathroom 啟發式分類器，它會**標錯**影格
（例如走廊裡有洗手台 → 被判成 "bathroom"；醫療氣體床頭板 → 被判成 "corridor"），接著就掛上
錯誤的 prompt **與**錯誤的風格參考。由於 Cosmos 沒有自動推理器，*而且* **seg+depth control +
guided mask 本來就已經編碼了真實的場景結構**，我們把分類器整個移除：
- **Prompt** 改為與場景無關 —— 「一張寫實的台灣醫院室內照，包含 `<這張影格實際的物件>`；
  `<共用材質>`。」場景由 control 決定。
- **已驗證：** 影格 `0006` 現在正確算成**醫療氣體床頭板**（先前被誤判為 "corridor"），
  `0008` 正確算成**有垃圾桶的雜物/走廊區**（先前被誤判為 "bathroom"）。

### 3.7 以內容匹配風格參考（物件清單的 Jaccard）
如何穩健地為每張影格挑選真實的 `image_context_path`：
- **不要**用 DINOv2 最近鄰 —— 在 DINOv2 空間裡 sim ≠ real，會配錯。
- **不要**用場景標籤 —— 太脆弱（見 3.6）。
- **要：** 把每張真實測試照片以其**物件類別集合**建索引，然後對每張 sim 影格，挑選類別集合
  **Jaccard** 重疊度最高的真實照片（取 top‑K；`--vary-style` 會在 top‑8 中抽樣以增加多樣性）。
  兩邊都有 COCO 標註，所以是依「實際存在的物件」來匹配。（前提：真實參考需有標註 ——
  `ward_v3/test`，728 張已標註照片。）

## 4. 目前的配方（`gen_cosmos_jobs.py` 預設值）

| 控制桿 | 數值 | 理由 |
|---|---|---|
| `seg` control_weight | **0.6** | 保留物件區域，但不讓調色盤顏色被印上去 |
| `depth` control_weight | **0.8** | 強力錨定幾何/視角 |
| `edge` control_weight | **1.0** | 給輪廓/形狀但不給顏色（即時計算） |
| `vis` | **關閉** | vis 會保留 *sim 的* 顏色/紋理 —— 與我們要的相反 |
| `guided_generation_step_threshold` | **10**（約 35 步中） | 早期錨定結構，後期釋放以擬真重新上樣式 |
| guided mask | 物件 mask 聯集，`.npz arr_0 (1,H,W)` | 錨定有標註的前景；阻止 context 誤生成 |
| `guidance` | **3** | 適中，避免文字蓋過結構 |
| `prompt` | 與場景無關 + 影格**實際的 COCO 物件** | 正確的物件類型；不靠脆弱的場景分類 |
| `image_context_path` | 真實照片，依**物件清單匹配**（Jaccard） | 每張影格有合適的外觀 |
| `--vary-style` | 從 top‑8 參考抽樣 + 隨機 seed + 光線修飾語 | 為訓練集帶來風格多樣性 |

每張影格 pipeline 會寫出：`cosmos_jobs/seg/<stem>.png`、`depth/<stem>.png`（RGB）、
`fgmask/<stem>.npz`、以及 `configs/<stem>.json`。批次（`run_all.sh`）**可續跑** —— 會跳過
已有輸出的影格，若某張影格中止則對其餘的重新開始。

## 5. 已驗證 / 待辦

**已驗證：** 在病房/床頭板/走廊/浴室影格上端到端跑通 —— 擬真、符合我們病房的風格、場景與
物件類型正確、標註位置保留（seg+depth），且 guided mask 能阻止風格參考劫持內容。

**待辦：**
- 逐區域的 class→外觀目前仍只是*被引導*、而非被綁定。要硬性保證 → 在我們的分類體系上
  **對 seg 分支做 post‑train / LoRA**（Cosmos 提供 vid2vid 的 post‑training 設定）。
- 完整 2700 張影格的批次在單張 GB10 上約需 ~30 小時（單 GPU、循序、約 40 秒/影格）。
- 最終的取捨指標是**在真實 holdout 上的偵測器 AP**（`train_yolo_da.py`），而不是 DINOv2/CLIP
  差距（那只是參考）。
