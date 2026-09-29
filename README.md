# Kaggle × Jev：用 Jev 模型重做 5 個經典 ML 題目

把 5 個經典 Kaggle 題目的傳統解法，改成用 [TypeSafe](https://typesafe.ai) 的 **Jev**（System One 模型）推論，
在同一份測試集上和 Kaggle 經典解法比較，並測試 few-shot、分群挑例子、stacking 三種延伸做法。

| # | Kaggle 題目 | 類型 | Kaggle 經典解法 | Jev 問法 |
| --- | --- | --- | --- | --- |
| 1 | [Titanic](https://www.kaggle.com/competitions/titanic) | 表格・二分類 | HistGradientBoosting + Title/FamilySize/HasCabin 特徵工程 | `Noul`：這位乘客是否生還？ |
| 2 | [SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) | 文字・二分類 | TF-IDF + Multinomial Naive Bayes | `Noul`：這則簡訊是否為垃圾訊息？ |
| 3 | [IMDB 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) | 文字・情感分析 | TF-IDF (1-2 gram) + Logistic Regression | `Noul`：整體是否為正面評論？ |
| 4 | [BBC News Classification](https://www.kaggle.com/competitions/learn-ai-bbc) | 文字・5 類 | TF-IDF + Linear SVM | `Choice`：business / entertainment / politics / sport / tech |
| 5 | [Iris](https://www.kaggle.com/datasets/uciml/iris) | 數值・3 類 | StandardScaler + Logistic Regression | `Choice`：setosa / versicolor / virginica |

這 5 題刻意混合了 Jev 應該擅長的文字判斷（2–4），以及偏數值規則的表格題（1、5），這樣才看得出 Jev 適合用在哪裡。

**一句話結論**：文字分類用 Jev 0-shot 就能打平或勝過 Kaggle 解法（IMDB 0.963 vs 0.918），而且不需要訓練資料；
表格題仍以傳統模型為佳。詳見[結果](#結果)。

---

## 快速開始

```bash
git clone https://github.com/lichin1/jev-application.git
cd jev-application
python -m pip install -e ".[dev]"      # 安裝套件與 jev-bench 指令（需要 Python 3.10+）

make test                               # 離線測試，不需要 API key
jev-bench baseline                      # 只跑 Kaggle 解法，不需要 API key

cp .env.example .env                    # 填入 TYPESAFE_API_KEY（到 https://typesafe.ai 取得）
set -a; source .env; set +a
jev-bench all --concurrency 16          # 完整實驗：Kaggle、所有 Jev 方法、stacking、報告
```

結果會寫進 `results/`，報告在 [`results/summary.md`](results/summary.md)。

> **成本**：整個實驗實際花費 **約 US$2.8**（約 6 萬次 Jev 呼叫、6,730 萬 input tokens），其中 IMDB 佔約 7 成，
> 細項見[成本](#成本)。想先試水溫可以加 `--scored-n 200`（每題只抽 200 筆測試資料）或 `--tasks titanic iris`。
> 答案會快取在 `.cache/jev/`，中斷後重跑只會補送沒回答過的資料，不會重複計費。

## 指令

每個指令都可以用 `jev-bench <指令>` 或 `python -m jev_bench <指令>` 執行，`make help` 列出對應的 make 捷徑。

| 指令 | 做什麼 | 需要 API key | make |
| --- | --- | :---: | --- |
| `jev-bench all` | 完整實驗：Kaggle → Jev 0-shot / 3-shot / cluster / 隨機對照 → stacking → 報告 | ✓ | `make all` |
| `jev-bench baseline` | 訓練並評分 Kaggle 解法 | | `make baseline` |
| `jev-bench jev --shots 0 3` | 問 Jev，`--shots` 指定方法（見下） | ✓ | `make jev` / `make fewshot` |
| `jev-bench jev --shots cluster cluster-random` | 分群挑例子，以及數量相同的隨機對照 | ✓ | `make cluster` |
| `jev-bench stack` | 把 Jev 機率當特徵，和 Kaggle 分數一起訓練第二階段模型 | ✓ | `make stack` |
| `jev-bench examples --shots 0 3 cluster` | 輸出每種方法送給 Jev 的範例請求 | | `make examples` |
| `jev-bench report` | 從 `results/*.json` 重建 `summary.md` | | `make report` |

共用選項：

| 選項 | 預設 | 說明 |
| --- | --- | --- |
| `--tasks` | 全部 5 題 | 例如 `--tasks titanic iris` |
| `--shots` | `0`（`all` 為 `0 3 cluster cluster-random`） | 整數 k＝每類 k 個隨機例子；`cluster`；`cluster-random` |
| `--stack-shots` | `0 cluster` | stacking 用哪種 Jev 機率當特徵 |
| `--scored-n` | `0`（完整測試集） | 只用分層抽樣的 n 筆測試資料評分，省呼叫次數 |
| `--model` | `jev-latest` | Jev 模型，例如 `jev-preview` |
| `--concurrency` | `8` | 同時送出的請求數 |

每個指令只會更新自己負責的結果，所以可以分開跑、任意組合；`report` 永遠讀取所有已存在的結果。

## 專案結構

```
jev_bench/
  config.py         路徑、隨機種子、預設值（所有設定集中在這裡）
  datasets.py       下載 Kaggle 資料（GitHub 鏡像）、載入、分層切分
  tasks.py          5 題的定義：Kaggle 解法 + Jev 的 state／問題／解碼
  metrics.py        所有方法共用的指標（accuracy、macro-F1、正類 F1、ROC AUC）
  jev_runner.py     呼叫 Jev：並行、快取、失敗自動重試
  few_shot.py       從訓練集挑例子，掛到答案的 criteria 上
  cluster_shots.py  k-means 分群 × 類別交集挑例子，以及數量相同的隨機對照
  stacking.py       Kaggle out-of-fold 分數 + Jev 機率的第二階段模型，以及只用 Kaggle 的對照組
  methods.py        方法清單：每種方法在結果檔和報告裡的名稱
  experiments.py    每個階段對一題的評估（baseline / jev / stack / examples）
  report.py         產生 results/summary.md
  pipeline.py       依序執行各階段
  cli.py            命令列介面（jev-bench）
tests/              離線測試：用永遠答對的假 Jev 跑完整 pipeline
results/            每題的結果（<task>.json）、範例請求、報告 summary.md
```

資料流：

```
datasets.split ─► experiments.run_baseline ─────────────────────────┐
              ├─► experiments.run_jev   ─► jev_runner.ask_jev (快取) ├─► results/<task>.json ─► report ─► summary.md
              └─► experiments.run_stack ─► stacking.run_stacking ────┘
```

官方 Python SDK 是 [`typesafe-sdk`](https://pypi.org/project/typesafe-sdk/)（`import typesafe_sdk`），
不是 `typesafe` 或 `typesafe-ai`（後兩者分別是無關套件與防搶註的轉址套件）。

---

## 實驗設計

- **切分**：每個資料集做分層 80/20 切分（`random_state=42`）。Kaggle 模型只用 80% 訓練；
  所有方法都在**完整的 20% 測試集**上評分，也就是考同一份、訓練時沒看過的考卷。
- **Jev 0-shot**：Jev 看不到任何訓練標籤，只收到單筆資料的 `state` 和一個有型別的問題。
  - 二分類用 `Noul`，取 `noul ≥ 0.5` 為正類，並用 `noul` 機率算 ROC AUC。
  - 多分類用 `Choice`，取 `choice` 為預測，另外統計 `confidence ≥ 0.8` 時的準確率與覆蓋率。
- **Few-shot（`--shots k`）**：從**訓練集**每類抽 k 筆有標籤的例子（固定 seed，所有測試資料共用同一組），
  附加到該答案的 criteria 裡；instructions 不變，所以 k-shot 與 0-shot 只差在例子：
  ```json
  "criteria": {
    "true":  {"description": "The passenger survived the sinking.", "examples": [{"passenger": {...}}, ...]},
    "false": {"description": "The passenger died in the sinking.",  "examples": [...]}
  }
  ```
  優先挑短的例子（JSON ≤ 700 字）；沒有夠短的（BBC 幾乎全部）就把例子裡的文字截到前 600 字。
  用到的訓練列 index 記在 `results/<task>.json` 的 `shot_train_rows`，可重現、可確認沒有測試資料混入。
- **Cluster few-shot（`--shots cluster`）**：隨機抽樣多半抽到每類「最典型」的樣本。這裡先用 k-means 把訓練集分群
  （特徵與 Kaggle 模型相同：文字用 TF-IDF→SVD 100 維，表格用標準化數值＋one-hot），
  群數在 `類別數..8` 之間取 silhouette 最高者；再把**每個 y 類別與每個群取交集**，
  每個 ≥ 3 筆的交集格取一個代表例（最接近該格中心的樣本）。
  一個 y 分散在多個群 → 每個群都出一個例子，其中 y 在該群是少數的格子（`label_share_of_cluster < 0.5`）
  就是靠近決策邊界的「邊緣 pattern」。分群結果記在 `results/<task>.json` 的 `jev_cluster.clustering`。
- **隨機對照（`--shots cluster-random`）**：每類抽**與 cluster 相同數量**的隨機例子，
  用來區分「多樣性」與「單純例子變多」的效果。
- **Stacking（`jev-bench stack`）**：把 Jev 的機率當特徵，和 Kaggle 模型的分數一起餵給第二階段的邏輯迴歸，
  並用一個不含 Jev 的對照組衡量 Jev 的貢獻（見[結果](#stacking把-jev-機率當特徵加進-kaggle-模型)）。
- **避免作弊**：
  - Titanic 不送乘客全名，只送稱謂（Mr/Mrs/Miss…），以免 Jev 靠記憶認出真實人物。
  - Iris 的選項描述只有物種名稱，不寫花瓣長度門檻，否則等於是我們自己寫規則，而不是 Jev 的判斷。
  - Few-shot 例子只從訓練集挑；stacking 時被當成例子的訓練列不會進入第二階段。
- **指標**：accuracy、macro-F1；二分類另有正類 F1 與 ROC AUC；Jev 另記錄 p50/p95 延遲與 token 用量。

---

## 結果

model `jev-latest`（= `jev-1.13.0`），所有方法都在完整 20% 測試集上評分。
完整數字見 [`results/summary.md`](results/summary.md)，分成四張表：**Summary**（每題的資料集、筆數、Jev 問題、範例 state）、
**Accuracy**、**Macro-F1**、**ROC AUC**（僅二分類題）；每張指標表的最佳分數標成粗體加底線。

| 題目 | Full test n | Kaggle | Jev 0-shot | Jev 3-shot | Jev cluster | 隨機對照 | Kaggle + Jev 0-shot | Kaggle + Jev cluster |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IMDB 影評 | 10,000 | 0.918 | 0.963 | 0.963 | 0.963 | 0.963 | **0.964** | **0.964** |
| SMS Spam | 1,115 | 0.985 | 0.979 | 0.980 | 0.980 | 0.981 | **0.992** | 0.991 |
| BBC News | 445 | 0.987 | 0.982 | 0.978 | 0.982 | 0.975 | 0.989 | **0.991** |
| Titanic | 179 | **0.821** | 0.648 | 0.682 | 0.754 | 0.682 | 0.793 | 0.799 |
| Iris | 30 | 0.933 | 0.600 | 0.900 | 0.967 | 0.833 | 0.967 | **1.000** |

（Accuracy。每次呼叫 p50 延遲約 0.29 秒、p95 約 0.45–0.63 秒。）

### 怎麼判讀

- **看筆數判斷差距是不是雜訊**：Iris 只有 30 筆，差 1 朵就是 0.033；n=445 時差 1 筆是 0.002。
  Titanic 的 cluster vs 3-shot（179 筆多對 13 筆）、IMDB 的 Jev vs Kaggle（10,000 筆差 4.5 個百分點）才是可信的差距。
- **Accuracy 與 AUC 要一起看**：AUC 只看排序（生還機率高的是否真的較常生還），accuracy 看以 0.5 為門檻切得對不對。
  AUC 高、accuracy 低代表排序對了、門檻不對。
- **Macro-F1 明顯低於 accuracy** 代表少數類別常被判錯（例如 Titanic 0-shot 0.648 vs 0.593，生還者常被判成死亡）。

### Jev 0-shot

- **IMDB：Jev 在 10,000 筆上贏 Kaggle 4.5 個百分點（0.963 vs 0.918），AUC 0.993 vs 0.975。**
  Kaggle 模型用了 40,000 筆訓練資料，Jev 一筆都沒用。這是整份報告最可信的結論（樣本數最大）。
- **SMS、BBC：打平。** SMS 準確率差 0.6 個百分點、AUC 幾乎一樣（0.991 vs 0.992）；
  BBC 在 445 篇中少對 2 篇。BBC 有 93.9% 的答案信心 ≥ 0.8，這部分準確率 99.3%，信心可用來決定哪些要人工複查。
- **Titanic：Jev 沒有抓到「婦孺優先」。** 女性平均生還機率只給 0.40（實際 0.74），男性 0.33（實際 0.20），
  幾乎沒有區分性別；機率整體偏低，門檻 0.5 時只預測 25% 生還（實際 38%）。
- **Iris：Jev 從來沒有預測 virginica**（10 朵全被判為 versicolor）。純數值、需要從資料學邊界的問題不是 Jev 的強項。

### Few-shot

- **文字題：例子沒有幫助。** IMDB 四種 Jev 方法都是 0.963、SMS 在 0.979–0.981 之間，BBC 3-shot 反而少對 2 篇；
  token 成本卻是 0-shot 的 2–3.5 倍（IMDB：620 萬 → 1,435 萬 token）。
- **表格／數值題：例子有明顯幫助**，Iris 0.60 → 0.90、Titanic 0.648 → 0.682（每類 3 個隨機例子）。

### Cluster few-shot：用分群挑「邊緣 pattern」

| 題目 | k | 例子數（每類） | 0-shot | 3-shot | **cluster** | 隨機對照（同數量） | Kaggle |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Titanic | 3 | 3 / 3 | 0.648 | 0.682 | **0.754** | 0.682 | 0.821 |
| Iris | 3 | 1 / 2 / 2 | 0.600 | 0.900 | **0.967** | 0.833 | 0.933 |
| BBC News | 8 | 3 / 4 / 2 / 3 / 3 | 0.982 | 0.978 | 0.982 | 0.975 | 0.987 |
| SMS Spam | 8 | 8 / 5 | 0.979 | 0.980 | 0.980 | 0.981 | 0.985 |
| IMDB | 2 | 2 / 2 | 0.963 | 0.963 | 0.963 | 0.963 | 0.918 |

cluster 要和**隨機對照**比（兩者例子數相同，只差挑法），不是和 3-shot 比。

- **Titanic：0.682 → 0.754，AUC 0.783 → 0.826。** 例子數與 token 幾乎一樣，唯一差別是挑法：
  分群挑到了「生還者中的少數群」（生還者只佔該群 25%）與「死者中的少數群」（死者只佔該群 30%）。
  隨機對照剛好與 3-shot 抽到同一組例子（同數量、同 seed），所以兩者分數相同；179 筆中 cluster 多對 13 筆。
- **Iris：0.833 → 0.967。** 只用 5 個例子就勝過 3-shot 的 9 個，所以提升來自多樣性而不是例子數。
  分群挑到的是 5 朵「長得像 virginica 的 versicolor」與 14 朵「長得像 versicolor 的 virginica」這兩個混淆區的代表。
  Iris 只有 30 筆，0.967 vs Kaggle 0.933 只差 1 朵，不能說贏過 Kaggle。
- **文字題沒有差別。** 高維文字的 silhouette 只有 0.02–0.08，分群本身就不太明確
  （IMDB 只分出 2 群，SMS／BBC 都頂到上限 8 群）。

### Stacking：把 Jev 機率當特徵加進 Kaggle 模型

#### 做法

```
Kaggle 模型分數 ─┐
                 ├─► 第二階段邏輯迴歸 ─► 「Kaggle + Jev 0-shot」／「Kaggle + Jev cluster」
Jev 機率 ────────┘

Kaggle 模型分數 ───► 同一個第二階段邏輯迴歸 ─► 「Stage 2 · Kaggle only」（對照組）
```

1. **第一階段**：Kaggle 模型用 5-fold 交叉驗證，在訓練集上產生 out-of-fold 分數——每筆的分數都來自**沒看過它**的模型，
   和測試資料的處境一樣；測試集分數來自用整個訓練集訓練的模型。
2. **Jev 特徵**：在最多 2,000 筆分層抽樣的訓練資料上取得 Jev 機率（Titanic、Iris 用全部訓練資料），測試集沿用快取。
   第二階段只有兩個輸入，不需要 IMDB 的 4 萬筆；用 cluster 例子時，被當成例子的訓練列不會進入第二階段。
3. **第二階段**：`StandardScaler + LogisticRegression`，輸入是兩者的 logit（多分類是每個類別一欄），學出各自的權重。
4. **對照組**：同樣的第二階段只用 Kaggle 分數。

#### 為什麼一定要看對照組

重新訓練第二階段本身就會改變決策門檻，分數可能因此變動——那不是 Jev 的功勞。
例如 Iris 光是重新訓練就從 0.933 變成 0.967；若拿 Kaggle + Jev 的 1.000 和原本的 0.933 比，
會誤以為 Jev 貢獻了 0.067，扣掉重新訓練的效果其實只多對 1 朵花。

三個比較各回答一個問題：

| 比較 | 回答的問題 |
| --- | --- |
| **Kaggle + Jev** vs **Stage 2 · Kaggle only** | **Jev 本身有沒有貢獻**（唯一差別就是有沒有 Jev，最重要） |
| **Kaggle + Jev** vs 「Kaggle、Jev 0-shot 中較高者」 | **合起來是否比單獨用任一個好**（值不值得多做這一步） |
| **Stage 2 · Kaggle only** vs **Kaggle** | **重新訓練本身有沒有影響**（這部分不算 Jev 的功勞） |

#### 結果（Accuracy）

| 題目 | n | Kaggle | Jev 0-shot | 對照（只用 Kaggle） | + Jev 0-shot | + Jev cluster | Jev 的貢獻 | 比兩者單獨都好？ | 判讀 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| SMS | 1,115 | 0.985 | 0.979 | 0.985 | **0.992** | 0.991 | +0.007（多對 8 封） | 是 | **互補，最成功的一題** |
| BBC | 445 | 0.987 | 0.982 | 0.987 | 0.989 | **0.991** | +0.004（多對 2 篇） | 是 | 小幅互補 |
| IMDB | 10,000 | 0.918 | 0.963 | 0.918 | **0.964** | **0.964** | +0.046 | 勉強（Jev 單獨 0.963） | 幾乎全靠 Jev，Kaggle 只補 0.001 |
| Titanic | 179 | **0.821** | 0.648 | 0.810 | 0.793 | 0.799 | −0.011（少對 2 人） | 否 | 準確率沒幫助，但 AUC 0.841 → 0.859 |
| Iris | 30 | 0.933 | 0.600 | 0.967 | 0.967 | **1.000** | +0.033（多對 1 朵） | 是 | 只差 1 朵，不能下結論 |

#### Titanic：準確率降、AUC 升

- **AUC 看排序**：把生還機率由高到低排，生還者是否排在死者前面——加了 Jev 後排序變好（0.841 → 0.859）。
- **Accuracy 看門檻**：以 0.5 為界切成生還／死亡，切對幾個——加了 Jev 後多切錯 2 人。

排序變好但切錯變多，代表 Jev 對「誰比較可能生還」有資訊，但剛好在 0.5 附近的幾個人被推到錯的一邊；
179 筆中差 2 人屬於雜訊範圍。結論：**Jev 對 Titanic 的排序有幫助，對分類準確率沒有。**

#### 第二階段的權重

二分類題會在 Run details 記錄 `stage2_weight`。兩個輸入都標準化過，所以權重可以直接比大小：

| 題目 | Kaggle 權重 | Jev 權重 | 意思 |
| --- | ---: | ---: | --- |
| IMDB | 1.17 | 4.20 | 主要相信 Jev |
| SMS | 1.89 | 3.07 | 兩邊都用，偏向 Jev |
| Titanic（cluster） | 1.67 | 0.56 | 主要相信 Kaggle |

哪一邊單獨比較準，第二階段就給它較高的權重。

### 成本

整個實驗（含所有 Jev 方法與 stacking）實際花費 **約 US$2.8**。

| 項目 | 數量 |
| --- | ---: |
| 成功的 Jev 呼叫 | 59,717 次 |
| Input tokens | 6,729 萬 |
| Output tokens | 140 萬 |
| 平均每 100 萬 input tokens | 約 US$0.042 |
| 平均每 1,000 次呼叫 | 約 US$0.047 |

以下依 input tokens 按比例分攤（output tokens 只佔約 2%，所以按 input 分攤；是估算，不是帳單明細）：

| 題目 | 呼叫次數 | Input tokens | 估計花費 |
| --- | ---: | ---: | ---: |
| IMDB | 43,988 | 4,827 萬 | ~US$2.01 |
| BBC News | 5,261 | 1,154 萬 | ~US$0.48 |
| SMS Spam | 8,267 | 575 萬 | ~US$0.24 |
| Titanic | 1,846 | 152 萬 | ~US$0.06 |
| Iris | 355 | 21 萬 | ~US$0.01 |

依方法看（只算測試集，5 題合計）：

| 方法 | Input tokens | 估計花費 | 相對 0-shot |
| --- | ---: | ---: | ---: |
| Jev 0-shot | 715 萬 | ~US$0.30 | ×1 |
| Jev 3-shot | 1,683 萬 | ~US$0.70 | ×2.4 |
| Jev cluster | 1,493 萬 | ~US$0.62 | ×2.1 |
| 隨機對照 | 1,441 萬 | ~US$0.60 | ×2.0 |

其餘約 US$0.58（1,398 萬 tokens）是 stacking 在訓練資料上的呼叫（每題最多 2,000 筆 × 0-shot、cluster 兩種特徵；BBC 的 cluster 請求特別長）。

換算成實際用途：**用 Jev 0-shot 分類 10,000 則 IMDB 影評約 US$0.26**（每則約 620 tokens），
不需要任何訓練資料，準確率 0.963。文字題加例子會讓成本變成 2–3.5 倍，但準確率沒有提升，所以 0-shot 最划算。

### 結論

- **文字分類：直接用 Jev 0-shot。** IMDB 明顯勝過 Kaggle 解法，SMS、BBC 打平，而且完全不需要訓練資料；
  加例子只會增加成本。要榨出最後一點準確率，就把 Jev 機率和 Kaggle 分數做 stacking（SMS、BBC 都超過兩者單獨的表現）。
- **表格／數值題：Kaggle 傳統模型仍然較好。** 若要用 Jev，務必加例子，而且用「分群挑邊緣例子」比隨機挑更好；
  stacking 只改善了 Titanic 的排序（AUC），沒有改善準確率。

### 下一步可以試

1. **門檻校正**：用訓練集（而非測試集）挑 Jev 的決策門檻，針對 Titanic「排序對、門檻偏」的問題。
2. **更多例子／動態例子**：`--shots 10` 看 Titanic 能否繼續進步；或針對每筆測試資料，從訓練集挑最相似的例子（kNN few-shot）。
   Cluster 版本也可以把 `cluster_shots.MAX_K` 調大，讓 Titanic 分出更細的子群。
3. `--model jev-preview` 比較預覽版。

---

## 在 Claude Code 雲端環境執行

API key 以 Bearer credential 存在環境設定（Allowed website `api.typesafe.ai`），由代理在送出時注入，
容器內看不到 key。SDK 仍要求 `TYPESAFE_API_KEY` 有值，所以隨便給一個佔位值即可：

```bash
TYPESAFE_API_KEY=injected-by-proxy jev-bench all --concurrency 32
```
