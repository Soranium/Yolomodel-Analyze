# Minami さんへ — 作業依頼書

## はじめに

このプロジェクトは、YOLOモデルのセグメンテーション精度を改善するためのものです。
3つのフォルダがあり、それぞれ役割が違います。

**全体の流れ:**
```
Debug（目視確認）→ Analyze（弱点を数値化）→ Augmentation（弱点を補強するデータ生成）
```

つまり:
1. まず Debug で「モデルがどこを間違えているか」を動画で目視確認する
2. 次に Analyze で「どんな画角・条件で精度が落ちるか」を数値とグラフで明らかにする
3. 最後に Augmentation で「弱い部分を重点的に補強するデータ」を自動生成する

---

## 環境の準備

### 1. Parsec でリモート接続

このPCに Parsec でリモート接続して作業してもらいます。
- Parsec: https://parsec.app/
- 接続先の情報は別途お伝えします

### 2. リポジトリのクローン

ターミナル（PowerShell や Git Bash）で以下を実行:

```bash
git clone https://github.com/Soranium/Yolomodel-Analyze.git
cd Yolomodel-Analyze
```

### 3. データセットのダウンロード

以下のGoogle Driveリンクからデータセットをダウンロードしてください:

https://drive.google.com/file/d/1AmIHgb3lOlqunrvJ8gApuw8PCe6C9DNt/view?usp=sharing

ダウンロードしたZIPを解凍して、以下の場所に配置:

```
Yolomodel-Analyze/
└── Augmentation/
    └── SEX-2000image-dataset/    ← ここに解凍して配置
        ├── data.yaml
        ├── train.txt
        ├── images/train/          ← 画像 2152枚
        └── labels/train/          ← ラベル 2152ファイル
```

### 4. Python 環境

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install ultralytics opencv-python numpy pyyaml rich psutil av matplotlib seaborn pandas
```

---

## プロジェクト全体像

```
Yolomodel-Analyze/
├── Debug/          ← モデル精度の目視確認ツール（既にある）
├── Analyze/        ← 【作業①】モデルの弱点を解析するスクリプトを作る
└── Augmentation/   ← 【作業②】弱点を補強するデータを自動生成するスクリプトを作る
```

---

## フォルダ別の説明

---

### Debug/ — モデル精度の目視確認ツール（既存・作業なし）

**これは何？**

動画を入力すると、YOLOモデルが各フレームで対象物を検出し、
検出結果をマスクで重ねた動画を出力するツールです。
「モデルがちゃんと検出できてるかな？」を目で見て確認するためのものです。

**既に完成しています。改修の必要はありません。**

**使い方:**
1. `import/` に確認したいMP4動画を置く
2. `mosaic_yolov26_yoloonly_debug.py` を実行
3. `export/` に出力された動画を見て確認

**確認のポイント:**
- マスクが対象物にちゃんと重なっているか
- 検出漏れ（マスクがつかないフレーム）はどんな場面で起きるか
- 誤検出（対象じゃない場所にマスクがつく）はどんな場面で起きるか

→ ここで気づいた「弱い場面」が、次の Analyze の開発に活きます

**詳細:** `docs/debug_pipeline_guide.md` を参照

---

### Analyze/ — 【作業①】モデルの弱点解析スクリプトの開発

**これは何？**

Debug で目視確認した「モデルが苦手な場面」を、
**数値とグラフで客観的に明らかにする**ためのスクリプトを作ります。

**既存ファイル（参考にしてください）:**

| ファイル | 役割 |
|---------|------|
| `labels_converter_MargeFix.py` | CVATデータの統合ツール（既存） |
| `analyze_integrated_dataset.py` | データセットの統計分析（既存） |
| `visualize_report.py` | 分析結果のグラフ化（既存） |
| `analysis_report.json` | 既存の分析結果 |

上記は「データセット自体」の分析ツールです。
今回新しく作るのは **「モデルの推論結果」を分析するスクリプト** です。

---

**やってほしいこと: モデル弱点解析スクリプトの新規作成**

データセットの画像に対してYOLOモデルで推論を実行し、
**正解ラベルと推論結果を比較して、どこが弱いかを分析・グラフ化する**スクリプトを作ってください。

#### 入力
- 画像フォルダ（`images/train/`）
- 正解ラベル（`labels/train/`）— YOLO seg 形式
- YOLOモデル（`.pt` ファイル）

#### 分析すべき項目

**A. フレーム単位の精度分析**

| 分析項目 | 説明 | なぜ必要か |
|---------|------|-----------|
| 検出成功率 | 正解があるのに検出できなかった画像の割合 | 全体の精度把握 |
| 誤検出率 | 正解がないのに検出してしまった画像の割合 | 偽陽性の把握 |
| IoU分布 | 正解マスクと推論マスクの重なり具合の分布 | マスク精度の把握 |
| 信頼度分布 | 検出された物体の信頼度スコアの分布 | 閾値チューニングの参考 |

**B. 弱い条件の特定（これが最重要）**

「どんな条件のとき精度が落ちるか」を明らかにする:

| 条件 | 具体的に何を見るか |
|------|------------------|
| **オブジェクトのサイズ** | 小さいオブジェクト vs 大きいオブジェクトで精度が違うか |
| **画像内の位置** | 画面の端 vs 中央で精度が違うか |
| **オブジェクトの数** | 1個だけの画像 vs 密集している画像で精度が違うか |
| **クラス別** | sex / penis / vagina のどれが苦手か |
| **アスペクト比** | 縦長 vs 横長 vs 正方形で精度が違うか |
| **重なり** | 他のオブジェクトと重なっているとき精度が落ちるか |
| **画像端の接触** | オブジェクトが画面端で切れているとき精度が落ちるか |

**C. グラフ出力**

以下のグラフをPNG画像として出力してください:

1. **検出精度ヒートマップ** — 画像のどの位置が得意/苦手かを5x5グリッドで色分け
2. **サイズ別 検出率グラフ** — オブジェクトの大きさ別の検出成功率（棒グラフ）
3. **クラス別 精度比較** — 各クラスのIoU平均・検出率（棒グラフ）
4. **信頼度 vs IoU 散布図** — 信頼度が高い＝精度が高いか確認
5. **失敗画像一覧** — 精度が特に低い画像のファイル名リスト（CSV出力）
6. **弱点サマリー** — 「こういう条件で精度が落ちる」を一覧にしたテキスト

**D. 出力形式**

```
Analyze/
├── model_weakness_report.json    ← 数値データ（Augmentationスクリプトが読む）
├── weakness_heatmap.png          ← 位置別の精度ヒートマップ
├── weakness_by_size.png          ← サイズ別の精度グラフ
├── weakness_by_class.png         ← クラス別の精度グラフ
├── weakness_confidence.png       ← 信頼度 vs IoU
├── failed_images.csv             ← 精度が低い画像リスト
└── weakness_summary.txt          ← 弱点の要約
```

**`model_weakness_report.json` の構造（例）:**

これが Augmentation に渡される重要なファイルです。
「どういう条件の画像を重点的に増やすべきか」がわかるようにしてください:

```json
{
  "overall": {
    "detection_rate": 0.85,
    "mean_iou": 0.62,
    "false_positive_rate": 0.08
  },
  "weakness_by_size": {
    "tiny(<5%)": {"detection_rate": 0.45, "mean_iou": 0.35},
    "small(5-15%)": {"detection_rate": 0.72, "mean_iou": 0.58},
    "medium(15-40%)": {"detection_rate": 0.91, "mean_iou": 0.74},
    "large(>40%)": {"detection_rate": 0.95, "mean_iou": 0.82}
  },
  "weakness_by_position": {
    "center": {"detection_rate": 0.92},
    "edge": {"detection_rate": 0.61},
    "corner": {"detection_rate": 0.48}
  },
  "weakness_by_class": {
    "sex": {"detection_rate": 0.88, "mean_iou": 0.65},
    "penis": {"detection_rate": 0.82, "mean_iou": 0.60},
    "vagina": {"detection_rate": 0.79, "mean_iou": 0.55}
  },
  "weakness_by_overlap": {
    "no_overlap": {"detection_rate": 0.90},
    "with_overlap": {"detection_rate": 0.65}
  },
  "augmentation_priority": [
    "tiny objects: detection_rate=0.45 → scale augmentation + mosaic 優先",
    "edge/corner objects: detection_rate=0.48 → translate augmentation 優先",
    "overlapping objects: detection_rate=0.65 → copy-paste augmentation 優先"
  ]
}
```

---

### Augmentation/ — 【作業②】弱点補強データ自動生成スクリプトの開発

**これは何？**

Analyze で出力された `model_weakness_report.json` を読み込んで、
**モデルが苦手な条件に特化したAugmentationデータを自動生成する**スクリプトを作ります。

ただのランダムなAugmentationではなく、
**「弱い部分を重点的に補強する」スマートなAugmentation** です。

---

**やってほしいこと:**

#### 入力
- `SEX-2000image-dataset/` の画像 + ラベル
- `Analyze/model_weakness_report.json` — モデルの弱点データ

#### 処理の流れ

```
model_weakness_report.json を読む
        ↓
弱点に応じてAugmentation戦略を自動決定
        ↓
  例: 小物体の検出率が低い
      → 小物体を含む画像を多めにAugmentation
      → Scale拡大で小物体を大きくした画像を生成
      → Mosaicで小物体の学習機会を増やす
        ↓
  例: 画像端の検出率が低い
      → 平行移動で中央のオブジェクトを端にずらした画像を生成
      → 反転で左右・上下のバリエーションを増やす
        ↓
  例: 重なりがあると精度が落ちる
      → Copy-Paste で意図的に重なりを作った画像を生成
        ↓
画像 + 変換済みラベルを出力
```

#### 弱点 → Augmentation の対応表

| 弱点 | 適用するAugmentation | 重みの上げ方 |
|------|---------------------|-------------|
| 小物体が苦手 | Mosaic合成 / Scale拡大 | 小物体を含む画像の生成数を2倍に |
| 画像端が苦手 | 平行移動 / 反転 | translate量を大きく(±20%) |
| 特定クラスが苦手 | そのクラスの画像を多めに生成 | 生成数を1.5倍に |
| 重なりが苦手 | Copy-Paste（別の画像から貼り付け） | 重なりシーンを意図的に生成 |
| 暗い/ボケた画像が苦手 | 明度変換 / ブラー追加 | 色変換の範囲を広げる |

#### ラベル形式（重要）

ラベルファイル（.txt）の1行:
```
クラスID x1 y1 x2 y2 x3 y3 ...
```
- 座標は 0〜1 の正規化値（画像サイズで割った値）
- 多角形の頂点を順番に並べたもの
- **画像を変換したら、この座標も必ず同じ変換をかけること**

例: 画像を左右反転したら → 各座標の x を `1.0 - x` に変換

#### 出力形式

```
{出力先}/
├── data.yaml
├── train.txt
├── augmentation_log.json         ← 何をどれだけ生成したかの記録
├── images/train/
│   ├── 元画像.png                ← 元画像（そのままコピー）
│   ├── 元画像_aug00.png          ← 生成画像
│   └── ...
└── labels/train/
    ├── 元画像.txt                ← 元ラベル
    ├── 元画像_aug00.txt          ← 変換済みラベル
    └── ...
```

#### 注意点
- 変換後にポリゴンが画像外にはみ出たら `[0, 1]` にクリップする
- 変換後にオブジェクトの面積が極小になったら、そのラベルは除外する
- 元画像は必ずそのままコピーして含める
- `augmentation_log.json` に「どの弱点に対して何枚生成したか」を記録する

**詳細な設計は `docs/augmentation_guide.md` を参照してください。**

---

## 作業の順番

```
Step 1: Debug の出力動画を見て、モデルの苦手な場面を把握する
            ↓
Step 2: 【作業①】Analyze/ に弱点解析スクリプトを作る
         → model_weakness_report.json とグラフが出力される
            ↓
Step 3: 【作業②】Augmentation/ に弱点補強スクリプトを作る
         → model_weakness_report.json を読んで、弱い部分を
           重点的に補強するデータを生成する
```

**作業①が先です。** 作業②は作業①の出力（`model_weakness_report.json`）を使うため、
作業①が完成してから取りかかってください。

---

## 質問・連絡先

わからないことがあれば遠慮なく聞いてください。
各フォルダの `docs/` にある `.md` ファイルに詳しい説明があるので、
まずそちらを確認してもらえると助かります。
