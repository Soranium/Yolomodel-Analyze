# Minami さんへ — 作業依頼書

## はじめに

このドキュメントは、YOLOモデルのセグメンテーション精度を改善するプロジェクトの作業依頼です。
3つのフォルダに分かれており、それぞれ独立した作業になっています。
わからないことがあれば気軽に聞いてください。

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
pip install ultralytics opencv-python numpy pyyaml rich psutil av
```

---

## プロジェクト全体像

```
Yolomodel-Analyze/
├── Analyze/        ← 【済】データセットの分析ツール（既に完成）
├── Augmentation/   ← 【作業①】データを増やすスクリプトを作る
└── Debug/          ← 【作業②】推論の精度を上げるスクリプトを改善する
```

**目的**: YOLOモデルが動画の中の対象物をもっと正確に検出できるようにする

---

## フォルダ別の作業内容

---

### Analyze/ — データセット分析ツール（参照用・作業なし）

**これは何？**
学習データの中身を分析して「どんな問題があるか」をレポートにまとめるツールです。
**既に完成しています。作業は不要です。** ただし分析結果は他の作業の参考になります。

**中身:**
| ファイル | 役割 |
|---------|------|
| `labels_converter_MargeFix.py` | CVATからエクスポートしたデータを統合・分割するツール |
| `analyze_integrated_dataset.py` | データセットの偏りや問題点を数値で分析するツール |
| `visualize_report.py` | 分析結果をグラフ画像にするツール |
| `analysis_report.json` | 分析結果のデータ（これが重要） |
| `Docs/pipeline_guide.md` | 上記ツールの使い方ドキュメント |

**分析で見つかった問題（これを頭に入れて作業してください）:**
- 小さいオブジェクトが 18.7% ある → 見逃しやすい
- オブジェクトが画像の真ん中に偏っている → 端にあるものが苦手になる
- オブジェクト同士が重なっているケースが 23% ある → 分離が難しい
- ゴミみたいな極小ラベル（ノイズ）が 312個ある → 学習の邪魔になる

---

### Augmentation/ — 【作業①】データ増強スクリプトの開発

**これは何？**

「データ増強（Augmentation）」とは、元の画像を回転・反転・色変換などして、
学習データのバリエーションを増やすことです。
データが多いほどモデルは賢くなります。

**やってほしいこと:**

Pythonスクリプトを1本作って、以下ができるようにしてください:

1. `SEX-2000image-dataset/` の画像とラベルを読み込む
2. 画像に変換を加えて新しい画像を生成する
3. **ラベル（ポリゴン座標）も画像と同じ変換をかけて正しく更新する**
4. 元画像 + 生成画像をまとめて出力する

**ラベルの形式（これが大事）:**

ラベルファイル（.txt）の1行はこういう形:
```
クラスID x1 y1 x2 y2 x3 y3 ...
```
例: `0 0.437 0.570 0.434 0.572 0.429 0.575 ...`

- 数字は画像の幅・高さで割った0〜1の値
- 多角形の頂点を順番に並べたもの
- 画像を回転させたら、この座標も同じだけ回転させる必要がある

**実装すべき変換:**

| 変換 | 何をするか | なぜ必要か |
|------|-----------|-----------|
| 左右反転 | 画像を鏡のように反転 | 位置のバリエーションを増やす |
| 上下反転 | 画像を上下ひっくり返す | 端にあるオブジェクトの学習強化 |
| 回転 | -15°〜+15° 傾ける | 角度のバリエーション |
| 拡大縮小 | 0.7倍〜1.4倍 | 小さいオブジェクト対策 |
| 平行移動 | 上下左右に ±15% ずらす | 画像中央への偏り対策 |
| 色変換 | 明るさ・色合いを変える | 照明条件のバリエーション |
| Mosaic | 4枚を1枚に合成 | 小物体の学習機会を大幅に増やす |

**詳細な設計は `docs/augmentation_guide.md` に書いてあります。必ず読んでください。**

**出力形式:**
```
{出力先}/
├── data.yaml
├── train.txt
├── images/train/
│   ├── 元画像.png
│   ├── 元画像_aug00.png    ← 生成画像
│   ├── 元画像_aug01.png
│   └── 元画像_aug02.png
└── labels/train/
    ├── 元画像.txt
    ├── 元画像_aug00.txt    ← 変換済みラベル
    ├── 元画像_aug01.txt
    └── 元画像_aug02.txt
```

**注意点:**
- 変換後にポリゴンが画像の外にはみ出たら、画像の端でクリップ（切り取り）する
- 変換後にオブジェクトが消滅（面積が極小）したら、そのラベルは除外する
- 元画像はそのままコピーして含める（Augmentedだけにしない）

---

### Debug/ — 【作業②】推論スクリプトの精度改善

**これは何？**

動画ファイルを入力すると、YOLOモデルが各フレームで対象物を検出し、
検出結果を緑色のマスクで重ねた動画を出力するツールです。
**既にあるスクリプト `mosaic_yolov26_yoloonly_debug.py` を改善してください。**

**やってほしいこと:**

以下の改善を `mosaic_yolov26_yoloonly_debug.py` に加えてください:

#### 改善1: 信頼度閾値の最適化

現在の値:
```python
CONF_THRESHOLD = 0.15   # 低すぎて偽陽性（誤検出）が多い
YOLO_CONF = 0.1
```

改善:
```python
CONF_THRESHOLD = 0.25   # 偽陽性を減らす
YOLO_CONF = 0.15
```

#### 改善2: NMS（重複検出の除去）設定追加

重なったオブジェクトが多いため、NMSの設定を追加:
```python
results = self.yolo.predict(
    img, verbose=False, device=self.device,
    conf=YOLO_CONF,
    iou=0.5,             # ← 追加: NMSの閾値
    agnostic_nms=True     # ← 追加: クラスをまたいだNMS
)
```

#### 改善3: 最大検出数の緩和

```python
MAX_ANCHOR_OBJECTS = 5   # 現在: 最大5個
# ↓
MAX_ANCHOR_OBJECTS = 10  # 改善: 密集シーンに対応
```

#### 改善4: マスク補間の改善

```python
# 現在: bilinear（カクカクしやすい）
up = F.interpolate(mask_f, size=(H, W), mode="bilinear", align_corners=False)
# ↓
# 改善: bicubic（マスク境界が滑らか）
up = F.interpolate(mask_f, size=(H, W), mode="bicubic", align_corners=False)
```

#### 改善5: クラス別の色分け表示

現在は全部緑色で表示していますが、クラスごとに色を変えると
どのクラスが正しく検出できているか確認しやすくなります:

```python
CLASS_COLORS = {
    0: (0, 255, 0),    # sex → 緑
    1: (255, 0, 0),    # penis → 青 (BGRなので)
    2: (0, 0, 255),    # vagina → 赤
}
```

**詳細は `docs/debug_pipeline_guide.md` を参照してください。**

**テスト方法:**
1. `import/` フォルダにMP4動画を置く
2. スクリプトを実行
3. ダイアログで `import/` と `export/` を選択
4. 出力動画を見て、マスクが対象物に正しく重なっているか確認

---

## 作業の優先順位

```
① Augmentation スクリプトの開発（最優先）
   → これが完成しないと学習データを増やせない

② Debug スクリプトの改善
   → ①と並行して進められる
```

---

## 質問・連絡先

わからないことがあれば遠慮なく聞いてください。
各フォルダの `docs/` にある `.md` ファイルに詳しい説明があるので、
まずそちらを確認してもらえると助かります。
