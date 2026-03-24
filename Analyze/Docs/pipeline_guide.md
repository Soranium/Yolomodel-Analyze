# Label Converter パイプライン ガイド

CVAT からエクスポートした YOLO セグメンテーションデータを
学習用に変換・解析・可視化する 3 本のスクリプトで構成されています。

---

## 全体の流れ

```
 CVAT で作成
      |
      v
 ┌─────────────────────────────────┐
 │ 1. labels_converter_MargeFix.py │  統合 & train/val 分割
 │    (入力: CVATエクスポート複数)   │
 └──────────────┬──────────────────┘
                |  出力: 統合済みデータセット
                v
 ┌─────────────────────────────────┐
 │ 2. analyze_integrated_dataset.py│  データセット品質解析
 │    (入力: 統合済みデータセット)   │
 └──────────────┬──────────────────┘
                |  出力: analysis_report.json
                v
 ┌─────────────────────────────────┐
 │ 3. visualize_report.py          │  グラフ画像生成
 │    (入力: analysis_report.json)  │
 └──────────────┬──────────────────┘
                |  出力: report_page1〜3.png
                v
           学習設定の参考にする
```

---

## 1. labels_converter_MargeFix.py

### 何をするスクリプト？

CVAT から「YOLO セグメンテーション形式」でエクスポートすると、
タスクごとにフォルダーが分かれます。
このスクリプトは **複数フォルダーを 1 つに統合** し、
**train / val に自動分割** します。

### 使い方

1. スクリプトを実行
2. ダイアログで **入力フォルダー** を選択（CVATエクスポートが入っている親フォルダー）
3. ダイアログで **出力先フォルダー** を選択

### 入力フォルダーの構造（期待する形）

```
選択するフォルダー/
├── task_A/               ← CVATのタスク別エクスポート
│   ├── data.yaml
│   ├── images/train/     ← 画像
│   └── labels/train/     ← ラベル (.txt)
├── task_B/
│   ├── data.yaml
│   ├── images/train/
│   └── labels/train/
└── ...
```

### 出力されるもの

```
{入力フォルダー名}_integrated_done/
├── data.yaml             ← 全クラスを統合した設定ファイル
├── images/
│   ├── train/            ← 学習用画像 (85%)
│   └── val/              ← 検証用画像 (15%)
└── labels/
    ├── train/            ← 学習用ラベル
    └── val/              ← 検証用ラベル
```

### 処理の詳細

| ステップ | 処理内容 |
|----------|----------|
| 1. 収集 | 全サブフォルダーから画像+ラベルのペアを探す。ラベルが無い画像はスキップ |
| 2. 統合複製 | 全ペアを 1 つのフォルダーにコピー。ファイル名が被る場合は `_1`, `_2` とリネーム |
| 3. 分割 | ランダムに 85% を train、15% を val に振り分け |
| 4. YAML生成 | 全サブフォルダーの data.yaml からクラスを集めて統合した data.yaml を出力 |

### 設定値

- `split_ratio=0.85` — train に割り当てる比率（変更可能）

---

## 2. analyze_integrated_dataset.py

### 何をするスクリプト？

統合済みデータセットの **ラベルを解析** して、
データの偏りや特性を数値化した JSON レポートを出力します。
将来のオーグメンテーション設定を決めるための情報源になります。

### 使い方

1. スクリプトを実行
2. ダイアログで **統合済みデータセット** のフォルダーを選択
3. スクリプトと同じフォルダーに `analysis_report.json` が出力される

### ラベル形式について

CVAT の YOLO セグメンテーション形式は **ポリゴン座標** です。

```
クラスID x1 y1 x2 y2 x3 y3 ...
```

- 座標は 0〜1 に正規化されている（画像サイズで割った値）
- バウンディングボックスではなく、多角形の頂点を順番に並べたもの

例（1行 = 1つのオブジェクト）:
```
1 0.543 0.390 0.534 0.393 0.531 0.396 ...
0 0.515 0.261 0.500 0.286 0.491 0.306 ...
```

### 解析内容の一覧

| 項目 | 説明 | オーグメンテーションへの影響 |
|------|------|------|
| **summary** | 総画像数、総インスタンス数、画像あたりの平均/最大 | データ量の全体把握 |
| **train_val_split** | train/val の画像数・インスタンス数・クラス別 | 分割バランスの確認 |
| **class_detail** | クラスごとの面積・アスペクト比・位置・頂点数・エッジ接触 | クラスごとに戦略を変える判断材料 |
| **size_distribution** | ポリゴン面積の5段階分類 + パーセンタイル | Scale augmentation の範囲決定 |
| **aspect_ratio** | 縦長/正方形/横長 の比率 | Resize/Stretch の制限値 |
| **spatial_distribution** | 5x5 グリッドでの出現頻度 + 中心位置統計 | Translation/Affine の必要性 |
| **polygon_complexity** | 頂点数の統計（平均/最大/パーセンタイル） | マスク解像度の設定 |
| **edge_touch** | 画像端に触れる割合（上下左右の辺別） | Crop/Padding の方向別設定 |
| **overlap_analysis** | クラスペアごとの重なり(IoU)統計 | Copy-Paste augmentation の適用判断 |
| **image_density_distribution** | 1画像に何個のオブジェクトがあるかの分布 | Mosaic augmentation の適用判断 |
| **augmentation_hints** | 上記の結果から自動生成される推奨事項 | そのまま設定の参考に |

---

## 3. visualize_report.py

### 何をするスクリプト？

`analysis_report.json` を読み込んで、
**3 ページ分のグラフ画像 (PNG)** を生成します。
数値だけでは把握しにくいデータの傾向を視覚的に確認できます。

### 使い方

1. スクリプトを実行
2. ダイアログで `analysis_report.json` を選択
3. スクリプトと同じフォルダーに 3 枚の PNG が出力される

### 出力される画像

#### report_page1_overview.png（概要）
- データセットの基本情報テキスト
- クラス分布の棒グラフ
- Train/Val のクラス別比較
- ポリゴン面積のサイズ分布（5段階）
- ポリゴン面積のパーセンタイル折れ線
- アスペクト比の分布
- アスペクト比の統計値
- BBox 面積のパーセンタイル

#### report_page2_spatial.png（空間・重なり）
- 5x5 空間ヒートマップ（オブジェクトがどこに集中しているか）
- 空間中心の統計値
- 辺別エッジ接触の棒グラフ（上下左右どこに多いか）
- クラスペア別の重なり解析（IoU 0.1/0.3/0.5 閾値）
- 画像密度分布（1画像あたり何個あるか）
- ポリゴン複雑度の統計値
- オーグメンテーション推奨事項

#### report_page3_class_detail.png（クラス別詳細）
- クラス別ポリゴン面積（エラーバー付き）
- クラス別 BBox 面積（エラーバー付き）
- クラス別アスペクト比（エラーバー付き）
- クラス別エッジ接触率
- クラス別頂点数（エラーバー付き）
- クラス別中心位置の散布図（標準偏差の円付き）
- クラス別サマリーテーブル

---

## analysis_report.json の構造

```
{
  "summary": {                         ← 全体の概要
    "total_images": 4909,
    "total_instances": 12046,
    ...
  },
  "train_val_split": {                 ← train/val の内訳
    "train": { "images": 4172, ... },
    "val":   { "images": 737,  ... }
  },
  "class_detail": {                    ← クラスごとの詳細統計
    "0_クラス名": {
      "polygon_area": { ... },
      "bbox_area": { ... },
      "aspect_ratio": { ... },
      "center_position": { ... },
      "vertex_count": { ... },
      "edge_touch_ratio": 0.09
    },
    ...
  },
  "size_distribution": { ... },        ← 全体のサイズ分布
  "aspect_ratio": { ... },             ← 全体のアスペクト比
  "spatial_distribution": {            ← 空間分布
    "grid_5x5_counts": [[...], ...]    ← 5x5のヒートマップ数値
  },
  "polygon_complexity": { ... },       ← ポリゴン頂点数の統計
  "edge_touch": {                      ← 画像端への接触
    "by_side": { "top": ..., ... },    ← 辺ごとの件数
    "by_side_ratio": { ... }           ← 辺ごとの割合
  },
  "overlap_analysis": {                ← 重なり解析
    "0-1": { ... },                    ← クラス0とクラス1のペア
    ...
  },
  "image_density_distribution": {...}, ← 画像密度分布
  "augmentation_hints": [...]          ← 自動生成された推奨事項
}
```

---

## フォルダー構成

```
.label-converter/
├── Docs/
│   └── pipeline_guide.md     ← このファイル
└── YoloSegment/
    ├── labels_converter_MargeFix.py      ← Step1: 統合 & 分割
    ├── analyze_integrated_dataset.py     ← Step2: 解析
    ├── visualize_report.py               ← Step3: グラフ生成
    ├── analysis_report.json              ← 解析結果
    ├── report_page1_overview.png         ← グラフ: 概要
    ├── report_page2_spatial.png          ← グラフ: 空間・重なり
    ├── report_page3_class_detail.png     ← グラフ: クラス別詳細
    ├── import/                           ← CVATエクスポートの生データ
    └── export/                           ← 変換後の出力データ
```
