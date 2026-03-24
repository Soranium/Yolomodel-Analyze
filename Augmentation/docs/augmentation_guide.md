# YOLO セグメンテーション 弱点補強 Augmentation ガイド

モデルの弱点解析結果（`Analyze/model_weakness_report.json`）をもとに、
**苦手な条件を重点的に補強する**Augmentationデータを自動生成するスクリプトの設計書。

ランダムなAugmentationではなく、「弱い部分を狙い撃ちする」のがポイント。

---

## 前提: Analyze からの入力

このスクリプトは `Analyze/model_weakness_report.json` を読み込んで動作する。
そのため **Analyze の弱点解析スクリプトが先に完成している必要がある**。

`model_weakness_report.json` の中身の例:
```json
{
  "weakness_by_size": {
    "tiny(<5%)": {"detection_rate": 0.45}
  },
  "weakness_by_position": {
    "edge": {"detection_rate": 0.61}
  },
  "augmentation_priority": [
    "tiny objects: detection_rate=0.45 → scale + mosaic",
    "edge objects: detection_rate=0.61 → translate"
  ]
}
```

弱点の `detection_rate` が低い条件ほど、Augmentation の生成数・強度を上げる。

---

## 対象データセット

```
SEX-2000image-dataset/
├── data.yaml              ← クラス定義 (sex, penis, vagina)
├── train.txt
├── images/train/          ← 2152 枚 (.png)
└── labels/train/          ← 2152 ファイル (.txt, ポリゴン座標)
```

### ラベル形式

YOLO セグメンテーション形式（1行 = 1オブジェクト）:

```
class_id x1 y1 x2 y2 x3 y3 ...
```

- 座標は 0〜1 に正規化（画像幅・高さで割った値）
- ポリゴン頂点を順番に並べたもの（平均 123 頂点）

---

## データセット分析から判明した課題

| 課題 | 数値 | 影響 |
|------|------|------|
| 極小オブジェクトが多い | tiny 18.7% (面積 < 0.5%) | 小物体の検出漏れ |
| 空間的偏り | spatial_cv = 1.24、中央集中 | 画像端のオブジェクトが学習不足 |
| エッジ接触が多い (cls 1_1) | 48.4% | 切れたオブジェクトの検出精度低下 |
| クラス間の重なり | IoU >= 0.1 が 23.1% | 重なったシーンで分離困難 |
| 極小マスク (ノイズ) | 312 個 (面積 < 0.1%) | 学習の妨げ |
| ポリゴンが複雑 | 平均 123 頂点、最大 1884 | 変換時の精度が重要 |

---

## Augmentation 戦略

### 1. 幾何変換（画像 + ラベル座標を同時変換）

| 変換 | パラメータ | 確率 | 目的 |
|------|-----------|------|------|
| **左右反転** | - | 50% | 位置バリエーション増加 |
| **上下反転** | - | 20% | 端オブジェクト対策 |
| **回転** | -15° 〜 +15° | 40% | 姿勢バリエーション |
| **スケール** | 0.7x 〜 1.4x | 50% | 小物体・大物体の両方に対応 |
| **平行移動** | ±15% | 50% | 空間偏り (spatial_cv=1.24) の補正 |
| **せん断** | -5° 〜 +5° | 20% | 形状バリエーション |

#### ポリゴン座標の変換方法

```
1. 正規化座標 (0〜1) → ピクセル座標に変換
2. アフィン変換行列を構築（回転+スケール+移動+せん断の合成）
3. 行列をポリゴン全頂点に適用
4. 画像範囲 [0, w) × [0, h) にクリップ
5. 変換後のポリゴン面積が10px未満なら除外
6. ピクセル座標 → 正規化座標に戻す
```

### 2. 色変換（画像のみ、ラベル変換不要）

| 変換 | パラメータ | 確率 | 目的 |
|------|-----------|------|------|
| **色相 (H)** | ±0.015 × 180° | 40% | 肌色・照明バリエーション |
| **彩度 (S)** | 0.7x 〜 1.3x | 40% | 色の濃淡バリエーション |
| **明度 (V)** | 0.7x 〜 1.3x | 40% | 明暗バリエーション |
| **ブラー** | kernel 3〜7 | 10% | ボケ画像への耐性 |
| **ガウスノイズ** | std 5〜15 | 10% | ノイズ耐性 |

### 3. Mosaic Augmentation（4枚合成）

```
┌──────────┬──────────┐
│  画像A   │  画像B   │
│          │          │
├──────────┼──────────┤
│  画像C   │  画像D   │
│          │          │
└──────────┴──────────┘
```

- 確率: 30%
- 4枚の画像をランダムに選択し、1枚に合成
- 合成境界はランダム（中心の30〜70%範囲）
- 各画像のラベル座標を合成後の位置に再計算
- **効果**: 小物体の学習機会増加、空間分布の均一化

---

## 出力仕様

### ディレクトリ構造

```
{出力先}/
├── data.yaml              ← 元データセットからコピー
├── train.txt              ← 全画像パスのリスト（再生成）
├── images/train/
│   ├── {元ファイル名}.png         ← 元画像（そのままコピー）
│   ├── {元ファイル名}_aug00.png   ← Augmented #1
│   ├── {元ファイル名}_aug01.png   ← Augmented #2
│   └── {元ファイル名}_aug02.png   ← Augmented #3
└── labels/train/
    ├── {元ファイル名}.txt         ← 元ラベル（そのままコピー）
    ├── {元ファイル名}_aug00.txt   ← 変換済みラベル #1
    ├── {元ファイル名}_aug01.txt   ← 変換済みラベル #2
    └── {元ファイル名}_aug02.txt   ← 変換済みラベル #3
```

### 生成数の目安

| 元画像数 | 生成数/枚 | Augmented | 合計 |
|---------|----------|-----------|------|
| 2152 | 3 | ~6456 | ~8608 |
| 2152 | 5 | ~10760 | ~12912 |

---

## 推奨設定値

データセット分析レポートに基づく推奨パラメータ:

```python
AUG_CONFIG = {
    # 幾何変換
    "flip_lr_prob": 0.5,
    "flip_ud_prob": 0.2,
    "rotate_prob": 0.4,
    "rotate_range": (-15, 15),        # 度
    "scale_prob": 0.5,
    "scale_range": (0.7, 1.4),
    "translate_prob": 0.5,
    "translate_range": (-0.15, 0.15), # 画像比率
    "shear_prob": 0.2,
    "shear_range": (-5, 5),           # 度

    # 色変換
    "hsv_h_range": (-0.015, 0.015),
    "hsv_s_range": (0.7, 1.3),
    "hsv_v_range": (0.7, 1.3),
    "blur_ksize": (3, 7),
    "noise_std": (5, 15),
}
```

---

## 実装時の注意点

### ポリゴン座標変換の精度

- 頂点数が多い（平均123、最大1884）ため、全頂点に行列を正確に適用すること
- `cv2.getRotationMatrix2D` で回転+スケール行列を作り、せん断・移動を合成する
- 変換後は必ず `[0, 1]` にクリップし、面積チェック（10px 未満は除外）

### 除外すべきケース

- 変換後にポリゴンの面積が極小（< 10px）→ ノイズになるため除外
- 変換後に全ラベルが消失した画像 → 出力しない
- 元データの極小マスク（312個, 面積 < 0.1%）→ 事前クリーニング推奨

### Mosaic 合成時の座標計算

```
元の正規化座標 (px, py) → モザイク上の正規化座標:

  nx = (region_x1 + px * region_width) / mosaic_width
  ny = (region_y1 + py * region_height) / mosaic_height
```

### 必要なライブラリ

```
opencv-python (cv2)    - 画像変換、アフィン変換
numpy                  - 座標演算
PyYAML                 - data.yaml 読み書き
```

---

## YOLO 学習時の追加 Augmentation 設定

スクリプトで生成した Augmentation に加え、
YOLO の学習設定でも以下を有効にすると効果的:

```yaml
# YOLOv11 / Ultralytics 学習設定
imgsz: 1024           # 小物体が多いため解像度を上げる
mask_ratio: 2          # マスク解像度 (デフォルト4 → 2)

# 学習時の追加 Augmentation（オフラインと重複しないもの）
mixup: 0.15            # MixUp
copy_paste: 0.3        # Copy-Paste（セグメンテーション用）
erasing: 0.1           # Random Erasing
overlap_mask: true     # 重なりマスク対応
```

---

## 弱点 → Augmentation の自動マッピング

スクリプトは `model_weakness_report.json` を読み込み、
以下のルールで Augmentation 戦略を自動決定する:

### マッピングルール

| 弱点の種類 | detection_rate が低い場合 | 適用するAugmentation | 調整方法 |
|-----------|------------------------|---------------------|---------|
| **小物体** | tiny < 0.6 | Mosaic合成 / Scale拡大 | 小物体画像の生成数を2倍、scale_range上限を1.5xに |
| **画像端** | edge < 0.7 | 平行移動 / 反転 | translate_range を ±20% に拡大 |
| **角** | corner < 0.6 | 大きめの平行移動 + 反転 | translate_range を ±25% に拡大 |
| **特定クラス** | class X < 0.7 | そのクラスの画像を多めに生成 | 生成数を1.5〜2倍に |
| **重なり** | overlap < 0.7 | Copy-Paste で重なりシーンを作成 | 意図的に他オブジェクトを貼り付け |
| **エッジ接触** | edge_touch < 0.7 | 平行移動で端に寄せる | オブジェクトが端に来るようtranslate |

### 生成数の重み付け

```python
# detection_rate が低いほど多く生成する
base_augments = 3  # デフォルト生成数

def calc_weight(detection_rate):
    """弱い条件ほど生成数を増やす"""
    if detection_rate < 0.5:
        return 3.0   # 3倍生成
    elif detection_rate < 0.7:
        return 2.0   # 2倍生成
    elif detection_rate < 0.85:
        return 1.5   # 1.5倍生成
    else:
        return 1.0   # 通常
```

### augmentation_log.json

何をどれだけ生成したかを記録する:

```json
{
  "total_original": 2152,
  "total_augmented": 7530,
  "by_weakness": {
    "tiny_objects": {"target_images": 403, "generated": 1209, "augmentations": ["mosaic", "scale"]},
    "edge_objects": {"target_images": 312, "generated": 624, "augmentations": ["translate", "flip"]},
    "class_vagina": {"target_images": 580, "generated": 870, "augmentations": ["all"]}
  }
}
```

---

## 課題と対策の対応表（データセット分析ベース）

| 分析レポートの課題 | Augmentation 対策 | 設定 |
|-------------------|------------------|------|
| tiny 18.7% | Mosaic + Scale up (1.4x) | mosaic_prob=0.3, scale_range=(0.7, 1.4) |
| spatial_cv=1.24 | 平行移動 | translate_range=±15% |
| edge_touch 48.4% | 上下左右反転 | flip_lr=50%, flip_ud=20% |
| overlap IoU 23.1% | Copy-Paste (学習時) | copy_paste=0.3 |
| 312個の極小マスク | 事前クリーニング | 面積 < 0.1% を除去 |
| 頂点数 平均123 | imgsz=1024 + mask_ratio=2 | 学習設定で対応 |
