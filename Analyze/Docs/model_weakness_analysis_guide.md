# モデル弱点解析スクリプト 設計ガイド

## 概要

YOLOモデルの推論結果と正解ラベルを比較し、
**どんな条件のときに精度が落ちるか**を数値化・グラフ化するスクリプトの設計書。

---

## 処理フロー

```
データセット (images/ + labels/)
        ↓
YOLOモデルで全画像に対して推論
        ↓
正解ラベル vs 推論結果 を1画像ずつ比較
        ↓
条件別に精度を集計（サイズ別、位置別、クラス別...）
        ↓
グラフ出力 + JSON出力
```

---

## 正解と推論のマッチング方法

### IoU（Intersection over Union）で対応づける

正解ポリゴンと推論ポリゴンの重なり度合いで対応を決める:

```
IoU = (重なり面積) / (合計面積 - 重なり面積)
```

- IoU >= 0.5 → 検出成功（True Positive）
- 正解があるのに IoU >= 0.5 の推論がない → 検出漏れ（False Negative）
- 推論があるのに IoU >= 0.5 の正解がない → 誤検出（False Positive）

### ポリゴンのIoU計算

YOLO seg形式はポリゴン座標なので、以下の手順:

1. 正規化座標 → ピクセル座標に変換（画像サイズを掛ける）
2. `cv2.fillPoly()` でマスク画像に変換
3. マスク同士の AND / OR でIoUを計算

```python
import cv2
import numpy as np

def polygon_to_mask(points_normalized, img_w, img_h):
    """正規化ポリゴン座標 → バイナリマスク"""
    pts = np.array([(int(x * img_w), int(y * img_h)) for x, y in points_normalized])
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask

def calc_iou(mask_a, mask_b):
    """2つのマスクのIoUを計算"""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return intersection / union if union > 0 else 0
```

---

## 分析項目の詳細

### A. 全体指標

```python
{
    "detection_rate":     TP / (TP + FN),     # 検出率（再現率）
    "precision":          TP / (TP + FP),     # 精度
    "false_positive_rate": FP / total_predictions,
    "mean_iou":           mean(成功したペアのIoU),
}
```

### B. サイズ別精度

オブジェクトのポリゴン面積（正規化済み）で分類:

| カテゴリ | 面積範囲 | 分析値 |
|---------|---------|--------|
| tiny | < 0.5% | detection_rate, mean_iou |
| small | 0.5% - 5% | detection_rate, mean_iou |
| medium | 5% - 20% | detection_rate, mean_iou |
| large | > 20% | detection_rate, mean_iou |

### C. 位置別精度

オブジェクトの中心座標で分類:

```
画像を3x3のグリッドに分割:

┌────────┬────────┬────────┐
│ corner │  edge  │ corner │  上端
├────────┼────────┼────────┤
│  edge  │ center │  edge  │  中央
├────────┼────────┼────────┤
│ corner │  edge  │ corner │  下端
└────────┴────────┴────────┘
  左端      中央     右端
```

- center: 中心の1/3 × 1/3 の領域
- edge: 端の領域（角以外）
- corner: 四隅

### D. クラス別精度

data.yaml のクラス定義:
```yaml
0: sex
1: penis
2: vagina
```

クラスごとに detection_rate, mean_iou, precision を計算。

### E. 重なり条件別

- 同じ画像内に他のオブジェクトがあるか
- 正解ラベル同士のIoU > 0 があるか

### F. エッジ接触条件別

ポリゴンの座標が画像端（0.0 or 1.0 付近）に触れているか:

```python
def is_edge_touching(points, threshold=0.02):
    for x, y in points:
        if x < threshold or x > (1 - threshold):
            return True
        if y < threshold or y > (1 - threshold):
            return True
    return False
```

---

## 出力ファイル仕様

### model_weakness_report.json

Augmentation スクリプトが読み込む重要なファイル。
以下の構造で出力する:

```json
{
  "overall": { ... },
  "weakness_by_size": { ... },
  "weakness_by_position": { ... },
  "weakness_by_class": { ... },
  "weakness_by_overlap": { ... },
  "weakness_by_edge_touch": { ... },
  "weakness_by_aspect_ratio": { ... },
  "weakness_by_density": { ... },
  "failed_images": ["img001.png", "img002.png", ...],
  "augmentation_priority": [
    "tiny objects: detection_rate=0.45 → scale + mosaic",
    ...
  ]
}
```

### グラフ PNG

matplotlib / seaborn で以下を出力:

1. `weakness_heatmap.png` — 5x5 グリッドの精度ヒートマップ
2. `weakness_by_size.png` — サイズ別精度の棒グラフ
3. `weakness_by_class.png` — クラス別精度の棒グラフ
4. `weakness_confidence.png` — 信頼度 vs IoU の散布図
5. `weakness_overview.png` — 上記をまとめた1枚のダッシュボード

### failed_images.csv

```csv
filename,class,gt_area,iou,confidence,failure_reason
img001.png,0,0.003,0.0,0.0,not_detected
img002.png,2,0.15,0.32,0.45,low_iou
```

---

## 実行方法（想定）

```bash
python analyze_model_weakness.py
```

1. ダイアログでデータセットフォルダを選択
2. ダイアログでYOLOモデル（.pt）を選択
3. 全画像を推論 → 比較 → 集計
4. 同じフォルダにJSON + グラフを出力
