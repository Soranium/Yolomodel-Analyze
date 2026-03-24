# labels_analyzer_integrated.py
# 統合済みYOLOセグメンテーションデータセットを解析するGUI版
# Augmentation計画用の詳細解析対応

import os
import sys
import time
import json
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import numpy as np
import yaml

TOTAL_STEPS = 6
current_step = 0

def print_progress_inline():
    filled = int(25 * current_step / TOTAL_STEPS)
    bar = "█" * filled + "→" + "-" * (25 - filled)
    percent = int(current_step / TOTAL_STEPS * 100)
    sys.stdout.write(f"\r[{bar}] {percent}%")
    sys.stdout.flush()

def select_directory(title: str):
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title=title)
    root.destroy()
    return directory

def polygon_area_shoelace(xs, ys):
    """Shoelace公式でポリゴン面積を計算（正規化座標）"""
    n = len(xs)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j]
        area -= xs[j] * ys[i]
    return abs(area) / 2.0

def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)

def parse_yolo_seg_line(line):
    """
    YOLOセグメンテーション行をパース
    形式: class_id x1 y1 x2 y2 x3 y3 ...
    Returns: (class_id, xs_list, ys_list) or None
    """
    parts = line.split()
    if len(parts) < 7:  # class + 最低3頂点(6座標)
        return None
    cls = int(parts[0])
    coords = list(map(float, parts[1:]))
    if len(coords) % 2 != 0:
        return None
    xs = coords[0::2]
    ys = coords[1::2]
    return cls, xs, ys

def analyze_dataset(dataset_dir, output_dir):
    global current_step
    print_progress_inline()
    time.sleep(0.2)

    dataset_path = Path(dataset_dir)

    # data.yaml からクラス名を読み込み
    class_names = {}
    data_yaml = dataset_path / "data.yaml"
    if data_yaml.exists():
        with open(data_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        names = data.get("names", {})
        if isinstance(names, dict):
            class_names = {int(k): v for k, v in names.items()}
        elif isinstance(names, list):
            class_names = {i: v for i, v in enumerate(names)}

    # train / val 両方のラベルを取得
    label_files_train = list(dataset_path.glob("labels/train/*.txt"))
    label_files_val = list(dataset_path.glob("labels/val/*.txt"))
    label_files_all = label_files_train + label_files_val

    if not label_files_all:
        print("\033[31mラベルが見つかりません。\033[0m")
        return

    current_step += 1
    print_progress_inline()

    # --- データ収集用リスト ---
    # 全体
    all_polygon_areas = []
    all_bbox_areas = []
    all_aspect_ratios = []
    all_center_xs = []
    all_center_ys = []
    all_vertex_counts = []
    all_edge_flags = []
    all_edge_sides = {"top": 0, "bottom": 0, "left": 0, "right": 0}
    per_image_counts = []
    all_overlaps = []

    # クラス別
    per_class_polygon_areas = {}
    per_class_bbox_areas = {}
    per_class_aspect_ratios = {}
    per_class_center_xs = {}
    per_class_center_ys = {}
    per_class_vertex_counts = {}
    per_class_edge_count = {}
    per_class_count = {}

    # train/val 別カウント
    split_counts = {"train": {"images": len(label_files_train), "instances": 0, "per_class": {}},
                    "val":   {"images": len(label_files_val),   "instances": 0, "per_class": {}}}

    # 画像密度分布
    density_bins = []  # 1画像あたりのインスタンス数

    # 空間グリッド (5x5) — オブジェクト中心の出現頻度
    grid_size = 5
    spatial_grid = np.zeros((grid_size, grid_size), dtype=int)

    current_step += 1
    print_progress_inline()

    # --- パース & 集計 ---
    for file in label_files_all:
        is_train = "train" in file.parts
        split_key = "train" if is_train else "val"

        lines = file.read_text().strip().splitlines()
        boxes = []
        box_classes = []
        img_instance_count = 0

        for line in lines:
            parsed = parse_yolo_seg_line(line)
            if parsed is None:
                continue

            cls, xs, ys = parsed
            img_instance_count += 1

            # ポリゴン面積 (Shoelace)
            poly_area = polygon_area_shoelace(xs, ys)
            all_polygon_areas.append(poly_area)

            # バウンディングボックス
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            bbox_w = x_max - x_min
            bbox_h = y_max - y_min
            bbox_area = bbox_w * bbox_h
            all_bbox_areas.append(bbox_area)

            # アスペクト比
            aspect = (bbox_w / bbox_h) if bbox_h > 0 else 0
            all_aspect_ratios.append(aspect)

            # 中心座標
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            all_center_xs.append(cx)
            all_center_ys.append(cy)

            # 空間グリッド
            gx = min(int(cx * grid_size), grid_size - 1)
            gy = min(int(cy * grid_size), grid_size - 1)
            spatial_grid[gy][gx] += 1

            # 頂点数
            n_verts = len(xs)
            all_vertex_counts.append(n_verts)

            # エッジ接触 (ポリゴン頂点ベース)
            touch_left   = any(x < 0.02 for x in xs)
            touch_right  = any(x > 0.98 for x in xs)
            touch_top    = any(y < 0.02 for y in ys)
            touch_bottom = any(y > 0.98 for y in ys)
            is_edge = touch_left or touch_right or touch_top or touch_bottom
            all_edge_flags.append(is_edge)
            if touch_top:    all_edge_sides["top"] += 1
            if touch_bottom: all_edge_sides["bottom"] += 1
            if touch_left:   all_edge_sides["left"] += 1
            if touch_right:  all_edge_sides["right"] += 1

            # BBox for IoU
            boxes.append((x_min, y_min, x_max, y_max))
            box_classes.append(cls)

            # --- クラス別集計 ---
            per_class_count[cls] = per_class_count.get(cls, 0) + 1
            per_class_polygon_areas.setdefault(cls, []).append(poly_area)
            per_class_bbox_areas.setdefault(cls, []).append(bbox_area)
            per_class_aspect_ratios.setdefault(cls, []).append(aspect)
            per_class_center_xs.setdefault(cls, []).append(cx)
            per_class_center_ys.setdefault(cls, []).append(cy)
            per_class_vertex_counts.setdefault(cls, []).append(n_verts)
            if is_edge:
                per_class_edge_count[cls] = per_class_edge_count.get(cls, 0) + 1

            # split別
            split_counts[split_key]["instances"] += 1
            split_counts[split_key]["per_class"][cls] = split_counts[split_key]["per_class"].get(cls, 0) + 1

        per_image_counts.append(img_instance_count)
        density_bins.append(img_instance_count)

        # IoU (同一画像内)
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                ov = bbox_iou(boxes[i], boxes[j])
                all_overlaps.append({
                    "iou": ov,
                    "cls_pair": tuple(sorted([box_classes[i], box_classes[j]]))
                })

    current_step += 1
    print_progress_inline()

    # --- numpy変換 ---
    polygon_areas = np.array(all_polygon_areas)
    bbox_areas = np.array(all_bbox_areas)
    aspect_ratios = np.array(all_aspect_ratios)
    center_xs = np.array(all_center_xs)
    center_ys = np.array(all_center_ys)
    vertex_counts = np.array(all_vertex_counts)
    per_image_arr = np.array(per_image_counts)
    iou_values = np.array([o["iou"] for o in all_overlaps]) if all_overlaps else np.array([])

    total = len(polygon_areas)

    # --- クラス別レポート ---
    per_class_report = {}
    for cls in sorted(per_class_count.keys()):
        name = class_names.get(cls, f"class_{cls}")
        pa = np.array(per_class_polygon_areas[cls])
        ba = np.array(per_class_bbox_areas[cls])
        ar = np.array(per_class_aspect_ratios[cls])
        vc = np.array(per_class_vertex_counts[cls])
        cxs = np.array(per_class_center_xs[cls])
        cys = np.array(per_class_center_ys[cls])
        edge_cnt = per_class_edge_count.get(cls, 0)
        cnt = per_class_count[cls]

        per_class_report[f"{cls}_{name}"] = {
            "count": cnt,
            "ratio": round(cnt / total, 4),
            "polygon_area": {
                "mean": round(float(pa.mean()), 6),
                "std":  round(float(pa.std()), 6),
                "p10":  round(float(np.percentile(pa, 10)), 6),
                "p50":  round(float(np.percentile(pa, 50)), 6),
                "p90":  round(float(np.percentile(pa, 90)), 6),
            },
            "bbox_area": {
                "mean": round(float(ba.mean()), 6),
                "p10":  round(float(np.percentile(ba, 10)), 6),
                "p50":  round(float(np.percentile(ba, 50)), 6),
                "p90":  round(float(np.percentile(ba, 90)), 6),
            },
            "aspect_ratio": {
                "mean": round(float(ar.mean()), 4),
                "std":  round(float(ar.std()), 4),
                "min":  round(float(ar.min()), 4),
                "max":  round(float(ar.max()), 4),
            },
            "center_position": {
                "mean_x": round(float(cxs.mean()), 4),
                "mean_y": round(float(cys.mean()), 4),
                "std_x":  round(float(cxs.std()), 4),
                "std_y":  round(float(cys.std()), 4),
            },
            "vertex_count": {
                "mean": round(float(vc.mean()), 1),
                "min":  int(vc.min()),
                "max":  int(vc.max()),
            },
            "edge_touch_ratio": round(edge_cnt / cnt, 4) if cnt > 0 else 0,
        }

    # --- 重なり解析（クラスペア別） ---
    overlap_by_pair = {}
    for o in all_overlaps:
        pair_key = f"{o['cls_pair'][0]}-{o['cls_pair'][1]}"
        overlap_by_pair.setdefault(pair_key, []).append(o["iou"])

    overlap_report = {}
    for pair_key, ious in overlap_by_pair.items():
        ious_arr = np.array(ious)
        overlap_report[pair_key] = {
            "total_pairs": len(ious_arr),
            "mean_iou": round(float(ious_arr.mean()), 4),
            "ratio_iou>=0.1": round(float(np.mean(ious_arr >= 0.1)), 4),
            "ratio_iou>=0.3": round(float(np.mean(ious_arr >= 0.3)), 4),
            "ratio_iou>=0.5": round(float(np.mean(ious_arr >= 0.5)), 4),
        }

    # --- 画像密度分布 ---
    density_arr = np.array(density_bins)
    density_distribution = {}
    for threshold in [0, 1, 2, 3, 4, 5]:
        if threshold == 0:
            density_distribution["0_instances"] = int(np.sum(density_arr == 0))
        elif threshold < 5:
            density_distribution[f"{threshold}_instances"] = int(np.sum(density_arr == threshold))
        else:
            density_distribution[f"{threshold}+_instances"] = int(np.sum(density_arr >= threshold))

    current_step += 1
    print_progress_inline()

    # --- レポート構築 ---
    report = {
        "summary": {
            "total_images": len(label_files_all),
            "total_instances": total,
            "mean_instances_per_image": round(float(per_image_arr.mean()), 2),
            "std_instances_per_image":  round(float(per_image_arr.std()), 2),
            "max_instances_per_image":  int(per_image_arr.max()),
            "images_with_no_label": int(np.sum(per_image_arr == 0)),
        },
        "train_val_split": {
            "train": {
                "images": split_counts["train"]["images"],
                "instances": split_counts["train"]["instances"],
                "per_class": {f"{k}_{class_names.get(k, '')}": v for k, v in split_counts["train"]["per_class"].items()},
            },
            "val": {
                "images": split_counts["val"]["images"],
                "instances": split_counts["val"]["instances"],
                "per_class": {f"{k}_{class_names.get(k, '')}": v for k, v in split_counts["val"]["per_class"].items()},
            },
        },
        "class_detail": per_class_report,
        "size_distribution": {
            "polygon_area": {
                "tiny(<0.005)":   round(float(np.mean(polygon_areas < 0.005)), 4),
                "small(0.005-0.01)": round(float(np.mean((polygon_areas >= 0.005) & (polygon_areas < 0.01))), 4),
                "medium(0.01-0.05)": round(float(np.mean((polygon_areas >= 0.01) & (polygon_areas < 0.05))), 4),
                "large(0.05-0.2)":   round(float(np.mean((polygon_areas >= 0.05) & (polygon_areas < 0.2))), 4),
                "xlarge(>=0.2)":     round(float(np.mean(polygon_areas >= 0.2)), 4),
                "percentiles": {
                    "p5":  round(float(np.percentile(polygon_areas, 5)), 6),
                    "p10": round(float(np.percentile(polygon_areas, 10)), 6),
                    "p25": round(float(np.percentile(polygon_areas, 25)), 6),
                    "p50": round(float(np.percentile(polygon_areas, 50)), 6),
                    "p75": round(float(np.percentile(polygon_areas, 75)), 6),
                    "p90": round(float(np.percentile(polygon_areas, 90)), 6),
                    "p95": round(float(np.percentile(polygon_areas, 95)), 6),
                },
            },
            "bbox_area_percentiles": {
                "p10": round(float(np.percentile(bbox_areas, 10)), 6),
                "p50": round(float(np.percentile(bbox_areas, 50)), 6),
                "p90": round(float(np.percentile(bbox_areas, 90)), 6),
            },
        },
        "aspect_ratio": {
            "mean":  round(float(aspect_ratios.mean()), 4),
            "std":   round(float(aspect_ratios.std()), 4),
            "min":   round(float(aspect_ratios.min()), 4),
            "max":   round(float(aspect_ratios.max()), 4),
            "narrow(<0.5)":  round(float(np.mean(aspect_ratios < 0.5)), 4),
            "square(0.5-2)": round(float(np.mean((aspect_ratios >= 0.5) & (aspect_ratios <= 2.0))), 4),
            "wide(>2)":      round(float(np.mean(aspect_ratios > 2.0)), 4),
        },
        "spatial_distribution": {
            "center_position": {
                "mean_x": round(float(center_xs.mean()), 4),
                "mean_y": round(float(center_ys.mean()), 4),
                "std_x":  round(float(center_xs.std()), 4),
                "std_y":  round(float(center_ys.std()), 4),
            },
            "grid_5x5_counts": spatial_grid.tolist(),
        },
        "polygon_complexity": {
            "mean_vertices": round(float(vertex_counts.mean()), 1),
            "std_vertices":  round(float(vertex_counts.std()), 1),
            "min_vertices":  int(vertex_counts.min()),
            "max_vertices":  int(vertex_counts.max()),
            "percentiles": {
                "p10": int(np.percentile(vertex_counts, 10)),
                "p50": int(np.percentile(vertex_counts, 50)),
                "p90": int(np.percentile(vertex_counts, 90)),
            },
        },
        "edge_touch": {
            "overall_ratio": round(float(np.mean(all_edge_flags)), 4),
            "by_side": {k: v for k, v in all_edge_sides.items()},
            "by_side_ratio": {k: round(v / total, 4) for k, v in all_edge_sides.items()},
        },
        "overlap_analysis": overlap_report,
        "image_density_distribution": density_distribution,
        "augmentation_hints": {},
    }

    # --- Augmentation推奨ヒント自動生成 ---
    hints = []

    # 小物体が多い場合
    tiny_ratio = float(np.mean(polygon_areas < 0.005))
    if tiny_ratio > 0.1:
        hints.append(f"tiny objects={tiny_ratio:.1%} → Mosaic/MixUp augmentationで小物体の学習機会を増やす")

    # エッジ接触が多い場合
    edge_ratio = float(np.mean(all_edge_flags))
    if edge_ratio > 0.3:
        hints.append(f"edge_touch={edge_ratio:.1%} → RandomCropは控えめに。Padding付きリサイズを推奨")

    # 重なりが多い場合
    if iou_values.size > 0:
        heavy_overlap = float(np.mean(iou_values >= 0.3))
        if heavy_overlap > 0.2:
            hints.append(f"heavy_overlap(IoU>=0.3)={heavy_overlap:.1%} → Copy-Paste augmentationは重なり増加に注意")

    # アスペクト比の偏り
    narrow = float(np.mean(aspect_ratios < 0.5))
    wide = float(np.mean(aspect_ratios > 2.0))
    if narrow > 0.15 or wide > 0.15:
        hints.append(f"aspect偏り(narrow={narrow:.1%}, wide={wide:.1%}) → RandomResizeの縦横比制限を推奨")

    # 空間偏り
    grid_flat = spatial_grid.flatten()
    grid_cv = float(grid_flat.std() / grid_flat.mean()) if grid_flat.mean() > 0 else 0
    if grid_cv > 0.5:
        hints.append(f"spatial_cv={grid_cv:.2f} → 空間偏りあり。RandomAffine/Translation augmentationが有効")

    # サイズ分散が大きい場合
    area_cv = float(polygon_areas.std() / polygon_areas.mean()) if polygon_areas.mean() > 0 else 0
    if area_cv > 1.5:
        hints.append(f"size_cv={area_cv:.2f} → サイズ分散大。マルチスケール学習(imgsz変動)を推奨")

    # ポリゴン頂点数が多い場合
    if float(vertex_counts.mean()) > 50:
        hints.append(f"mean_vertices={vertex_counts.mean():.0f} → ポリゴンが複雑。学習時のマスク解像度を上げることを検討")

    # クラス不均衡チェック
    counts_arr = np.array(list(per_class_count.values()))
    if counts_arr.max() / counts_arr.min() > 2.0:
        hints.append(f"class_imbalance_ratio={counts_arr.max()/counts_arr.min():.1f} → クラスごとのサンプリング重み調整を推奨")

    report["augmentation_hints"] = hints

    current_step += 1
    print_progress_inline()

    # --- 出力 ---
    output_path = os.path.join(output_dir, "analysis_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    current_step += 1
    print_progress_inline()
    print(f"\n\033[32m解析完了 → {output_path}\033[0m")
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    dataset_dir = select_directory("解析する統合済みデータセットを選択してください")
    if not dataset_dir:
        print("\033[31m入力フォルダーが選択されませんでした。\033[0m")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    current_step = 0
    analyze_dataset(dataset_dir, script_dir)
