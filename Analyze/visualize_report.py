# visualize_report.py
# analysis_report.json からグラフを生成するスクリプト

import os
import sys
import json
import tkinter as tk
from tkinter import filedialog
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# 日本語フォント設定
plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False

# 配色
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]
BG_COLOR = "#F8F8F8"
GRID_COLOR = "#E0E0E0"

def select_file(title: str):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title, filetypes=[("JSON", "*.json")])
    root.destroy()
    return path

def styled_ax(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_facecolor(BG_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.5)
    ax.tick_params(labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

def bar_value_labels(ax, bars, fmt="{:.0f}"):
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h, fmt.format(h),
                    ha="center", va="bottom", fontsize=7, fontweight="bold")

def pct_value_labels(ax, bars):
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.1%}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")

# ======================================================================
# Page 1: 概要 + クラス分布 + サイズ分布 + アスペクト比
# ======================================================================
def draw_page1(report, output_dir):
    fig = plt.figure(figsize=(16, 20), facecolor="white")
    fig.suptitle("(データセット解析レポート — 第1頁: 概要)\nDataset Analysis Report — Page 1: Overview",
                 fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(4, 2, hspace=0.45, wspace=0.30, left=0.07, right=0.93, top=0.93, bottom=0.04)

    summary = report["summary"]
    split = report["train_val_split"]
    class_detail = report["class_detail"]
    size_dist = report["size_distribution"]
    aspect = report["aspect_ratio"]

    # --- (0,0) Summary テキスト ---
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    lines = [
        f"(総画像数)      Total Images:  {summary['total_images']}",
        f"(総インスタンス数) Total Instances:  {summary['total_instances']}",
        f"(画像あたり平均/標準偏差) Mean / Std per Image:  {summary['mean_instances_per_image']} / {summary['std_instances_per_image']}",
        f"(画像あたり最大) Max per Image:  {summary['max_instances_per_image']}",
        f"(ラベル無し画像) No-label Images:  {summary['images_with_no_label']}",
        "",
        f"(学習用) Train:  {split['train']['images']} images / {split['train']['instances']} instances",
        f"(検証用) Val:    {split['val']['images']} images / {split['val']['instances']} instances",
    ]
    ax.text(0.03, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=9.5, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F0FE", edgecolor="#4C72B0"))
    ax.set_title("(概要)\nSummary", fontsize=12, fontweight="bold", pad=8)

    # --- (0,1) クラス分布 ---
    ax = fig.add_subplot(gs[0, 1])
    cls_names = list(class_detail.keys())
    cls_counts = [class_detail[c]["count"] for c in cls_names]
    bars = ax.bar(cls_names, cls_counts, color=COLORS[:len(cls_names)], edgecolor="white", width=0.5)
    bar_value_labels(ax, bars)
    styled_ax(ax, "(クラス分布)\nClass Distribution", ylabel="(件数) Count")

    # --- (1,0) Train/Val クラス別比較 ---
    ax = fig.add_subplot(gs[1, 0])
    train_cls = split["train"]["per_class"]
    val_cls = split["val"]["per_class"]
    all_keys = sorted(set(list(train_cls.keys()) + list(val_cls.keys())))
    x = np.arange(len(all_keys))
    w = 0.35
    bars1 = ax.bar(x - w/2, [train_cls.get(k, 0) for k in all_keys], w, label="(学習) Train", color=COLORS[0])
    bars2 = ax.bar(x + w/2, [val_cls.get(k, 0) for k in all_keys], w, label="(検証) Val", color=COLORS[1])
    ax.set_xticks(x)
    ax.set_xticklabels(all_keys, rotation=15, ha="right")
    ax.legend(fontsize=8)
    bar_value_labels(ax, bars1)
    bar_value_labels(ax, bars2)
    styled_ax(ax, "(学習/検証 クラス別分割)\nTrain / Val Split by Class", ylabel="(件数) Count")

    # --- (1,1) ポリゴン面積サイズ分布 ---
    ax = fig.add_subplot(gs[1, 1])
    pa = size_dist["polygon_area"]
    size_labels = ["(極小) tiny\n<0.5%", "(小) small\n0.5-1%", "(中) medium\n1-5%", "(大) large\n5-20%", "(特大) xlarge\n>=20%"]
    size_keys = ["tiny(<0.005)", "small(0.005-0.01)", "medium(0.01-0.05)", "large(0.05-0.2)", "xlarge(>=0.2)"]
    size_vals = [pa[k] for k in size_keys]
    bars = ax.bar(size_labels, size_vals, color=["#C44E52", "#DD8452", "#CCB974", "#55A868", "#4C72B0"], edgecolor="white")
    pct_value_labels(ax, bars)
    styled_ax(ax, "(ポリゴン面積サイズ分布)\nPolygon Area Size Distribution", ylabel="(割合) Ratio")
    ax.set_ylim(0, max(size_vals) * 1.2 if max(size_vals) > 0 else 1)

    # --- (2,0) ポリゴン面積パーセンタイル ---
    ax = fig.add_subplot(gs[2, 0])
    pcts = pa["percentiles"]
    pct_labels = list(pcts.keys())
    pct_vals = list(pcts.values())
    ax.plot(pct_labels, pct_vals, "o-", color=COLORS[0], linewidth=2, markersize=6)
    for i, v in enumerate(pct_vals):
        ax.annotate(f"{v:.4f}", (pct_labels[i], v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7)
    styled_ax(ax, "(ポリゴン面積パーセンタイル)\nPolygon Area Percentiles",
              xlabel="(パーセンタイル) Percentile", ylabel="(面積・正規化) Area (normalized)")

    # --- (2,1) アスペクト比分布 ---
    ax = fig.add_subplot(gs[2, 1])
    ar_labels = ["(縦長) narrow\n(<0.5)", "(正方形) square\n(0.5-2.0)", "(横長) wide\n(>2.0)"]
    ar_keys = ["narrow(<0.5)", "square(0.5-2)", "wide(>2)"]
    ar_vals = [aspect[k] for k in ar_keys]
    bars = ax.bar(ar_labels, ar_vals, color=[COLORS[1], COLORS[2], COLORS[4]], edgecolor="white", width=0.5)
    pct_value_labels(ax, bars)
    styled_ax(ax, "(アスペクト比分布)\nAspect Ratio Distribution", ylabel="(割合) Ratio")
    ax.set_ylim(0, max(ar_vals) * 1.2 if max(ar_vals) > 0 else 1)

    # --- (3,0) アスペクト比統計テキスト ---
    ax = fig.add_subplot(gs[3, 0])
    ax.axis("off")
    ar_lines = [
        f"(平均)   Mean:  {aspect['mean']:.4f}",
        f"(標準偏差) Std:   {aspect['std']:.4f}",
        f"(最小)   Min:   {aspect['min']:.4f}",
        f"(最大)   Max:   {aspect['max']:.4f}",
    ]
    ax.text(0.05, 0.95, "\n".join(ar_lines), transform=ax.transAxes,
            fontsize=11, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF3E0", edgecolor="#DD8452"))
    ax.set_title("(アスペクト比統計)\nAspect Ratio Stats", fontsize=12, fontweight="bold", pad=8)

    # --- (3,1) BBox面積パーセンタイル ---
    ax = fig.add_subplot(gs[3, 1])
    bbox_p = size_dist["bbox_area_percentiles"]
    bp_labels = list(bbox_p.keys())
    bp_vals = list(bbox_p.values())
    bars = ax.bar(bp_labels, bp_vals, color=COLORS[2], edgecolor="white", width=0.4)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.4f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    styled_ax(ax, "(BBox面積パーセンタイル)\nBBox Area Percentiles", ylabel="(面積・正規化) Area (normalized)")

    fig.savefig(os.path.join(output_dir, "report_page1_overview.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\033[32m  Page 1 saved\033[0m")

# ======================================================================
# Page 2: 空間分布 + エッジ + 重なり + 密度 + ポリゴン複雑度
# ======================================================================
def draw_page2(report, output_dir):
    fig = plt.figure(figsize=(16, 20), facecolor="white")
    fig.suptitle("(データセット解析レポート — 第2頁: 空間・重なり)\nDataset Analysis Report — Page 2: Spatial & Overlap",
                 fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(4, 2, hspace=0.45, wspace=0.30, left=0.07, right=0.93, top=0.93, bottom=0.04)

    spatial = report["spatial_distribution"]
    edge = report["edge_touch"]
    overlap = report["overlap_analysis"]
    density = report["image_density_distribution"]
    poly_comp = report["polygon_complexity"]

    # --- (0,0) 空間ヒートマップ ---
    ax = fig.add_subplot(gs[0, 0])
    grid = np.array(spatial["grid_5x5_counts"])
    im = ax.imshow(grid, cmap="YlOrRd", interpolation="nearest", aspect="equal")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, str(grid[i][j]), ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white" if grid[i][j] > grid.max() * 0.6 else "black")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([f"{i*20}-{(i+1)*20}%" for i in range(5)], fontsize=7)
    ax.set_yticklabels([f"{i*20}-{(i+1)*20}%" for i in range(5)], fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("(空間ヒートマップ 5x5グリッド)\nSpatial Heatmap (5x5 grid)", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("(X位置) X position", fontsize=9)
    ax.set_ylabel("(Y位置) Y position", fontsize=9)

    # --- (0,1) 空間中心統計 ---
    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    cp = spatial["center_position"]
    sp_lines = [
        "(オブジェクト中心位置)",
        "Center Position of Objects",
        "",
        f"(X平均) Mean X:  {cp['mean_x']:.4f}  (0.5 = (中央) center)",
        f"(Y平均) Mean Y:  {cp['mean_y']:.4f}  (0.5 = (中央) center)",
        f"(X標準偏差) Std X:   {cp['std_x']:.4f}",
        f"(Y標準偏差) Std Y:   {cp['std_y']:.4f}",
        "",
        f"(X中心からのずれ) X offset from center: {abs(cp['mean_x'] - 0.5):.4f}",
        f"(Y中心からのずれ) Y offset from center: {abs(cp['mean_y'] - 0.5):.4f}",
    ]
    ax.text(0.03, 0.95, "\n".join(sp_lines), transform=ax.transAxes,
            fontsize=9.5, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F5E9", edgecolor="#55A868"))
    ax.set_title("(空間中心統計)\nSpatial Center Stats", fontsize=12, fontweight="bold", pad=8)

    # --- (1,0) エッジ接触 (辺別) ---
    ax = fig.add_subplot(gs[1, 0])
    sides = list(edge["by_side"].keys())
    side_jp = {"top": "(上) top", "bottom": "(下) bottom", "left": "(左) left", "right": "(右) right"}
    side_labels = [side_jp.get(s, s) for s in sides]
    side_counts = [edge["by_side"][s] for s in sides]
    side_ratios = [edge["by_side_ratio"][s] for s in sides]
    bar_colors = ["#C44E52", "#4C72B0", "#DD8452", "#55A868"]
    bars = ax.bar(side_labels, side_counts, color=bar_colors, edgecolor="white", width=0.5)
    for bar, r in zip(bars, side_ratios):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h}\n({r:.1%})",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    styled_ax(ax, f"(辺別エッジ接触 全体: {edge['overall_ratio']:.1%})\nEdge Touch by Side (overall: {edge['overall_ratio']:.1%})",
              ylabel="(件数) Count")

    # --- (1,1) 重なり解析 ---
    ax = fig.add_subplot(gs[1, 1])
    if overlap:
        pair_keys = list(overlap.keys())
        x = np.arange(len(pair_keys))
        w = 0.25
        iou_thresholds = ["ratio_iou>=0.1", "ratio_iou>=0.3", "ratio_iou>=0.5"]
        iou_labels = ["IoU>=0.1", "IoU>=0.3", "IoU>=0.5"]
        iou_colors = [COLORS[0], COLORS[1], COLORS[3]]
        for idx, (key, label, color) in enumerate(zip(iou_thresholds, iou_labels, iou_colors)):
            vals = [overlap[pk].get(key, 0) for pk in pair_keys]
            bars = ax.bar(x + idx * w, vals, w, label=label, color=color)
            pct_value_labels(ax, bars)
        ax.set_xticks(x + w)
        pair_display = [f"cls {pk}" for pk in pair_keys]
        ax.set_xticklabels(pair_display, fontsize=8)
        ax.legend(fontsize=8)
        styled_ax(ax, "(重なり解析 クラスペア別)\nOverlap Analysis (by class pair)", ylabel="(割合) Ratio")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "(重なりデータなし) No overlap data", ha="center", va="center", fontsize=12, transform=ax.transAxes)

    # --- (2,0) 画像密度分布 ---
    ax = fig.add_subplot(gs[2, 0])
    d_labels = list(density.keys())
    d_vals = list(density.values())
    d_display = [l.replace("_instances", "").replace("_", " ") for l in d_labels]
    bars = ax.bar(d_display, d_vals, color=COLORS[2], edgecolor="white")
    bar_value_labels(ax, bars)
    styled_ax(ax, "(画像密度分布 画像あたりインスタンス数)\nImage Density Distribution (instances/image)",
              xlabel="(画像あたりインスタンス数) Instances per image",
              ylabel="(画像数) Image count")

    # --- (2,1) ポリゴン複雑度 ---
    ax = fig.add_subplot(gs[2, 1])
    ax.axis("off")
    pc_lines = [
        "(ポリゴン複雑度 — 頂点数)",
        "Polygon Complexity (vertex count)",
        "",
        f"(平均)   Mean:  {poly_comp['mean_vertices']}",
        f"(標準偏差) Std:   {poly_comp['std_vertices']}",
        f"(最小)   Min:   {poly_comp['min_vertices']}",
        f"(最大)   Max:   {poly_comp['max_vertices']}",
        "",
        "(パーセンタイル) Percentiles:",
        f"  p10:  {poly_comp['percentiles']['p10']}",
        f"  p50:  {poly_comp['percentiles']['p50']}",
        f"  p90:  {poly_comp['percentiles']['p90']}",
    ]
    ax.text(0.05, 0.95, "\n".join(pc_lines), transform=ax.transAxes,
            fontsize=10, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F3E5F5", edgecolor="#8172B3"))
    ax.set_title("(ポリゴン複雑度)\nPolygon Complexity", fontsize=12, fontweight="bold", pad=8)

    # --- (3,0-1) Augmentation Hints ---
    ax = fig.add_subplot(gs[3, :])
    ax.axis("off")
    hints = report.get("augmentation_hints", [])
    if hints:
        hint_text = "(オーグメンテーション推奨事項)\nAugmentation Hints\n\n" + "\n".join([f"  {i+1}. {h}" for i, h in enumerate(hints)])
    else:
        hint_text = "(オーグメンテーション推奨事項)\nAugmentation Hints\n\n  (特に警告なし。データセットは均衡です)\n  No specific warnings detected. Dataset looks balanced."
    ax.text(0.02, 0.95, hint_text, transform=ax.transAxes,
            fontsize=11, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#FFF8E1", edgecolor="#F9A825", linewidth=2))
    ax.set_title("(オーグメンテーション推奨)\nAugmentation Recommendations", fontsize=12, fontweight="bold", pad=8)

    fig.savefig(os.path.join(output_dir, "report_page2_spatial.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\033[32m  Page 2 saved\033[0m")

# ======================================================================
# Page 3: クラス別詳細比較
# ======================================================================
def draw_page3(report, output_dir):
    class_detail = report["class_detail"]
    cls_names = list(class_detail.keys())
    n_cls = len(cls_names)

    fig = plt.figure(figsize=(16, 20), facecolor="white")
    fig.suptitle("(データセット解析レポート — 第3頁: クラス別詳細)\nDataset Analysis Report — Page 3: Per-Class Detail",
                 fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(4, 2, hspace=0.45, wspace=0.30, left=0.07, right=0.93, top=0.93, bottom=0.04)

    # --- (0,0) クラス別ポリゴン面積 ---
    ax = fig.add_subplot(gs[0, 0])
    means = [class_detail[c]["polygon_area"]["mean"] for c in cls_names]
    p10s = [class_detail[c]["polygon_area"]["p10"] for c in cls_names]
    p90s = [class_detail[c]["polygon_area"]["p90"] for c in cls_names]
    x = np.arange(n_cls)
    bars = ax.bar(x, means, color=COLORS[:n_cls], edgecolor="white", width=0.5)
    ax.errorbar(x, means, yerr=[np.array(means) - np.array(p10s), np.array(p90s) - np.array(means)],
                fmt="none", ecolor="black", capsize=5, linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cls_names, fontsize=8)
    styled_ax(ax, "(クラス別ポリゴン面積 平均, p10-p90)\nPolygon Area by Class (mean, p10-p90)", ylabel="(面積) Area")

    # --- (0,1) クラス別BBox面積 ---
    ax = fig.add_subplot(gs[0, 1])
    bmeans = [class_detail[c]["bbox_area"]["mean"] for c in cls_names]
    bp10s = [class_detail[c]["bbox_area"]["p10"] for c in cls_names]
    bp90s = [class_detail[c]["bbox_area"]["p90"] for c in cls_names]
    bars = ax.bar(x, bmeans, color=COLORS[:n_cls], edgecolor="white", width=0.5)
    ax.errorbar(x, bmeans, yerr=[np.array(bmeans) - np.array(bp10s), np.array(bp90s) - np.array(bmeans)],
                fmt="none", ecolor="black", capsize=5, linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cls_names, fontsize=8)
    styled_ax(ax, "(クラス別BBox面積 平均, p10-p90)\nBBox Area by Class (mean, p10-p90)", ylabel="(面積) Area")

    # --- (1,0) クラス別アスペクト比 ---
    ax = fig.add_subplot(gs[1, 0])
    ar_means = [class_detail[c]["aspect_ratio"]["mean"] for c in cls_names]
    ar_stds = [class_detail[c]["aspect_ratio"]["std"] for c in cls_names]
    bars = ax.bar(x, ar_means, color=COLORS[:n_cls], edgecolor="white", width=0.5)
    ax.errorbar(x, ar_means, yerr=ar_stds, fmt="none", ecolor="black", capsize=5, linewidth=1.5)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, label="(正方形) square (1.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(cls_names, fontsize=8)
    ax.legend(fontsize=8)
    styled_ax(ax, "(クラス別アスペクト比 平均 +/- 標準偏差)\nAspect Ratio by Class (mean +/- std)", ylabel="(アスペクト比) Aspect Ratio")

    # --- (1,1) クラス別エッジ接触率 ---
    ax = fig.add_subplot(gs[1, 1])
    edge_ratios = [class_detail[c]["edge_touch_ratio"] for c in cls_names]
    bars = ax.bar(x, edge_ratios, color=COLORS[:n_cls], edgecolor="white", width=0.5)
    pct_value_labels(ax, bars)
    ax.set_xticks(x)
    ax.set_xticklabels(cls_names, fontsize=8)
    styled_ax(ax, "(クラス別エッジ接触率)\nEdge Touch Ratio by Class", ylabel="(割合) Ratio")
    ax.set_ylim(0, max(edge_ratios) * 1.3 if max(edge_ratios) > 0 else 1)

    # --- (2,0) クラス別頂点数 ---
    ax = fig.add_subplot(gs[2, 0])
    v_means = [class_detail[c]["vertex_count"]["mean"] for c in cls_names]
    v_mins = [class_detail[c]["vertex_count"]["min"] for c in cls_names]
    v_maxs = [class_detail[c]["vertex_count"]["max"] for c in cls_names]
    bars = ax.bar(x, v_means, color=COLORS[:n_cls], edgecolor="white", width=0.5)
    ax.errorbar(x, v_means, yerr=[np.array(v_means) - np.array(v_mins), np.array(v_maxs) - np.array(v_means)],
                fmt="none", ecolor="black", capsize=5, linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cls_names, fontsize=8)
    bar_value_labels(ax, bars, fmt="{:.1f}")
    styled_ax(ax, "(クラス別頂点数 平均, 最小-最大)\nVertex Count by Class (mean, min-max)", ylabel="(頂点数) Vertices")

    # --- (2,1) クラス別中心位置 ---
    ax = fig.add_subplot(gs[2, 1])
    for i, c in enumerate(cls_names):
        cp = class_detail[c]["center_position"]
        ax.scatter(cp["mean_x"], cp["mean_y"], s=200, color=COLORS[i], edgecolors="black",
                   linewidth=1.5, zorder=5, label=c)
        ax.add_patch(plt.Circle((cp["mean_x"], cp["mean_y"]),
                                max(cp["std_x"], cp["std_y"]),
                                fill=False, color=COLORS[i], linestyle="--", linewidth=1.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.legend(fontsize=8)
    styled_ax(ax, "(クラス別中心位置 平均 + 標準偏差円)\nCenter Position by Class (mean + std circle)", xlabel="X", ylabel="Y")

    # --- (3,0-1) クラス別詳細テーブル ---
    ax = fig.add_subplot(gs[3, :])
    ax.axis("off")
    headers = [
        "(クラス)\nClass",
        "(件数)\nCount",
        "(割合)\nRatio",
        "(ポリゴン面積)\nPolyArea\nmean",
        "(BBox面積)\nBBoxArea\nmean",
        "(アスペクト比)\nAspect\nmean",
        "(頂点数)\nVertices\nmean",
        "(エッジ接触)\nEdge\ntouch",
    ]
    cell_data = []
    for c in cls_names:
        d = class_detail[c]
        cell_data.append([
            c,
            str(d["count"]),
            f"{d['ratio']:.1%}",
            f"{d['polygon_area']['mean']:.4f}",
            f"{d['bbox_area']['mean']:.4f}",
            f"{d['aspect_ratio']['mean']:.2f}",
            f"{d['vertex_count']['mean']:.0f}",
            f"{d['edge_touch_ratio']:.1%}",
        ])
    table = ax.table(cellText=cell_data, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.2)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4C72B0")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#F0F4FA" if row % 2 == 0 else "white")
        cell.set_edgecolor(GRID_COLOR)
    ax.set_title("(クラス別サマリーテーブル)\nPer-Class Summary Table", fontsize=12, fontweight="bold", pad=12)

    fig.savefig(os.path.join(output_dir, "report_page3_class_detail.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\033[32m  Page 3 saved\033[0m")

# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    json_path = select_file("analysis_report.json を選択してください")
    if not json_path:
        print("\033[31mファイルが選択されませんでした。\033[0m")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("\033[32mグラフ生成中...\033[0m")
    draw_page1(report, script_dir)
    draw_page2(report, script_dir)
    draw_page3(report, script_dir)
    print(f"\033[32m完了 → {script_dir}\033[0m")
    print("  report_page1_overview.png")
    print("  report_page2_spatial.png")
    print("  report_page3_class_detail.png")
