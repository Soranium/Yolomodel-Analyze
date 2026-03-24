# labels_converter_CVAT_integrated.py — サブフォルダーを統合する拡張版
import os
import shutil
import random
import sys
import time
import tkinter as tk
import yaml
from tkinter import filedialog

# 進捗バー（4ステップ：収集→複製（統合）→分割→完了）
TOTAL_STEPS = 4
current_step = 0

def print_progress_inline():
    filled = int(25 * current_step / TOTAL_STEPS)
    bar = "█" * filled + "→" + "-" * (25 - filled)
    percent = int(current_step / TOTAL_STEPS * 100)
    sys.stdout.write(f"\r[{bar}] {percent}%")
    sys.stdout.flush()

def load_classes_from_data_yaml(source_dir: str):
    """
    source_dir にある data.yaml からクラス名リストを返す
    """
    data_yaml_path = os.path.join(source_dir, "data.yaml")
    if not os.path.isfile(data_yaml_path):
        print(f"\033[33m⚠ data.yaml が見つかりません: {data_yaml_path}\033[0m")
        return []
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    class_list = data.get("names", [])
    if not class_list:
        print("\033[33m⚠ data.yaml に 'names' が見つかりません。\033[0m")
    return class_list

def merge_class_lists(subdirs: list):
    """
    すべてのサブフォルダーの data.yaml からクラスをマージ（重複除去）
    """
    all_classes = set()
    for subdir in subdirs:
        classes = load_classes_from_data_yaml(subdir)
        all_classes.update(classes)
    merged_list = sorted(list(all_classes))
    if not merged_list:
        print("\033[33m⚠ すべての data.yaml からクラスが見つかりませんでした。\033[0m")
    return merged_list

def generate_yaml(output_dir: str, class_list: list):
    """
    指定の出力ディレクトリに YOLOv8 セグメンテーション用 data.yaml を生成する。
    """
    yaml_path = os.path.join(output_dir, "data.yaml")
    data = {
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_list),
        "names": class_list,
        "task": "segment"  # ← これを入れるだけで世界が平和になる
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)
    print(f"\033[32m✅ data.yaml（Seg対応）を生成しました: {yaml_path}\033[0m")

def integrate_and_split_yolo_dataset(
    subdirs: list,
    output_dir: str,
    split_ratio: float = 0.75  # trainに残す割合（=0.75 → valに25%移動）
):
    global current_step
    print_progress_inline()
    time.sleep(0.2)

    # --- 収集フェーズ: すべてのペアをリストアップ ---
    paired = []
    valid_exts = (".jpg", ".jpeg", ".png")
    for source_dir in subdirs:
        img_train = os.path.join(source_dir, "images", "train")
        lbl_train = os.path.join(source_dir, "labels", "train")
        if not os.path.isdir(img_train) or not os.path.isdir(lbl_train):
            print(f"\033[33m⚠ サブフォルダーに images/train または labels/train が見つかりません: {source_dir}\033[0m")
            continue

        image_files = [f for f in os.listdir(img_train) if f.lower().endswith(valid_exts)]
        for img in image_files:
            base, ext = os.path.splitext(img)
            lbl_path = os.path.join(lbl_train, base + ".txt")
            if os.path.exists(lbl_path):
                paired.append((os.path.join(img_train, img), lbl_path, base, ext))
            else:
                print(f"\033[33m⚠ ラベルなし画像: {img}（スキップ） in {source_dir}\033[0m")

    if not paired:
        print("\033[31mラベル付き画像が見つかりませんでした。\033[0m")
        return

    current_step += 1
    print_progress_inline()
    print("\n\033[32m収集完了: 合計 {len(paired)} ペア\033[0m")

    # --- 統合複製フェーズ: 出力にコピー（重複リネーム） ---
    images_root = os.path.join(output_dir, "images")
    labels_root = os.path.join(output_dir, "labels")
    os.makedirs(images_root, exist_ok=True)
    os.makedirs(labels_root, exist_ok=True)

    name_counter = {}  # base名ごとのカウンター
    renamed_paired = []  # 新しいbase, extのリスト

    for img_src, lbl_src, base, ext in paired:
        new_base = base
        if base in name_counter:
            name_counter[base] += 1
            new_base = f"{base}_{name_counter[base]}"
        else:
            name_counter[base] = 0

        img_dest = os.path.join(images_root, new_base + ext)
        lbl_dest = os.path.join(labels_root, new_base + ".txt")

        shutil.copy2(img_src, img_dest)
        shutil.copy2(lbl_src, lbl_dest)

        renamed_paired.append((new_base, ext))

    current_step += 1
    print_progress_inline()
    print("\n\033[32m統合複製完了（重複リネーム済）\033[0m")

    # --- 分割処理 ---
    random.shuffle(renamed_paired)
    split_idx = int(len(renamed_paired) * split_ratio)
    train_set = renamed_paired[:split_idx]
    val_set = renamed_paired[split_idx:]

    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    def move_pair(file_list, subset):
        for base, ext in file_list:
            shutil.move(os.path.join(images_root, base + ext), os.path.join(output_dir, f"images/{subset}", base + ext))
            shutil.move(os.path.join(labels_root, base + ".txt"), os.path.join(output_dir, f"labels/{subset}", base + ".txt"))

    move_pair(train_set, "train")
    move_pair(val_set, "val")

    # 空の images と labels を削除
    if not os.listdir(images_root):
        shutil.rmtree(images_root)
    if not os.listdir(labels_root):
        shutil.rmtree(labels_root)

    print(f"\n\033[32mTrain {len(train_set)}件 / Val {len(val_set)}件 に分割完了\033[0m")

    current_step += 1
    print_progress_inline()
    print("\n\033[32m分割完了\033[0m")

    # --- YAML生成 ---
    class_list = merge_class_lists(subdirs)
    generate_yaml(output_dir, class_list)

    current_step += 1
    print_progress_inline()
    print("\n\033[32m処理完了\033[0m")

def select_directory(title: str):
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title=title)
    root.destroy()
    return directory

if __name__ == "__main__":
    base_source_dir = select_directory("入力フォルダー（Import）を選択してください")
    if not base_source_dir:
        print("\033[31m入力フォルダーが選択されませんでした。終了します。\033[0m")
        sys.exit(1)

    output_base_dir = select_directory("出力フォルダー（Export）を選択してください")
    if not output_base_dir:
        print("\033[31m出力フォルダーが選択されませんでした。終了します。\033[0m")
        sys.exit(1)

    subdirs = [os.path.join(base_source_dir, name) for name in os.listdir(base_source_dir)
               if os.path.isdir(os.path.join(base_source_dir, name))]

    if not subdirs:
        print("\033[31m入力フォルダーにサブフォルダが見つかりませんでした。終了します。\033[0m")
        sys.exit(1)

    original_name = os.path.basename(base_source_dir.rstrip("\\/"))
    output_dir = os.path.join(output_base_dir, f"{original_name}_integrated_done")
    os.makedirs(output_dir, exist_ok=True)

    current_step = 0
    print(f"\033[32m統合処理開始: {base_source_dir}\033[0m")
    integrate_and_split_yolo_dataset(subdirs, output_dir, split_ratio=0.85)