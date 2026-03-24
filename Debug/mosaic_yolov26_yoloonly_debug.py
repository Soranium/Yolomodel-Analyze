"""
YOLO-only セグメンテーション（デバッグ可視化版）

全フレームでYOLO推論を実行し、セグメンテーションマスクを即座にオーバーレイ。
SAM2を使用しない比較用スクリプト。1パスで処理。

メモリ使用量: ~4GB（YOLOモデル + 1フレーム分）
"""

import os
import cv2
import av
import shutil
import tempfile
import time
import logging
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tkinter import Tk, filedialog
from ultralytics import YOLO
from rich.progress import Progress, BarColumn, TextColumn, ProgressColumn
from rich.text import Text

# =========================
# 設定
# =========================
CONF_THRESHOLD = 0.15
MASK_ALPHA     = 0.45
MASK_THRESH    = 0.5
YOLO_CONF      = 0.1

# YOLOモデル: カレントディレクトリから.ptを自動検出
def find_yolo_model():
    script_dir = Path(__file__).parent
    pt_files = list(script_dir.glob("*.pt"))
    if pt_files:
        return str(pt_files[0])
    return None

MODEL_PATH = find_yolo_model()

MAX_ANCHOR_OBJECTS = 5

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# =========================
# メモリ監視ユーティリティ
# =========================
def get_memory_info() -> dict:
    """現在のRAM/VRAMメモリ使用量を取得"""
    info = {}
    try:
        import psutil
        proc = psutil.Process()
        mem = proc.memory_info()
        info["ram_rss_gb"] = mem.rss / (1024**3)
        sys_mem = psutil.virtual_memory()
        info["ram_total_gb"] = sys_mem.total / (1024**3)
        info["ram_available_gb"] = sys_mem.available / (1024**3)
        info["ram_percent"] = sys_mem.percent
    except ImportError:
        pass
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        info["vram_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
        info["vram_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
        info["vram_free_gb"] = free_b / (1024**3)
        info["vram_total_gb"] = total_b / (1024**3)
    return info


def log_memory(tag: str = ""):
    """メモリ使用量をログ出力"""
    m = get_memory_info()
    parts = []
    if "ram_rss_gb" in m:
        parts.append(
            f"RAM: proc={m['ram_rss_gb']:.2f}GB, "
            f"avail={m['ram_available_gb']:.1f}/{m['ram_total_gb']:.1f}GB ({m['ram_percent']:.0f}%)"
        )
    if "vram_allocated_gb" in m:
        parts.append(
            f"VRAM: alloc={m['vram_allocated_gb']:.2f}GB, "
            f"free={m['vram_free_gb']:.1f}/{m['vram_total_gb']:.1f}GB"
        )
    logger.info(f"[MEM:{tag}] {' | '.join(parts)}")


# =========================
# UI / ユーティリティ
# =========================
def choose_folder(title="フォルダを選択してください"):
    root = Tk(); root.withdraw()
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder


class FrameProgressColumn(ProgressColumn):
    def render(self, task):
        c, t = int(task.completed), int(task.total)
        p = c / t * 100 if t else 0
        return Text(f"{p:5.1f}%")


def get_gpu_info():
    if not torch.cuda.is_available():
        return {"available": False, "count": 0, "gpus": []}

    count = torch.cuda.device_count()
    gpus = []
    for i in range(count):
        torch.cuda.set_device(i)
        free_b, total_b = torch.cuda.mem_get_info()
        props = torch.cuda.get_device_properties(i)
        gpus.append({
            "id": i,
            "name": props.name,
            "total_gb": total_b / (1024**3),
            "free_gb":  free_b / (1024**3),
        })
    return {"available": True, "count": count, "gpus": gpus}


def pick_encoder():
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        c = av.open(tmp.name, mode="w")
        s = c.add_stream("h264_nvenc", rate=30)
        s.width, s.height = 64, 64
        s.pix_fmt = "yuv420p"
        c.close()
        os.remove(tmp.name)
        return "h264_nvenc"
    except Exception:
        try:
            if os.path.exists(tmp.name):
                os.remove(tmp.name)
        except Exception:
            pass
    return "libx264"


# =========================
# YOLO ヘルパー
# =========================
def extract_yolo_masks(result, H, W):
    """YOLOの結果からマスクを抽出 → list of (obj_id, mask_tensor)"""
    if result.boxes is None or result.masks is None:
        return []

    confs = result.boxes.conf
    if confs is None or len(confs) == 0:
        return []

    keep = confs >= CONF_THRESHOLD
    if keep.sum().item() == 0:
        return []

    masks = result.masks.data[keep]  # (K, h, w)

    obj_masks = []
    for i, mask in enumerate(masks):
        mask_f = mask[None, None].float()
        up = F.interpolate(mask_f, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        mask_bool = (up > MASK_THRESH)
        obj_masks.append((i, mask_bool))

    return obj_masks


def combine_masks_gpu(masks_list, H, W, device):
    """複数マスクを1つに合成"""
    if not masks_list:
        return None

    combined = torch.zeros((H, W), dtype=torch.bool, device=device)
    for _, mask in masks_list:
        combined = combined | mask
    return combined


def overlay_green_mask_gpu(img_bgr_u8: np.ndarray, mask_hw_bool_gpu, alpha: float, device: str):
    if mask_hw_bool_gpu is None:
        return img_bgr_u8

    img = torch.from_numpy(img_bgr_u8).to(device=device, non_blocking=True).float()
    green = torch.zeros_like(img)
    green[..., 1] = 255.0

    a = float(alpha)
    m = mask_hw_bool_gpu.unsqueeze(-1)

    blended = img * (1.0 - a) + green * a
    out = torch.where(m, blended, img).clamp(0, 255).byte()

    return out.cpu().numpy()


# =========================
# デバッグ用関数（映像オーバーレイ）
# =========================
def draw_debug_overlay(img: np.ndarray, frame_idx: int, obj_count: int = 0) -> np.ndarray:
    """デバッグ情報をフレームに描画（左上コーナー）"""
    out = img.copy()

    overlay = out.copy()
    cv2.rectangle(overlay, (10, 10), (250, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

    color = (0, 0, 255)  # 赤 (BGR)
    mode_text = "YOLO Only"

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, mode_text, (20, 40), font, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(out, f"Frame: {frame_idx}", (20, 65), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"Objects: {obj_count}", (20, 90), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def draw_text_on_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """マスク領域の中心に大きな文字を描画"""
    out = img.copy()

    if mask is None:
        return out

    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy().astype(np.uint8)
    else:
        mask_np = mask.astype(np.uint8)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)

    color = (0, 0, 255)
    text = "YOLO"

    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(1, num_labels):
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        area = stats[i, cv2.CC_STAT_AREA]

        if area < 500:
            continue

        font_scale = min(2.5, max(0.8, area / 20000))
        thickness = max(2, int(font_scale * 2))

        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = cx - text_w // 2
        text_y = cy + text_h // 2

        cv2.putText(out, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(out, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

    return out


# =========================
# メインクラス
# =========================
class YoloOnlyTracker:
    """
    YOLO-only セグメンテーショントラッカー（比較用）

    全フレームでYOLO推論を実行し、即座にオーバーレイ＆エンコード。
    1パスで処理。メモリ使用量: ~4GB（YOLOモデル + 1フレーム分）
    """

    def __init__(self, device="cuda"):
        self.device = device

        # YOLOモデル
        if MODEL_PATH is None:
            raise FileNotFoundError("YOLOモデル(.pt)が見つかりません。スクリプトと同じフォルダに配置してください。")
        self.yolo = YOLO(MODEL_PATH)
        logger.info(f"YOLO loaded: {MODEL_PATH}")

    # =========================================================
    # process_video: 1パス処理
    # =========================================================
    def process_video(self, input_path, output_path, encoder):
        """
        動画を処理（全フレームYOLO検出、1パス）

        Yields:
            (completed, total, phase_name): 進捗
        """
        tmp = None
        container_out = None
        t_video_start = time.time()

        try:
            cv2.setNumThreads(0)

            # 動画メタデータ取得
            container = av.open(input_path)
            v_in = container.streams.video[0]
            W = v_in.codec_context.width
            H = v_in.codec_context.height
            fps_frac = v_in.average_rate
            fps = float(fps_frac)
            bit_rate = v_in.bit_rate
            estimated_frames = v_in.frames or 0
            if estimated_frames <= 0:
                if container.duration:
                    duration_s = container.duration / 1_000_000
                    estimated_frames = max(int(duration_s * fps), 100)
                else:
                    estimated_frames = 1000
            container.close()

            logger.info("=" * 70)
            logger.info(f"VIDEO: {Path(input_path).name}")
            logger.info(f"  Resolution: {W}x{H} @ {fps:.2f}fps (est. {estimated_frames} frames)")
            logger.info(f"  Pipeline: YOLO-only (every frame)")
            logger.info(f"  Settings: CONF_THRESHOLD={CONF_THRESHOLD}, YOLO_CONF={YOLO_CONF}")
            log_memory("video_start")
            logger.info("=" * 70)

            grand_total = estimated_frames

            # 出力準備
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp.close()

            container_out = av.open(tmp.name, mode="w")
            v_out = container_out.add_stream(encoder, rate=fps_frac)
            v_out.width = W
            v_out.height = H
            v_out.pix_fmt = "yuv420p"
            if bit_rate:
                v_out.bit_rate = bit_rate
            v_out.options = {"preset": "medium", "bf": "2"}

            # 1パス処理
            container = av.open(input_path)
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"

            frame_idx = 0
            detect_count = 0
            t0 = time.time()

            logger.info("--- YOLO-only Processing ---")

            for frame in container.decode(stream):
                img = frame.to_ndarray(format="bgr24")

                with torch.no_grad():
                    results = self.yolo.predict(
                        img, verbose=False, device=self.device, conf=YOLO_CONF
                    )

                obj_masks = extract_yolo_masks(results[0], H, W)
                combined = combine_masks_gpu(
                    obj_masks[:MAX_ANCHOR_OBJECTS], H, W, self.device
                )

                if combined is not None:
                    detect_count += 1

                out_img = overlay_green_mask_gpu(img, combined, MASK_ALPHA, self.device)

                if combined is not None:
                    out_img = draw_text_on_mask(out_img, combined)

                out_img = draw_debug_overlay(out_img, frame_idx, len(obj_masks))

                # エンコード
                new_frame = av.VideoFrame.from_ndarray(out_img, format="bgr24")
                new_frame.pts = frame.pts
                new_frame.time_base = frame.time_base

                for pkt in v_out.encode(new_frame):
                    container_out.mux(pkt)

                frame_idx += 1
                grand_total = max(grand_total, frame_idx)
                yield (frame_idx, grand_total, "YOLO Detect")

                # 定期ログ
                if frame_idx % 500 == 0:
                    elapsed = time.time() - t0
                    logger.info(f"  [YOLO] {frame_idx} frames "
                                 f"({elapsed:.1f}s, {frame_idx / max(elapsed, 0.01):.0f} fps, "
                                 f"{detect_count} detections)")
                    log_memory(f"frame_{frame_idx}")

            container.close()

            # フラッシュ
            for pkt in v_out.encode():
                container_out.mux(pkt)

            container_out.close()
            container_out = None

            shutil.move(tmp.name, output_path)

            elapsed = time.time() - t0
            total_elapsed = time.time() - t_video_start
            logger.info(f"  [YOLO] Complete: {frame_idx} frames in {elapsed:.1f}s "
                         f"({frame_idx / max(elapsed, 0.01):.0f} fps, "
                         f"{detect_count} with detections)")
            logger.info("=" * 70)
            logger.info(f"COMPLETE: {Path(output_path).name}")
            logger.info(f"  {frame_idx} frames | {total_elapsed:.1f}s total "
                         f"({frame_idx / max(total_elapsed, 0.01):.1f} fps overall)")
            log_memory("video_done")
            logger.info("=" * 70)

        except Exception as e:
            logger.error(f"ERR: {input_path}: {e}", exc_info=True)
            if tmp and os.path.exists(tmp.name):
                os.remove(tmp.name)
            raise
        finally:
            if container_out:
                try:
                    container_out.close()
                except Exception:
                    pass


# =========================
# フォルダ処理 / メイン
# =========================
def process_folder(input_folder, output_folder, encoder):
    Path(output_folder).mkdir(exist_ok=True)

    files = [f for f in os.listdir(input_folder) if f.lower().endswith(".mp4")]
    if not files:
        print("MP4がありません")
        return

    input_paths = [os.path.join(input_folder, f) for f in files]
    output_paths = [
        os.path.join(output_folder, f"{Path(f).stem}_yoloonly.mp4") for f in files
    ]

    tracker = YoloOnlyTracker(device="cuda")

    with Progress(
        TextColumn("{task.fields[name]}", justify="right"),
        TextColumn("[bold cyan]{task.fields[phase]}"),
        BarColumn(),
        FrameProgressColumn(),
    ) as prog:
        task_id = prog.add_task("", name="(starting)", phase="", total=1)

        for inp, out in zip(input_paths, output_paths):
            prog.update(task_id, name=Path(inp).name, phase="", completed=0, total=1)

            last_total = 1
            for c, t, phase in tracker.process_video(inp, out, encoder):
                last_total = t
                prog.update(task_id, completed=c, total=t, phase=phase)

            prog.update(task_id, completed=last_total, total=last_total, phase="Done")


def main():
    print("入力フォルダを選択")
    ip = choose_folder("Input")
    if not ip:
        return

    print("出力フォルダを選択")
    op = choose_folder("Output")
    if not op:
        return

    if not torch.cuda.is_available():
        print("CUDAが使えません。GPU必須です。")
        return

    encoder = pick_encoder()
    gpu = get_gpu_info()
    g0 = gpu["gpus"][0]

    logger.info(
        f"GPU={g0['name']} free={g0['free_gb']:.1f}GB/{g0['total_gb']:.1f}GB | "
        f"encoder={encoder} | YOLO-only mode (every frame)"
    )

    process_folder(ip, op, encoder)


if __name__ == "__main__":
    main()
