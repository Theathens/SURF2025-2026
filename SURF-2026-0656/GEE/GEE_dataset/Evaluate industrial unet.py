"""
evaluate_industrial_unet.py
=============================
SURF 2026 | T2 | Step1 收尾脚本 ②

在真实 10m Sentinel-2 工业影像（GEE 导出的验证集）上评估
训练好的 U-Net，输出:
    1. 每类 IoU / Precision / Recall / F1（不只是平均 mIoU）
    2. 整体像素准确率、混淆矩阵
    3. 预测可视化（原图 | 真实掩码 | 预测掩码 三联图）
    4. Excel 格式的验证报告

为什么需要这一步？
─────────────────────────────────────────────────
之前在 DOTA 数据集上跑的 mIoU=0.467 衡量的是飞机/车辆检测能力，
和"工业设施边界分割"是完全不同的任务。重新在 GEE 工业数据上
训练后，必须用这批数据自己的验证集重新评估，
之前的指标不能沿用。

用法:
    python evaluate_industrial_unet.py \\
        --image_dir   ./gee_output/image \\
        --mask_dir    ./gee_output/mask \\
        --classes_json ./gee_output/classes.json \\
        --model_path  unet_industrial_best.pth \\
        --output_dir  ./eval_output \\
        --val_ratio   0.2 \\
        --n_visualize 12
─────────────────────────────────────────────────
"""

import os
import json
import warnings

import numpy as np
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch

warnings.filterwarnings("ignore")


# ============================================================
# 1. 复用训练脚本里的模型定义（避免重复维护两份代码）
# ============================================================
def load_model_class():
    """
    动态导入 train_unet_effb0.py 里的 EfficientUNet 类。
    要求 evaluate_industrial_unet.py 与 train_unet_effb0.py
    放在同一目录下。
    """
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(here, "train_unet_effb0_industrial.py")
    if not os.path.exists(train_script):
        raise FileNotFoundError(
            "未找到 train_unet_effb0.py，请将其放在与本脚本相同目录下"
        )
    spec = importlib.util.spec_from_file_location("train_unet_effb0", train_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EfficientUNet


# ============================================================
# 2. 每类指标计算
# ============================================================
def compute_per_class_metrics(preds: np.ndarray, targets: np.ndarray,
                              num_classes: int):
    """
    preds, targets: (N, H, W) numpy int array
    返回: dict，每个类别的 IoU / Precision / Recall / F1，
          以及整体像素准确率和混淆矩阵
    """
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(preds, targets):
        for c_true in range(num_classes):
            mask_true = (t == c_true)
            for c_pred in range(num_classes):
                confusion[c_true, c_pred] += int(
                    np.sum(mask_true & (p == c_pred))
                )

    per_class = {}
    for c in range(num_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        union = tp + fp + fn

        iou = tp / union if union > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 and not np.isnan(precision)
                 and not np.isnan(recall) else float("nan"))

        per_class[c] = {
            "IoU": iou, "Precision": precision,
            "Recall": recall, "F1": f1,
            "support_pixels": int(confusion[c, :].sum()),
        }

    pixel_acc = np.trace(confusion) / confusion.sum()
    mean_iou = np.nanmean([per_class[c]["IoU"] for c in range(num_classes)])

    return {
        "per_class": per_class,
        "confusion_matrix": confusion,
        "pixel_accuracy": float(pixel_acc),
        "mean_iou": float(mean_iou),
    }


# ============================================================
# 3. 推理
# ============================================================
def run_inference(model, image_paths, input_size, device):
    """对一批图像跑推理，返回 (preds, raw_images)"""
    preds_list = []
    images_list = []

    model.eval()
    with torch.no_grad():
        for path in image_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[:2] != input_size:
                img = cv2.resize(img, (input_size[1], input_size[0]),
                                 interpolation=cv2.INTER_LINEAR)

            t = torch.from_numpy(img.astype(np.float32) / 255.0)
            t = t.permute(2, 0, 1).unsqueeze(0).to(device)

            out = model(t)
            pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()

            preds_list.append(pred)
            images_list.append(img)

    return preds_list, images_list


def load_masks(mask_paths, input_size):
    masks = []
    for path in mask_paths:
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m.shape != input_size:
            m = cv2.resize(m, (input_size[1], input_size[0]),
                           interpolation=cv2.INTER_NEAREST)
        masks.append(m)
    return masks


# ============================================================
# 4. 可视化：原图 | 真实掩码 | 预测掩码
# ============================================================
def visualize_predictions(images, masks, preds, names, out_dir,
                          num_classes, n_visualize=12):
    os.makedirs(out_dir, exist_ok=True)
    n = min(n_visualize, len(images))

    cmap = plt.get_cmap("tab10")

    for i in range(n):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(images[i])
        axes[0].set_title("卫星影像 (RGB)", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(masks[i], cmap=cmap, vmin=0, vmax=num_classes - 1)
        axes[1].set_title("真实标签 (GT)", fontsize=10)
        axes[1].axis("off")

        axes[2].imshow(preds[i], cmap=cmap, vmin=0, vmax=num_classes - 1)
        axes[2].set_title("U-Net 预测", fontsize=10)
        axes[2].axis("off")

        fig.suptitle(names[i], fontsize=9)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"compare_{names[i]}")
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()

    print(f"  可视化对比图已保存 {n} 张 → {out_dir}/")


def plot_confusion_matrix(confusion, class_labels, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("混淆矩阵")

    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            ax.text(j, i, f"{confusion[i, j]:,}",
                    ha="center", va="center",
                    color="white" if confusion[i, j] > confusion.max() / 2 else "black",
                    fontsize=8)

    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# 5. Excel 报告
# ============================================================
def write_eval_report(metrics: dict, class_labels: list, out_path: str):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    wb = Workbook()
    ws = wb.active
    ws.title = "验证报告"

    header_fill = PatternFill("solid", fgColor="1F4E79")
    header_font = Font(color="FFFFFF", bold=True, name="Arial", size=10)
    border = Border(left=Side("thin"), right=Side("thin"),
                    top=Side("thin"), bottom=Side("thin"))

    # 总体指标
    ws["A1"] = "总体指标"
    ws["A1"].font = Font(bold=True, size=12)
    ws["A2"], ws["B2"] = "像素准确率 (Pixel Accuracy)", round(metrics["pixel_accuracy"], 4)
    ws["A3"], ws["B3"] = "平均 IoU (mIoU)", round(metrics["mean_iou"], 4)
    ws["A4"], ws["B4"] = "样本数 (验证集 patch 数)", metrics.get("n_samples", "")

    # 每类指标表
    start_row = 6
    headers = ["类别", "IoU", "Precision", "Recall", "F1", "像素数(支持度)"]
    for col_i, h in enumerate(headers, start=1):
        cell = ws.cell(row=start_row, column=col_i, value=h)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center")
        cell.border = border

    for i, c in enumerate(sorted(metrics["per_class"].keys())):
        row = start_row + 1 + i
        label = class_labels[c] if c < len(class_labels) else f"class_{c}"
        m = metrics["per_class"][c]
        vals = [label, m["IoU"], m["Precision"], m["Recall"], m["F1"], m["support_pixels"]]
        for col_i, v in enumerate(vals, start=1):
            cell = ws.cell(row=row, column=col_i, value=round(v, 4) if isinstance(v, float) else v)
            cell.border = border
            cell.alignment = Alignment(horizontal="center")
            cell.font = Font(name="Arial", size=10)

    for i, w in enumerate([16, 12, 14, 12, 12, 16], start=1):
        from openpyxl.utils import get_column_letter
        ws.column_dimensions[get_column_letter(i)].width = w

    wb.save(out_path)
    print(f"\nExcel 验证报告已保存: {out_path}")


# ============================================================
# 主流程
# ============================================================
def main():
    # Hardcoded paths
    image_dir = "//GEE/merged_industrial/image"
    mask_dir = "//GEE/merged_industrial/mask"
    classes_json = "D:/Pycharm Projects/SURF-2026-0656/GEE/merged_industrial/classes.json"
    model_path = "//GEE/output_industrial_combine_GEE_BEN/unet_industrial_combine_GEE_BEN_best.pth"
    output_dir = "./eval_output"
    input_size = (128, 128)
    val_ratio = 0.2
    n_visualize = 12

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    with open(classes_json) as f:
        class_info = json.load(f)
    num_classes = class_info["num_classes"]
    class_labels = ["background"] + class_info["classes"]
    print(f"类别数: {num_classes} → {class_labels}")

    # ── 划分验证集（与训练脚本相同的 random_state，避免数据泄漏） ──
    supported = (".png", ".jpg", ".jpeg")
    files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(supported))
    image_paths = [os.path.join(image_dir, f) for f in files
                   if os.path.exists(os.path.join(mask_dir, f))]
    mask_paths = [os.path.join(mask_dir, f) for f in files
                  if os.path.exists(os.path.join(mask_dir, f))]

    _, val_img, _, val_mask = train_test_split(
        image_paths, mask_paths, test_size=val_ratio, random_state=42
    )
    val_names = [os.path.basename(p) for p in val_img]
    print(f"验证集样本数: {len(val_img)}")

    # ── 加载模型 ──────────────────────────────────────────────
    EfficientUNet = load_model_class()
    model = EfficientUNet(num_classes=num_classes, pretrained=False).to(device)
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    print(f"已加载模型: {model_path}")

    # ── 推理 + 加载真实标签 ──────────────────────────────────
    print("正在对验证集推理...")
    preds, images = run_inference(model, val_img, input_size, device)
    masks = load_masks(val_mask, input_size)

    # ── 计算指标 ──────────────────────────────────────────────
    metrics = compute_per_class_metrics(
        np.array(preds), np.array(masks), num_classes
    )
    metrics["n_samples"] = len(val_img)

    print(f"\n像素准确率: {metrics['pixel_accuracy']:.4f}")
    print(f"平均 IoU:   {metrics['mean_iou']:.4f}")
    for c in sorted(metrics["per_class"].keys()):
        m = metrics["per_class"][c]
        label = class_labels[c] if c < len(class_labels) else f"class_{c}"
        print(f"  [{label:12s}] IoU={m['IoU']:.4f}  "
              f"Precision={m['Precision']:.4f}  Recall={m['Recall']:.4f}  "
              f"F1={m['F1']:.4f}  支持度={m['support_pixels']:,}px")

    # ── 可视化 ────────────────────────────────────────────────
    vis_dir = os.path.join(output_dir, "predictions")
    visualize_predictions(images, masks, preds, val_names, vis_dir,
                          num_classes, n_visualize)

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(metrics["confusion_matrix"], class_labels, cm_path)
    print(f"  混淆矩阵已保存 → {cm_path}")

    # ── Excel 报告 ────────────────────────────────────────────
    report_path = os.path.join(output_dir, "evaluation_report.xlsx")
    write_eval_report(metrics, class_labels, report_path)

    print(f"\n全部完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    main()