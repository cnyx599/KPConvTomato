# train_kpconv_bjss_attention_jl2_tvt.py
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src_my_kpconv.data_loader_tvt import TomatoDataset
from src_my_kpconv.model_kpconv_attention_jl2 import KPConvTomato
import numpy as np
from tqdm import tqdm
from datetime import datetime

# 基于kpconv的训练脚本（分别输出茎叶Iou、训练结果迭代、边界感知损失函数）

def compute_iou_per_class(pred, target, num_classes, class_names=None):
    iou_dict = {}
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        iou = float('nan') if union == 0 else intersection / union
        ious.append(iou)
        cls_key = class_names[cls] if class_names else f"class_{cls}"
        iou_dict[cls_key] = iou
    mean_iou = np.nanmean(ious) if ious else 0.0
    return mean_iou, iou_dict

# 边界感知损失函数
def boundary_aware_loss(pred, target, points, alpha=1.3, radius=0.018, ignore_index=-1):
    # 仅处理有效点
    valid_mask = (target != ignore_index)
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    points = points[valid_mask]
    target = target[valid_mask]
    pred = pred[valid_mask]
    N = points.shape[0]

    if N == 0:
        return torch.tensor(0.0, device=pred.device)

    # === 计算边界掩码 ===
    dists = torch.cdist(points, points)  # [N, N]

    # 计算局部平均邻域距离（动态 radius）
    k = 16  # 取16近邻
    _, knn_idx = torch.topk(dists, k, dim=1, largest=False)
    knn_dists = torch.gather(dists, 1, knn_idx)
    adaptive_radius = knn_dists.mean(dim=1, keepdim=True) * 1.5  # 动态阈值
    neighbor_mask = dists < adaptive_radius  # [N, N]

    label_i = target.unsqueeze(0).expand(N, -1)
    label_j = target.unsqueeze(1).expand(-1, N)
    has_diff_neighbor = ((label_i != label_j) & neighbor_mask).any(dim=1)  # [N]

    # === 构建样本级权重 ===
    sample_weights = torch.ones(N, device=target.device)
    sample_weights[has_diff_neighbor] = alpha  # 边界点权重=alpha

    # === 计算逐样本loss + 加权平均 ===
    loss_per_sample = nn.functional.cross_entropy(
        pred, target,
        ignore_index=ignore_index,
        reduction='none'
    )
    weighted_loss = (loss_per_sample * sample_weights).sum() / sample_weights.sum()

    # ========== 边界点连续性约束==========
    # 仅当存在边界点时计算（避免除零）
    if has_diff_neighbor.any():
        # 提取边界点子集
        boundary_points = points[has_diff_neighbor]  # [M, 3]
        boundary_labels = target[has_diff_neighbor]  # [M]
        M = boundary_points.shape[0]

        if M > 2:  # 至少3个边界点才计算
            # 计算边界点间距离矩阵
            bd_dists = torch.cdist(boundary_points, boundary_points)  # [M, M]

            # 自适应邻域半径：用边界点自身密度
            _, bd_knn_idx = torch.topk(bd_dists, min(8, M), dim=1, largest=False)
            bd_knn_dists = torch.gather(bd_dists, 1, bd_knn_idx)
            bd_radius = bd_knn_dists.mean() * 1.2

            # 构建“应同类别”掩码：距离 < bd_radius 的点对
            should_be_same = bd_dists < bd_radius  # [M, M], 对角线为True

            # 实际标签是否相同
            bd_label_i = boundary_labels.unsqueeze(0).expand(M, -1)
            bd_label_j = boundary_labels.unsqueeze(1).expand(-1, M)
            is_same_label = (bd_label_i == bd_label_j)  # [M, M]

            # 连续性损失：应同但不同 → 惩罚
            inconsistency = (~is_same_label) & should_be_same
            # 排除自比较（对角线）
            inconsistency = inconsistency & (~torch.eye(M, dtype=torch.bool, device=points.device))

            if inconsistency.any():
                continuity_loss = inconsistency.float().mean() * 0.2  # 权重=0.2（可调）
                weighted_loss = weighted_loss + continuity_loss
    # ========================================================

    return weighted_loss

def load_weights_if_exists(model, ckpt_path, device):
    if ckpt_path and os.path.isfile(ckpt_path):
        print(f"📥 Loading model weights from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        # 宽松加载：只加载匹配的参数（避免结构微调时报错）
        pretrained_dict = {k: v for k, v in ckpt['model_state_dict'].items() if k in model.state_dict()}
        model.load_state_dict(pretrained_dict, strict=False)
        n_loaded = len(pretrained_dict)
        n_total = len(model.state_dict())
        print(f"✅ Successfully loaded {n_loaded}/{n_total} parameter groups.")
        return True
    else:
        print(f"⚠️ Pretrained checkpoint not found: {ckpt_path}")
        print("🆕 Training from scratch.")
        return False

def nan_to_null(obj):
    if isinstance(obj, dict):
        return {k: nan_to_null(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [nan_to_null(v) for v in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    else:
        return obj

def main():
    # 指定预训练权重路径
    PRETRAINED_CKPT = "./checkpoints_20251230_121137_b+a1+j2_60+15+15/best_model.pth"

    # 配置
    data_root = '../dataset_my1128+1205_tvt'
    num_points = 8192
    batch_size = 1
    num_epochs = 50
    lr = 0.001
    num_classes = 2
    class_names = ['stem', 'leaf']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据集
    train_dataset = TomatoDataset(root_dir=data_root, num_points=num_points, split='train', augment=True)
    val_dataset = TomatoDataset(root_dir=data_root, num_points=num_points, split='val', augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, collate_fn=KPConvTomato.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=KPConvTomato.collate_fn)

    # 模型与优化器（始终新初始化）
    model = KPConvTomato(d_in=6, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

    # 加载预训练权重（若路径有效）
    load_weights_if_exists(model, PRETRAINED_CKPT, device)

    # 新建本次训练目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f'./checkpoints_separate_kpconv_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    print(f"🆕 Training session started. Results saved to: {save_dir}")

    # 初始化指标记录
    metrics = {
        'train_losses': [],
        'val_losses': [],
        'val_mious': [],
        'stem_ious': [],
        'leaf_ious': []
    }
    best_miou = 0.0  # 本次训练的 best mIoU

    for epoch in range(num_epochs):
        # ========== Train ==========
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"):
            batch_device = {
                'points': [p.to(device) for p in batch['points']],
                'features': [f.to(device) for f in batch['features']],
                'labels': [l.to(device) for l in batch['labels']],
                'lengths': batch['lengths'].to(device),
            }

            optimizer.zero_grad()
            outputs, orig_pts = model(batch_device)  # outputs: [N_orig, C], orig_pts: [N_orig, 3]
            labels_flat = torch.cat(batch_device['labels'], dim=0)  # [N_orig]

            # 直接使用 outputs（无需额外插值！）
            loss = boundary_aware_loss(
                outputs, labels_flat, orig_pts,
                alpha=2.0, radius=0.02, ignore_index=-1
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ========== Val ==========
        model.eval()
        val_loss = 0.0
        all_ious = []
        per_class_accum = {name: [] for name in class_names}

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch_device = {
                    'points': [p.to(device) for p in batch['points']],
                    'features': [f.to(device) for f in batch['features']],
                    'labels': [l.to(device) for l in batch['labels']],
                    'lengths': batch['lengths'].to(device),
                }

                outputs, orig_pts = model(batch_device)
                labels_flat = torch.cat(batch_device['labels'], dim=0)

                val_loss += boundary_aware_loss(
                    outputs, labels_flat, orig_pts,
                    alpha=2.0, radius=0.02, ignore_index=-1
                ).item()

                preds = torch.argmax(outputs, dim=1).cpu().numpy()      # [N_orig]
                labels_np = labels_flat.cpu().numpy()       # [N_orig]
                mean_iou, per_class_iou = compute_iou_per_class(preds, labels_np, num_classes, class_names=class_names)
                all_ious.append(mean_iou)

                for name in class_names:
                    iou_val = per_class_iou[name]
                    if not np.isnan(iou_val):
                        per_class_accum[name].append(iou_val)

        avg_val_loss = val_loss / len(val_loader)
        epoch_mean_iou = np.nanmean(all_ious) if all_ious else 0.0
        epoch_stem_iou = np.mean(per_class_accum['stem']) if per_class_accum['stem'] else float('nan')
        epoch_leaf_iou = np.mean(per_class_accum['leaf']) if per_class_accum['leaf'] else float('nan')

        # 记录
        metrics['train_losses'].append(avg_train_loss)
        metrics['val_losses'].append(avg_val_loss)
        metrics['val_mious'].append(epoch_mean_iou)
        metrics['stem_ious'].append(epoch_stem_iou)
        metrics['leaf_ious'].append(epoch_leaf_iou)

        print(f"Epoch {epoch + 1}: "
              f"Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, "
              f"mIoU={epoch_mean_iou:.4f}, "
              f"stem IoU={epoch_stem_iou:.4f}, "
              f"leaf IoU={epoch_leaf_iou:.4f}, "
              f"LR={scheduler.get_last_lr()[0]:.6f}")

        # 保存本次训练中最佳模型
        if epoch_mean_iou > best_miou:
            best_miou = epoch_mean_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"✅ New best model (mIoU: {best_miou:.4f}) saved to {save_dir}")

        scheduler.step()

        # 保存 metrics
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump(nan_to_null(metrics), f, indent=4)

    print(f"\n🎉 Training finished. Best mIoU: {best_miou:.4f}")

    # ========== Final Test Evaluation ==========
    print("\n🔍 Running final evaluation on TEST set...")
    test_dataset = TomatoDataset(root_dir=data_root, num_points=num_points, split='test', augment=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, collate_fn=KPConvTomato.collate_fn)

    model.eval()
    test_ious = []
    test_stem_ious = []
    test_leaf_ious = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            batch_device = {
                'points': [p.to(device) for p in batch['points']],
                'features': [f.to(device) for f in batch['features']],
                'labels': [l.to(device) for l in batch['labels']],
                'lengths': batch['lengths'].to(device),
            }

            outputs, pred_pts = model(batch_device)

            labels_flat = torch.cat(batch_device['labels'], dim=0)
            orig_pts = torch.cat(batch_device['points'], dim=0)

            dists = torch.cdist(orig_pts, pred_pts)
            _, nn_idx = torch.min(dists, dim=1)
            outputs_interp = outputs[nn_idx]

            preds = torch.argmax(outputs_interp, dim=1).cpu().numpy()
            labels_np = labels_flat.cpu().numpy()
            mean_iou, per_class_iou = compute_iou_per_class(
                preds, labels_np, num_classes, class_names=class_names
            )
            test_ious.append(mean_iou)

            if 'stem' in per_class_iou and not np.isnan(per_class_iou['stem']):
                test_stem_ious.append(per_class_iou['stem'])
            if 'leaf' in per_class_iou and not np.isnan(per_class_iou['leaf']):
                test_leaf_ious.append(per_class_iou['leaf'])

    final_test_miou = np.nanmean(test_ious)
    final_test_stem = np.mean(test_stem_ious) if test_stem_ious else float('nan')
    final_test_leaf = np.mean(test_leaf_ious) if test_leaf_ious else float('nan')

    print(f"\n🏆 FINAL TEST RESULTS:")
    print(f"   mIoU:      {final_test_miou:.4f}")
    print(f"   Stem IoU:  {final_test_stem:.4f}")
    print(f"   Leaf IoU:  {final_test_leaf:.4f}")

    # 保存 test 结果
    test_metrics = {
        'test_mIoU': final_test_miou,
        'test_stem_IoU': final_test_stem,
        'test_leaf_IoU': final_test_leaf,
        'test_samples': len(test_dataset)
    }
    with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
        json.dump(nan_to_null(test_metrics), f, indent=4)

    print(f"✅ Test results saved to {save_dir}/test_results.json")


if __name__ == '__main__':
    main()