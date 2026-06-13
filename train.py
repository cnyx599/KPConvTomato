# train 训练脚本

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader_kpconv_add_stratified import TomatoDataset
from model_kpconv_attention_jl_4unet import KPConvTomato
import numpy as np
from tqdm import tqdm
from datetime import datetime

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

def compute_boundary_iou(pred, target, points, k=16, device='cuda'):
    pred = torch.as_tensor(pred, device=device)
    target = torch.as_tensor(target, device=device)
    points = torch.as_tensor(points, device=device)
    valid_mask = (target != -1)
    if valid_mask.sum() == 0: return float('nan')
    p, t, pts = pred[valid_mask], target[valid_mask], points[valid_mask]
    N = len(p)
    if N < k: return float('nan')
    dists = torch.cdist(pts.unsqueeze(0), pts.unsqueeze(0)).squeeze(0)
    _, knn_idx = torch.topk(dists, k, dim=1, largest=False)
    t_exp = t.unsqueeze(1).expand(-1, k)
    neighbor_labels = t[knn_idx]
    is_boundary = (t_exp != neighbor_labels).any(dim=1)
    if is_boundary.sum() == 0: return 1.0
    p_b, t_b = p[is_boundary], t[is_boundary]
    boundary_ious = []
    for cls in [0, 1]:
        pred_cls = (p_b == cls)
        target_cls = (t_b == cls)
        inter = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        iou = inter / union if union > 0 else float('nan')
        boundary_ious.append(iou)
    return float(np.nanmean(boundary_ious))

def compute_pr_f1(pred, target, num_classes, class_names=None, eps=1e-8):
    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = np.sum((pred == c) & (target == c))
        fp = np.sum((pred == c) & (target != c))
        fn = np.sum((pred != c) & (target == c))
        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        f1 = 2 * p * r / (p + r + eps)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
    macro_p = np.mean(precisions)
    macro_r = np.mean(recalls)
    macro_f1 = np.mean(f1s)
    return precisions, recalls, f1s, macro_p, macro_r, macro_f1

# 边界感知损失函数
def boundary_aware_loss(pred, target, points, alpha=2.0, radius=0.018, ignore_index=-1, lambda_cont=0.2):
    valid_mask = (target != ignore_index)
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    
    points, target, pred = points[valid_mask], target[valid_mask], pred[valid_mask]
    N = points.shape[0]
    if N == 0:
        return torch.tensor(0.0, device=pred.device)
    
    # 边界加权分类损失
    dists = torch.cdist(points, points)
    k = 16
    _, knn_idx = torch.topk(dists, k, dim=1, largest=False)
    knn_dists = torch.gather(dists, 1, knn_idx)
    adaptive_radius = knn_dists.mean(dim=1, keepdim=True) * 1.5
    neighbor_mask = dists < adaptive_radius
    
    label_i = target.unsqueeze(0).expand(N, -1)
    label_j = target.unsqueeze(1).expand(-1, N)
    has_diff_neighbor = ((label_i != label_j) & neighbor_mask).any(dim=1)
    
    sample_weights = torch.ones(N, device=target.device)
    sample_weights[has_diff_neighbor] = alpha
    
    loss_per_sample = nn.functional.cross_entropy(pred, target, ignore_index=ignore_index, reduction='none')
    weighted_loss = (loss_per_sample * sample_weights).sum() / sample_weights.sum()
    
    # 边界连续性正则化
    if has_diff_neighbor.any():
        boundary_points = points[has_diff_neighbor]
        M = boundary_points.shape[0]
        if M > 2:
            bd_dists = torch.cdist(boundary_points, boundary_points)
            _, bd_knn_idx = torch.topk(bd_dists, min(8, M), dim=1, largest=False)
            bd_knn_dists = torch.gather(bd_dists, 1, bd_knn_idx)
            bd_radius = bd_knn_dists.mean() * 1.2
            
            should_be_same = bd_dists < bd_radius
            
            pred_probs = torch.softmax(pred[has_diff_neighbor], dim=1) # (M, C)
            
            diff = pred_probs.unsqueeze(1) - pred_probs.unsqueeze(0) # (M, M, C)
            dist_sq = torch.sum(diff ** 2, dim=-1) # (M, M)
            
            mask = should_be_same & (~torch.eye(M, dtype=torch.bool, device=points.device))
            
            if mask.any():
                continuity_loss = (dist_sq * mask.float()).sum() / mask.float().sum() * lambda_cont
                weighted_loss = weighted_loss + continuity_loss
                
    return weighted_loss

def load_weights_if_exists(model, ckpt_path, device):
    if ckpt_path and os.path.isfile(ckpt_path):
        print(f"📥 Loading model weights from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        pretrained_dict = {k: v for k, v in ckpt['model_state_dict'].items() if k in model.state_dict()}
        model.load_state_dict(pretrained_dict, strict=False)
        print(f"✅ Successfully loaded {len(pretrained_dict)}/{len(model.state_dict())} parameter groups.")
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

def save_ply_with_rgb_and_attention(points, rgb, attention, labels=None, filename="output.ply"):
    if isinstance(points, torch.Tensor): points = points.cpu().numpy()
    if isinstance(rgb, torch.Tensor): rgb = rgb.cpu().numpy()
    if isinstance(attention, torch.Tensor): attention = attention.cpu().numpy()
    if attention.ndim == 2: attention = attention.squeeze(1)
    if labels is not None and isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()
    if rgb.max() <= 1.0:
        rgb = (rgb * 255).astype(np.uint8)
    else:
        rgb = rgb.astype(np.uint8)
    N = points.shape[0]
    header_lines = ["ply", "format ascii 1.0", f"element vertex {N}", "property float x", "property float y",
                    "property float z", "property uchar red", "property uchar green", "property uchar blue",
                    "property float attention"]
    if labels is not None: header_lines.append("property int label")
    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"
    if labels is not None:
        data = np.column_stack([points, rgb, attention, labels])
        fmt = "%.6f %.6f %.6f %d %d %d %.6f %d"
    else:
        data = np.column_stack([points, rgb, attention])
        fmt = "%.6f %.6f %.6f %d %d %d %.6f"
    with open(filename, 'w') as f:
        f.write(header)
        np.savetxt(f, data, fmt=fmt)
    print(f"✅ Saved PLY with RGB + attention to: {filename}")

def main():
    PRETRAINED_CKPT = "../xxx"
    data_root = '../xxx'
    num_points = 8192
    batch_size = 1
    num_epochs = 100
    lr = 0.001
    num_classes = 2
    class_names = ['stem', 'leaf']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_dataset = TomatoDataset(root_dir=data_root, num_points=num_points, split='train', augment=True)
    val_dataset = TomatoDataset(root_dir=data_root, num_points=num_points, split='val', augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, collate_fn=KPConvTomato.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=KPConvTomato.collate_fn)

    model = KPConvTomato(d_in=6, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    load_weights_if_exists(model, PRETRAINED_CKPT, device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f'./checkpoints_kpconv_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    print(f"🆕 Training session started. Results saved to: {save_dir}")

    metrics = {
        'train_losses': [], 'val_losses': [], 'val_mious': [],
        'stem_ious': [], 'leaf_ious': [], 'boundary_ious': [],
        'stem_precisions': [], 'leaf_precisions': [], 'macro_precisions': [],
        'stem_recalls': [], 'leaf_recalls': [], 'macro_recalls': [],
        'stem_f1s': [], 'leaf_f1s': [], 'macro_f1s': []
    }
    best_miou = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"):
            batch_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else [t.to(device) for t in v]) for k, v in
                            batch.items()}
            optimizer.zero_grad()
            outputs, orig_pts = model(batch_device)
            labels_flat = torch.cat(batch_device['labels'], dim=0)
            loss = boundary_aware_loss(outputs, labels_flat, orig_pts, alpha=2.0, radius=0.02, ignore_index=-1)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        all_ious, boundary_ious = [], []
        per_class_accum = {name: [] for name in class_names}
        per_class_p_accum = {name: [] for name in class_names}
        per_class_r_accum = {name: [] for name in class_names}
        per_class_f1_accum = {name: [] for name in class_names}
        macro_p_list, macro_r_list, macro_f1_list = [], [], []

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                batch_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else [t.to(device) for t in v]) for k, v in batch.items()}
                outputs, orig_pts = model(batch_device)
                labels_flat = torch.cat(batch_device['labels'], dim=0)
                val_loss += boundary_aware_loss(outputs, labels_flat, orig_pts, alpha=2.0, radius=0.02, ignore_index=-1).item()

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                labels_np = labels_flat.cpu().numpy()
                mean_iou, per_class_iou = compute_iou_per_class(preds, labels_np, num_classes, class_names=class_names)
                all_ious.append(mean_iou)
                b_iou = compute_boundary_iou(preds, labels_np, orig_pts, k=16, device=device)
                if not np.isnan(b_iou): boundary_ious.append(b_iou)

                for name in class_names:
                    iou_val = per_class_iou[name]
                    if not np.isnan(iou_val): per_class_accum[name].append(iou_val)

                cls_p, cls_r, cls_f1, m_p, m_r, m_f1 = compute_pr_f1(preds, labels_np, num_classes, class_names)
                for i, name in enumerate(class_names):
                    per_class_p_accum[name].append(cls_p[i])
                    per_class_r_accum[name].append(cls_r[i])
                    per_class_f1_accum[name].append(cls_f1[i])
                macro_p_list.append(m_p)
                macro_r_list.append(m_r)
                macro_f1_list.append(m_f1)

                if epoch >= num_epochs - 5 or epoch % 10 == 0:
                    start_sample_idx = idx * val_loader.batch_size
                    end_sample_idx = start_sample_idx + len(batch['points'])
                    if start_sample_idx < 3:
                        for local_i, (pts, feats, lbls) in enumerate(zip(batch['points'], batch['features'], batch['labels'])):
                            global_sample_idx = start_sample_idx + local_i
                            if global_sample_idx >= 3: break
                            pts_dev = pts.to(device)
                            lbls_dev = lbls.to(device)
                            with torch.no_grad():
                                attention = model.boundary_attention(pts_dev, lbls_dev)
                            vis_dir = os.path.join(save_dir, "visualizations")
                            os.makedirs(vis_dir, exist_ok=True)
                            ply_path = os.path.join(vis_dir, f"epoch_{epoch + 1:03d}_sample_{global_sample_idx}.ply")
                            try:
                                save_ply_with_rgb_and_attention(pts.cpu(), feats.cpu()[:, 3:6], attention.cpu(),
                                                                lbls.cpu(), ply_path)
                            except Exception as e:
                                print(f"⚠️ Failed to save attention for sample {global_sample_idx}: {e}")

        avg_val_loss = val_loss / len(val_loader)
        epoch_mean_iou = np.nanmean(all_ious) if all_ious else 0.0
        epoch_stem_iou = np.mean(per_class_accum['stem']) if per_class_accum['stem'] else float('nan')
        epoch_leaf_iou = np.mean(per_class_accum['leaf']) if per_class_accum['leaf'] else float('nan')
        epoch_boundary_iou = np.mean(boundary_ious) if boundary_ious else float('nan')

        epoch_stem_p = np.mean(per_class_p_accum['stem']) if per_class_p_accum['stem'] else 0.0
        epoch_leaf_p = np.mean(per_class_p_accum['leaf']) if per_class_p_accum['leaf'] else 0.0
        epoch_macro_p = np.mean(macro_p_list) if macro_p_list else 0.0
        epoch_stem_r = np.mean(per_class_r_accum['stem']) if per_class_r_accum['stem'] else 0.0
        epoch_leaf_r = np.mean(per_class_r_accum['leaf']) if per_class_r_accum['leaf'] else 0.0
        epoch_macro_r = np.mean(macro_r_list) if macro_r_list else 0.0
        epoch_stem_f1 = np.mean(per_class_f1_accum['stem']) if per_class_f1_accum['stem'] else 0.0
        epoch_leaf_f1 = np.mean(per_class_f1_accum['leaf']) if per_class_f1_accum['leaf'] else 0.0
        epoch_macro_f1 = np.mean(macro_f1_list) if macro_f1_list else 0.0

        metrics['train_losses'].append(avg_train_loss)
        metrics['val_losses'].append(avg_val_loss)
        metrics['val_mious'].append(epoch_mean_iou)
        metrics['stem_ious'].append(epoch_stem_iou)
        metrics['leaf_ious'].append(epoch_leaf_iou)
        metrics['boundary_ious'].append(epoch_boundary_iou)
        metrics['stem_precisions'].append(epoch_stem_p)
        metrics['leaf_precisions'].append(epoch_leaf_p)
        metrics['macro_precisions'].append(epoch_macro_p)
        metrics['stem_recalls'].append(epoch_stem_r)
        metrics['leaf_recalls'].append(epoch_leaf_r)
        metrics['macro_recalls'].append(epoch_macro_r)
        metrics['stem_f1s'].append(epoch_stem_f1)
        metrics['leaf_f1s'].append(epoch_leaf_f1)
        metrics['macro_f1s'].append(epoch_macro_f1)

        print(f"Epoch {epoch + 1}: "
              f"Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, "
              f"mIoU={epoch_mean_iou:.4f}, BndryIoU={epoch_boundary_iou:.4f}, "
              f"Stem IoU={epoch_stem_iou:.4f}, Leaf IoU={epoch_leaf_iou:.4f}, "
              f"Macro F1={epoch_macro_f1:.4f} (Stem={epoch_stem_f1:.4f}, Leaf={epoch_leaf_f1:.4f}), "
              f"LR={optimizer.param_groups[0]['lr']:.6f}")

        if epoch_mean_iou > best_miou:
            best_miou = epoch_mean_iou
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'best_miou': best_miou,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"✅ New best model (mIoU: {best_miou:.4f}) saved to {save_dir}")

        scheduler.step(epoch_mean_iou)
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump(nan_to_null(metrics), f, indent=4)

    print(f"\n🎉 Training finished. Best mIoU: {best_miou:.4f}")

    print("\n🔍 Running final evaluation on TEST set...")
    test_dataset = TomatoDataset(root_dir=data_root, num_points=num_points, split='test', augment=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, collate_fn=KPConvTomato.collate_fn)

    model.eval()
    test_ious, test_boundary_ious = [], []
    test_stem_ious, test_leaf_ious = [], []
    test_stem_p, test_leaf_p = [], []
    test_stem_r, test_leaf_r = [], []
    test_stem_f1, test_leaf_f1 = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            batch_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else [t.to(device) for t in v]) for k, v in batch.items()}
            outputs, pred_pts = model(batch_device)
            labels_flat = torch.cat(batch_device['labels'], dim=0)
            orig_pts = torch.cat(batch_device['points'], dim=0)
            dists = torch.cdist(orig_pts, pred_pts)
            _, nn_idx = torch.min(dists, dim=1)
            outputs_interp = outputs[nn_idx]

            preds = torch.argmax(outputs_interp, dim=1).cpu().numpy()
            labels_np = labels_flat.cpu().numpy()
            mean_iou, per_class_iou = compute_iou_per_class(preds, labels_np, num_classes, class_names)
            test_ious.append(mean_iou)

            b_iou = compute_boundary_iou(preds, labels_np, orig_pts, k=16, device=device)
            if not np.isnan(b_iou): test_boundary_ious.append(b_iou)

            if 'stem' in per_class_iou and not np.isnan(per_class_iou['stem']): test_stem_ious.append(per_class_iou['stem'])
            if 'leaf' in per_class_iou and not np.isnan(per_class_iou['leaf']): test_leaf_ious.append(per_class_iou['leaf'])

            cls_p, cls_r, cls_f1, _, _, _ = compute_pr_f1(preds, labels_np, num_classes, class_names)
            test_stem_p.append(cls_p[0]);
            test_leaf_p.append(cls_p[1])
            test_stem_r.append(cls_r[0]);
            test_leaf_r.append(cls_r[1])
            test_stem_f1.append(cls_f1[0]);
            test_leaf_f1.append(cls_f1[1])

    final_test_miou = np.nanmean(test_ious)
    final_test_stem = np.mean(test_stem_ious) if test_stem_ious else float('nan')
    final_test_leaf = np.mean(test_leaf_ious) if test_leaf_ious else float('nan')
    final_test_boundary = np.mean(test_boundary_ious) if test_boundary_ious else float('nan')
    final_test_stem_p = np.mean(test_stem_p)
    final_test_leaf_p = np.mean(test_leaf_p)
    final_test_stem_r = np.mean(test_stem_r)
    final_test_leaf_r = np.mean(test_leaf_r)
    final_test_stem_f1 = np.mean(test_stem_f1)
    final_test_leaf_f1 = np.mean(test_leaf_f1)

    print(f"\n🏆 FINAL TEST RESULTS:")
    print(f"   mIoU: {final_test_miou:.4f} | BndryIoU: {final_test_boundary:.4f}")
    print(
        f"   Stem  -> IoU={final_test_stem:.4f} | P={final_test_stem_p:.4f} | R={final_test_stem_r:.4f} | F1={final_test_stem_f1:.4f}")
    print(
        f"   Leaf  -> IoU={final_test_leaf:.4f} | P={final_test_leaf_p:.4f} | R={final_test_leaf_r:.4f} | F1={final_test_leaf_f1:.4f}")

    test_metrics = {
        'test_mIoU': final_test_miou, 'test_BoundaryIoU': final_test_boundary,
        'test_stem_IoU': final_test_stem, 'test_leaf_IoU': final_test_leaf,
        'test_stem_P': final_test_stem_p, 'test_stem_R': final_test_stem_r, 'test_stem_F1': final_test_stem_f1,
        'test_leaf_P': final_test_leaf_p, 'test_leaf_R': final_test_leaf_r, 'test_leaf_F1': final_test_leaf_f1,
        'test_samples': len(test_dataset)
    }
    with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
        json.dump(nan_to_null(test_metrics), f, indent=4)
    print(f"✅ Test results saved to {save_dir}/test_results.json")


if __name__ == '__main__':
    main()