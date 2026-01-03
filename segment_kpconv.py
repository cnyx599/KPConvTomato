# segment_kpconv.py
import os
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from old.model_kpconv import KPConvTomato   # 换新神经网络 显存直接崩

# 基于kpconv的单株番茄点云器官分割

CHECKPOINT_PATH = "../checkpoints_20251222_140253_b+a1+j2_60+15+15/best_model.pth"
INPUT_PLY_PATH   = "../../materials/测试点云10.ply"
OUTPUT_PLY_PATH  = "../测试结果10.ply"

def read_ply_xyzrgb(file_path):
    ply = PlyData.read(file_path)
    v = ply['vertex'].data
    xyz = np.vstack([v['x'], v['y'], v['z']]).T.astype(np.float32)

    rgb = None
    for rname in ('red', 'r', 'Red'):
        if rname in v.dtype.names:
            red_name = rname
            break
    else:
        red_name = None

    if red_name is not None and 'green' in v.dtype.names and 'blue' in v.dtype.names:
        rgb = np.vstack([v[red_name], v['green'], v['blue']]).T.astype(np.float32) / 255.0
    else:
        rgb = np.ones_like(xyz) * 0.5

    return xyz, rgb

def save_ply_with_labels(xyz, rgb, labels, save_path):
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    vertex = np.array(
        [
            (
                xyz[i, 0], xyz[i, 1], xyz[i, 2],
                rgb_uint8[i, 0], rgb_uint8[i, 1], rgb_uint8[i, 2],
                int(labels[i] + 1)  # 0→1 (茎), 1→2 (叶)
            )
            for i in range(xyz.shape[0])
        ],
        dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('label', 'u1')
        ]
    )
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=True).write(save_path)

def main():
    os.makedirs(os.path.dirname(OUTPUT_PLY_PATH), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载 KPConvTomato 模型（num_classes 应与训练时一致）
    # 设为 num_classes=2（茎/叶），若实际 checkpoint 是 3 类需调整
    model = KPConvTomato(d_in=6, num_classes=2)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded KPConvTomato model from {CHECKPOINT_PATH}")

    # 读取点云
    xyz_orig, rgb_orig = read_ply_xyzrgb(INPUT_PLY_PATH)
    N_orig = xyz_orig.shape[0]
    print(f"Loaded {N_orig} points from {INPUT_PLY_PATH}")

    # 中心化（与训练时一致）
    centroid = np.mean(xyz_orig, axis=0, keepdims=True)
    xyz_centered = xyz_orig - centroid  # 不改变原始坐标存储，仅用于推理
    features = np.concatenate([xyz_centered, rgb_orig], axis=1)  # [N, 6]

    # KPConv 接收变长点云，直接整片推理（小点云可一次性；大点云需分块+合并）
    # 假设点云 < 100k，直接推理；否则需分块
    max_points = 5000  # 保守值（根据内存调整）
    if N_orig > max_points:
        print("⚠️ Large point cloud — splitting into chunks for inference...")
        all_preds = np.zeros(N_orig, dtype=np.int64)
        chunk_size = max_points
        for start in range(0, N_orig, chunk_size):
            end = min(start + chunk_size, N_orig)
            xyz_chunk = xyz_centered[start:end]
            feat_chunk = features[start:end]  # already centered + rgb

            preds_chunk = _infer_single_chunk(model, xyz_chunk, feat_chunk, device)
            all_preds[start:end] = preds_chunk
    else:
        all_preds = _infer_single_chunk(model, xyz_centered, features, device)

    # 保存结果（用原始坐标 xyz_orig）
    save_ply_with_labels(xyz_orig, rgb_orig, all_preds, OUTPUT_PLY_PATH)
    print(f"✅ Segmentation completed! Result saved to: {OUTPUT_PLY_PATH}")

# 对单个点云块进行 KPConv 推理，并插值回原始点
def _infer_single_chunk(model, xyz, features, device):
    N = xyz.shape[0]
    # 构造 batch dict（模拟 dataloader collate）
    batch = {
        'points': [torch.from_numpy(xyz).float().to(device)],
        'features': [torch.from_numpy(features).float().to(device)],
        'labels': [torch.zeros(N)],
        'lengths': torch.tensor([N], dtype=torch.long)
    }

    with torch.no_grad():
        model.eval()
        logits, pred_pts = model(batch)

    if pred_pts.shape[0] != N:
        print(f"⚠️ Warning: output points ({pred_pts.shape[0]}) ≠ input ({N}) — using NN interpolation")
        # 插值回原始点（保险起见）
        pred_pts_np = pred_pts.cpu().numpy()
        logits_np = logits.cpu().numpy()
        # 计算每个原始点最近的预测点
        from scipy.spatial import cKDTree
        tree = cKDTree(pred_pts_np)
        _, idx = tree.query(xyz, k=1)
        pred_labels = np.argmax(logits_np[idx], axis=1)
    else:
        pred_labels = torch.argmax(logits, dim=-1).cpu().numpy()

    return pred_labels.astype(np.int64)


if __name__ == '__main__':
    main()