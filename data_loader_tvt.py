# data_loader_tvt.py
import os
import numpy as np
from plyfile import PlyData
import torch
from torch.utils.data import Dataset

# 基于kpconv的预处理

class TomatoDataset(Dataset):
    def __init__(self, root_dir, num_points=8192, split='train', augment=True):
        # 支持 'test' 分割（兼容原有 train/val）
        if split == 'test':
            self.root_dir = os.path.join(root_dir, 'test')
        else:
            self.root_dir = os.path.join(root_dir, split)
        self.num_points = num_points
        self.augment = augment
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"{self.root_dir} does not exist")
        self.files = [os.path.join(self.root_dir, f)
                      for f in os.listdir(self.root_dir) if f.endswith('.ply')]
        self.files.sort()
        print(f"[TomatoDataset] Loaded {len(self.files)} files from {self.root_dir}")

    def __len__(self):
        return len(self.files)

    def _read_ply(self, file_path):
        ply = PlyData.read(file_path)
        v = ply['vertex'].data

        # 坐标
        try:
            x = np.vstack([v['x'], v['y'], v['z']]).T
        except Exception as e:
            raise RuntimeError(f"Cannot read x,y,z from {file_path}: {e}")

        # 颜色
        rgb = None
        for rname in ('red', 'r', 'Red'):
            if rname in v.dtype.names:
                red_name = rname
                break
        else:
            red_name = None

        if red_name is not None and 'green' in v.dtype.names and 'blue' in v.dtype.names:
            rgb = np.vstack([v[red_name], v['green'], v['blue']]).T / 255.0
        else:
            rgb = np.ones_like(x) * 0.5

        # label
        label = None
        for cand in ('sf', 'scalar_sf', 'label', 'scalar_label'):
            if cand in v.dtype.names:
                label = np.array(v[cand]).astype(np.int64)
                break
        if label is None:
            raise RuntimeError(f"No label field found in {file_path}.")

        # 映射：1→0（茎），2→1（叶）
        label = np.where(label == 1, 0, label)  # 先把 1 → 0
        label = np.where(label == 2, 1, label)  # 再把 2 → 1

        unique_labels = np.unique(label)
        valid_labels = {-1, 0, 1, 2}
        if not set(unique_labels).issubset(valid_labels):
            raise ValueError(f"Unexpected labels {unique_labels} in {file_path}.")

        return x.astype(np.float32), rgb.astype(np.float32), label.astype(np.int64)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        x, rgb, label = self._read_ply(file_path)

        N = x.shape[0]
        if N >= self.num_points:
            choice = np.random.choice(N, self.num_points, replace=False)
        else:
            choice = np.random.choice(N, self.num_points, replace=True)

        pts = x[choice]
        cols = rgb[choice]
        lbs = label[choice]

        # 中心化（必须）
        pts_centered = pts - np.mean(pts, axis=0, keepdims=True)

        # 茎缺失增强（仅训练）
        if self.augment and np.random.rand() < 0.3:
            stem_mask = (lbs == 0)
            stem_indices = np.where(stem_mask)[0]
            if len(stem_indices) > 0:
                drop_ratio = np.random.uniform(0.2, 0.8)
                n_drop = int(len(stem_indices) * drop_ratio)
                if n_drop > 0:
                    drop_idx = np.random.choice(stem_indices, n_drop, replace=False)
                    keep_mask = np.ones(len(lbs), dtype=bool)
                    keep_mask[drop_idx] = False
                    pts_centered = pts_centered[keep_mask]
                    cols = cols[keep_mask]
                    lbs = lbs[keep_mask]

        # 重采样保证 num_points
        N_current = pts_centered.shape[0]
        if N_current < self.num_points:
            choice = np.random.choice(N_current, self.num_points, replace=True)
        else:
            choice = np.random.choice(N_current, self.num_points, replace=False)

        pts_centered = pts_centered[choice]
        cols = cols[choice]
        lbs = lbs[choice]

        # 旋转增强（仅训练）
        if self.augment:
            theta = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0],
                          [s, c, 0],
                          [0, 0, 1]], dtype=np.float32)
            pts_centered = pts_centered @ R.T

        # features = XYZ + RGB (6D)
        features_6d = np.concatenate([pts_centered, cols], axis=1)  # [N, 6]

        return {
            'points': torch.from_numpy(pts_centered).float(),  # [N, 3]
            'features': torch.from_numpy(features_6d).float(),  # [N, 6]
            'labels': torch.from_numpy(lbs).long()
        }

