# data_loader.py 数据预处理

import os
import numpy as np
from plyfile import PlyData
import torch
from torch.utils.data import Dataset

class TomatoDataset(Dataset):
    def __init__(self, root_dir, num_points=8192, split='train', augment=True):
        if split == 'test':
            self.root_dir = os.path.join(root_dir, 'test')
        else:
            self.root_dir = os.path.join(root_dir, split)
        self.num_points = num_points
        self.augment = augment

        self.stem_ratio = 0.3
        self.min_stem_points = int(num_points * self.stem_ratio) 

        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"{self.root_dir} does not exist")
        self.files = [os.path.join(self.root_dir, f)
                      for f in os.listdir(self.root_dir) if f.endswith('.ply')]
        self.files.sort()
        print(f"[TomatoDataset] Loaded {len(self.files)} files from {self.root_dir}")
        print(
            f"[TomatoDataset] Stratified Sampling: stem_ratio={self.stem_ratio}, min_stem_points={self.min_stem_points}")

    def __len__(self):
        return len(self.files)

    def _read_ply(self, file_path):
        ply = PlyData.read(file_path)
        v = ply['vertex'].data
        try:
            x = np.vstack([v['x'], v['y'], v['z']]).T
        except Exception as e:
            raise RuntimeError(f"Cannot read x,y,z from {file_path}: {e}")

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

        label = None
        for cand in ('sf', 'scalar_sf', 'label', 'scalar_label'):
            if cand in v.dtype.names:
                label = np.array(v[cand]).astype(np.int64)
                break
        if label is None:
            raise RuntimeError(f"No label field found in {file_path}.")

        label = np.where(label == 1, 0, label)        # 映射：1→0（茎），2→1（叶）
        label = np.where(label == 2, 1, label)

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

        # 中心化
        pts_centered = pts - np.mean(pts, axis=0, keepdims=True)

        # ================ 茎补全增强 ====================
        if self.augment and np.random.rand() < 0.5:
            stem_mask = (lbs == 0)
            stem_indices = np.where(stem_mask)[0]
            if len(stem_indices) > 0:
                stem_pts = pts_centered[stem_mask]
                stem_cols = cols[stem_mask]
                n_stem = len(stem_pts)
                n_add = n_stem 
                add_indices = np.random.choice(n_stem, n_add, replace=True)
                jitter_std = 0.015
                noise = np.random.normal(0, jitter_std, size=(n_add, 3)).astype(np.float32)
                new_stem_pts = stem_pts[add_indices] + noise
                new_stem_cols = stem_cols[add_indices]
                new_stem_lbs = np.zeros(n_add, dtype=np.int64)

                pts_centered = np.vstack([pts_centered, new_stem_pts])
                cols = np.vstack([cols, new_stem_cols])
                lbs = np.concatenate([lbs, new_stem_lbs])
        # ==========================================

        # =============== 分层采样 ===================
        N_current = pts_centered.shape[0]
        stem_mask = (lbs == 0)
        leaf_mask = (lbs == 1)

        stem_indices_all = np.where(stem_mask)[0]
        leaf_indices_all = np.where(leaf_mask)[0]

        n_stem_available = len(stem_indices_all)
        n_leaf_available = len(leaf_indices_all)

        n_stem_needed = min(self.min_stem_points, n_stem_available)
        n_leaf_needed = self.num_points - n_stem_needed

        if n_leaf_needed > n_leaf_available:
            n_leaf_needed = n_leaf_available
            n_stem_needed = self.num_points - n_leaf_needed

        if n_stem_needed < 5 and n_stem_available >= 5:
            n_stem_needed = min(5, n_stem_available)
            n_leaf_needed = self.num_points - n_stem_needed

        if n_stem_available > 0:
            if n_stem_available >= n_stem_needed:
                stem_sampled = np.random.choice(stem_indices_all, n_stem_needed, replace=False)
            else:
                stem_sampled = np.random.choice(stem_indices_all, n_stem_needed, replace=True)
        else:
            stem_sampled = np.array([], dtype=np.int64)

        if n_leaf_available > 0:
            if n_leaf_available >= n_leaf_needed:
                leaf_sampled = np.random.choice(leaf_indices_all, n_leaf_needed, replace=False)
            else:
                leaf_sampled = np.random.choice(leaf_indices_all, n_leaf_needed, replace=True)
        else:
            leaf_sampled = np.array([], dtype=np.int64)

        choice = np.concatenate([stem_sampled, leaf_sampled])

        np.random.shuffle(choice)
        # ==========================================

        pts_centered = pts_centered[choice]
        cols = cols[choice]
        lbs = lbs[choice]

        # 旋转增强
        if self.augment:
            theta = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0],
                          [s, c, 0],
                          [0, 0, 1]], dtype=np.float32)
            pts_centered = pts_centered @ R.T

        features_6d = np.concatenate([pts_centered, cols], axis=1)  # features = XYZ + RGB (6D)

        # 打印采样统计用于调试
        if self.augment and idx < 3: 
            stem_count = np.sum(lbs == 0)
            leaf_count = np.sum(lbs == 1)
            print(f"[Sample {idx}] Stem: {stem_count} ({stem_count / self.num_points * 100:.1f}%), "
                  f"Leaf: {leaf_count} ({leaf_count / self.num_points * 100:.1f}%)")

        return {
            'points': torch.from_numpy(pts_centered).float(),
            'features': torch.from_numpy(features_6d).float(),
            'labels': torch.from_numpy(lbs).long()
        }