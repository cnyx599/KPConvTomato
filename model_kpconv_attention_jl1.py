# model_kpconv_attention_jl1.py
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict

# 基于改进kpconv+注意力机制+几何先验的神经网络

# 体素网格下采样
def batch_grid_subsampling(points, features, voxel_size):
    # .detach() 断开梯度，避免 numpy() 报错
    pts_np = points.detach().cpu().numpy()
    voxel_grid = np.floor(pts_np / voxel_size).astype(np.int32)
    _, unique_idx = np.unique(voxel_grid, axis=0, return_index=True)
    keep = np.sort(unique_idx)
    return points[keep], features[keep]  # 仍返回原始张量（保留梯度）

# 边界感知注意力模块 + 几何先验
class BoundaryAwareAttention(nn.Module):
    def __init__(self, k=16, radius_factor=1.2, stem_enhance=True, leaf_edge_protect=True):
        super().__init__()
        self.k = k
        self.radius_factor = radius_factor
        self.stem_enhance = stem_enhance
        self.leaf_edge_protect = leaf_edge_protect  # 新增开关

        # 茎参数 (保持不变)
        self.stem_radius_mm = 7.0
        self.stem_axis_weight = 0.6

        # 叶边缘参数 (新增)
        self.leaf_edge_density_threshold = 0.3
        self.leaf_edge_weight = 2.0

    # ========== 轻量级茎骨架约束 ==========
    def _compute_stem_prior(self, points, labels):
        stem_mask = (labels == 0)  # 茎标签=0
        num_stem = stem_mask.sum().item()

        if num_stem < 10:
            return torch.zeros_like(labels, dtype=torch.float32, device=points.device)

        stem_pts = points[stem_mask]
        centroid = stem_pts.mean(dim=0, keepdim=True)
        centered_pts = stem_pts - centroid

        cov = centered_pts.T @ centered_pts
        _, _, V = torch.svd(cov, some=False)
        main_axis = V[:, 0]

        vecs = centered_pts
        proj_lengths = torch.abs(vecs @ main_axis)
        perp_vecs = vecs - proj_lengths.unsqueeze(1) * main_axis.unsqueeze(0)
        perp_dists = torch.norm(perp_vecs, dim=1)

        norm_radius = self.stem_radius_mm / 1000.0
        norm_dists = perp_dists / norm_radius

        stem_scores = torch.sigmoid((norm_dists - 0.7) * 5.0)

        full_stem_scores = torch.zeros(points.shape[0], device=points.device)
        full_stem_scores[stem_mask] = stem_scores

        return full_stem_scores

    # ========== 叶边缘连续性保护 ==========
    def _compute_leaf_edge_prior(self, points, labels):
        device = points.device
        leaf_mask = (labels == 1)  # 叶标签=1
        num_leaf = leaf_mask.sum().item()

        if num_leaf < 30:
            return torch.ones(points.shape[0], device=device)

        leaf_pts = points[leaf_mask]
        centroid = leaf_pts.mean(dim=0, keepdim=True)
        centered = leaf_pts - centroid

        # 估计叶片平面
        cov = centered.T @ centered
        _, _, V = torch.svd(cov)
        normal = V[:, 2]  # 平面法向

        # 2D投影
        u = V[:, 0]
        v = V[:, 1]
        coords2d = torch.stack([
            centered @ u,
            centered @ v
        ], dim=1)

        # 识别边缘点 (外圈15%)
        dists_2d = torch.norm(coords2d, dim=1)
        edge_threshold = torch.quantile(dists_2d, 0.85)
        edge_mask = dists_2d > edge_threshold

        if not edge_mask.any():
            return torch.ones(points.shape[0], device=device)

        # 计算边缘密度 (分块避免内存溢出)
        edge_pts = leaf_pts[edge_mask]
        M = edge_pts.shape[0]
        k = min(6, M)

        densities = []
        chunk_size = 512
        for i in range(0, M, chunk_size):
            chunk = edge_pts[i:i + chunk_size]
            dists = torch.cdist(chunk, edge_pts)
            _, knn_idx = torch.topk(dists, k, largest=False, dim=1)
            knn_dists = torch.gather(dists, 1, knn_idx)
            density = 1.0 / (knn_dists.mean(dim=1) + 1e-6)
            densities.append(density)

        densities = torch.cat(densities, dim=0)
        low_density_mask = densities < torch.quantile(densities, self.leaf_edge_density_threshold)

        # 构建权重
        weights = torch.ones(points.shape[0], device=device)
        edge_indices = torch.where(leaf_mask)[0][edge_mask]
        weights[edge_indices[low_density_mask]] = self.leaf_edge_weight

        return weights

    def forward(self, points, labels=None):
        N = points.shape[0]
        if N == 0:
            return torch.ones(0, 1, device=points.device)

        # === 基础边界计算 ===
        chunk_size = min(1024, N)
        dists_list = []
        for i in range(0, N, chunk_size):
            chunk = points[i:i + chunk_size]
            dist_chunk = torch.cdist(chunk, points)
            dists_list.append(dist_chunk)
        dists = torch.cat(dists_list, dim=0)

        _, knn_idx = torch.topk(dists, k=self.k, dim=1, largest=False)
        knn_dists = torch.gather(dists, 1, knn_idx)
        adaptive_radius = knn_dists[:, -1:] * self.radius_factor

        # 计算法向量
        knn_pts = points[knn_idx]
        centroids = knn_pts.mean(dim=1, keepdim=True)
        cov = (knn_pts - centroids).transpose(1, 2) @ (knn_pts - centroids)
        eigvals, _ = torch.linalg.eigh(cov)
        normals = eigvals[:, 0] / (eigvals.sum(dim=1) + 1e-8)

        # === 边界分数计算 ===
        if self.training and labels is not None:
            label_i = labels.unsqueeze(1).expand(-1, self.k)
            neighbor_labels = labels[knn_idx]
            diff = (label_i != neighbor_labels).float().sum(dim=1)
            boundary_score = diff / self.k

            curvature_score = 1.0 - normals
            boundary_score = boundary_score * (0.5 + 0.5 * curvature_score)

            # 茎结构先验注入
            if self.stem_enhance and labels is not None:
                stem_prior = self._compute_stem_prior(points, labels)
                stem_mask = (labels == 0)
                boundary_score = torch.where(
                    stem_mask,
                    torch.max(boundary_score, stem_prior * self.stem_axis_weight),
                    boundary_score
                )
        else:
            density = (dists < adaptive_radius).sum(dim=1).float()
            density_dev = torch.abs(density - density.mean()) / (density.std() + 1e-6)
            curvature_score = 1.0 - normals
            boundary_score = torch.sigmoid(density_dev) * torch.sigmoid(curvature_score * 3.0)

        # 映射到 [0.8, 1.5]
        attention = 0.8 + 0.7 * boundary_score
        attention = attention.unsqueeze(1)

        # ===== 叶边缘连续性保护 (叠加在茎增强后) =====
        if self.training and labels is not None and self.leaf_edge_protect:
            leaf_edge_weights = self._compute_leaf_edge_prior(points, labels)
            attention = attention.squeeze(1) * leaf_edge_weights
            attention = attention.unsqueeze(1)

        return attention


class KPConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=15, radius=0.1,
                 sigma=0.03, deformable=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.radius = radius
        self.sigma = sigma
        self.deformable = deformable

        # 初始化 kernel points（center模式）
        kernel_points = np.zeros((kernel_size, 3), dtype=np.float32)
        if kernel_size > 1:
            kernel_points[1:] = np.random.randn(kernel_size - 1, 3) * radius / 2
        self.register_buffer('kernel_points', torch.from_numpy(kernel_points))

        self.weights = nn.Parameter(torch.randn(kernel_size, in_channels, out_channels) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_channels))

        if deformable:
            self.offset_conv = nn.Sequential(
                nn.Linear(in_channels, 32),
                nn.ReLU(),
                nn.Linear(32, kernel_size * 3)
            )
        else:
            self.offset_conv = None

    def forward(self, query_pts, support_pts, query_feat, support_feat):
        Nq, Ns = query_pts.shape[0], support_pts.shape[0]
        K = self.kernel_size

        # 计算距离矩阵（小规模OK）
        dists = torch.cdist(query_pts, support_pts)  # [Nq, Ns]
        knn_dists, knn_inds = torch.topk(dists, K, dim=1, largest=False)

        # 排除超半径点
        valid_mask = knn_dists < self.radius

        # 获取邻居
        neighbor_coords = support_pts[knn_inds]  # [Nq, K, 3]
        neighbor_feats = support_feat[knn_inds]  # [Nq, K, C_in]

        # 局部坐标系
        local_coords = neighbor_coords - query_pts.unsqueeze(1)  # [Nq, K, 3]

        # 可变形偏移
        if self.deformable and self.offset_conv is not None:
            offsets = self.offset_conv(query_feat)  # [Nq, K*3]
            offsets = offsets.view(Nq, K, 3)
            local_coords = local_coords + offsets

        # 计算 kernel 权重
        diffs = local_coords.unsqueeze(2) - self.kernel_points  # [Nq, K, 1, 3] - [1, 1, Kp, 3]
        sq_dists = torch.sum(diffs ** 2, dim=3)  # [Nq, K, Kp]
        kernel_weights = torch.exp(-sq_dists / (2 * self.sigma ** 2))

        # 加权聚合
        weighted = torch.einsum('qkp,kio->qkio', kernel_weights, self.weights)
        output = torch.einsum('qkc,qkco->qo', neighbor_feats, weighted)
        return output + self.bias


class KPConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, radius=0.1, deformable=False):
        super().__init__()
        self.kpconv = KPConv(in_dim, out_dim, radius=radius, deformable=deformable)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, query_pts, support_pts, query_feat, support_feat):
        # 显式参数名，支持关键字调用
        out = self.kpconv(query_pts, support_pts, query_feat, support_feat)
        out = self.bn(out)
        return self.act(out)


class KPConvUNet(nn.Module):
    def __init__(self, in_features_dim=6, num_classes=3):
        super().__init__()
        self.num_classes = num_classes

        # 编码器（轻量：3层足够小数据）
        self.enc1 = KPConvBlock(in_features_dim, 32, radius=0.02, deformable=True)
        self.enc2 = KPConvBlock(32, 64, radius=0.04, deformable=False)
        self.enc3 = KPConvBlock(64, 128, radius=0.08, deformable=False)

        # 瓶颈
        self.bottleneck = KPConvBlock(128, 128, radius=0.16)

        # 解码器
        self.dec2 = KPConvBlock(128 + 64, 64, radius=0.08)
        self.dec1 = KPConvBlock(64 + 32, 32, radius=0.04)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

        self.voxel_sizes = [0.01, 0.02, 0.04]

    def forward(self, batch):
        device = batch['points'][0].device
        B = len(batch['points'])

        all_points = torch.cat(batch['points'], dim=0)  # [N_total, 3]
        all_features = torch.cat(batch['features'], dim=0)  # [N_total, 6]
        lengths = batch['lengths']

        # 初始化
        current_pts = all_points
        current_feat = all_features
        skip_data = []

        # 编码器
        for i, enc in enumerate([self.enc1, self.enc2, self.enc3]):
            # 同时下采样 points & features
            subsampled_pts, subsampled_feat = batch_grid_subsampling(
                current_pts, current_feat, self.voxel_sizes[i]
            )

            # KPConv：新点查旧点
            new_feat = enc(
                query_pts=subsampled_pts,
                support_pts=current_pts,
                query_feat=subsampled_feat,  # 用新点对应特征
                support_feat=current_feat
            )

            skip_data.append((current_pts, current_feat))
            current_pts, current_feat = subsampled_pts, new_feat

        # 瓶颈
        bottleneck_out = self.bottleneck(
            current_pts, current_pts, current_feat, current_feat
        )

        # 解码器
        x = bottleneck_out
        for i, dec in enumerate([self.dec2, self.dec1]):
            skip_pts, skip_feat = skip_data[-(i + 1)]

            # 上采样
            dists = torch.cdist(skip_pts, current_pts)
            _, nn_idx = torch.min(dists, dim=1)
            upsampled = x[nn_idx]

            # 拼接
            fused = torch.cat([upsampled, skip_feat], dim=1)
            x = dec(skip_pts, skip_pts, fused, fused)
            current_pts = skip_pts  # 每次更新 current_pts 为当前层点坐标

        logits = self.classifier(x)
        return logits, current_pts  # 同时返回 logits 和对应点坐标


class KPConvTomato(nn.Module):
    def __init__(self, d_in=6, num_classes=2):
        super().__init__()
        self.backbone = KPConvUNet(in_features_dim=d_in, num_classes=num_classes)
        # 模型初始化
        self.boundary_attention = BoundaryAwareAttention(
            k=16,
            radius_factor=1.2,
            stem_enhance=True,  # 开启 (解决茎断裂)
            leaf_edge_protect=True  # 开启 (解决叶边缘缺失)
        )

    def forward(self, batch):
        device = batch['points'][0].device

        # Backbone forward
        logits_lowres, pts_lowres = self.backbone(batch)  # 低分辨率输出

        # 插值回原始点（与 loss 计算一致）
        orig_pts = torch.cat(batch['points'], dim=0).to(device)  # [N_orig, 3]
        dists = torch.cdist(orig_pts, pts_lowres)
        _, nn_idx = torch.min(dists, dim=1)
        logits_interp = logits_lowres[nn_idx]  # [N_orig, C]

        # 在原始点上计算边界注意力（尺寸对齐）
        if self.training and 'labels' in batch:
            labels_flat = torch.cat(batch['labels'], dim=0).to(device)  # [N_orig]
            attention = self.boundary_attention(orig_pts, labels_flat)
        else:
            attention = self.boundary_attention(orig_pts)  # 无监督模式

        # 调制插值后的 logits
        logits_final = logits_interp * attention  # [N_orig, C] * [N_orig, 1]

        return logits_final, orig_pts  # 返回与输入对齐的 logits 和点

    @staticmethod
    def collate_fn(batch):
        points = [item['points'] for item in batch]
        features = [item['features'] for item in batch]
        labels = [item['labels'] for item in batch]
        lengths = torch.tensor([p.shape[0] for p in points], dtype=torch.long)
        return {
            'points': points,
            'features': features,
            'labels': labels,
            'lengths': lengths
        }

