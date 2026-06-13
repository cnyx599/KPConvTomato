# model.py

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict

# 体素网格下采样
def batch_grid_subsampling(points, features, voxel_size):
    pts_np = points.detach().cpu().numpy()
    voxel_grid = np.floor(pts_np / voxel_size).astype(np.int32)
    _, unique_idx = np.unique(voxel_grid, axis=0, return_index=True)
    keep = np.sort(unique_idx)
    return points[keep], features[keep]

# 几何先验引导的边界感知注意力模块
class BoundaryAwareAttention(nn.Module):
    def __init__(self, k=16, radius_factor=1.2, stem_enhance=True, leaf_edge_protect=True, stem_lambda=0.6):
        super().__init__()
        self.k = k
        self.radius_factor = radius_factor
        self.stem_enhance = stem_enhance
        self.leaf_edge_protect = leaf_edge_protect
        self.stem_lambda = stem_lambda 

    # 动态茎骨架先验
    def _compute_stem_prior(self, points, labels):
        stem_mask = (labels == 0)
        num_stem = stem_mask.sum().item()
        if num_stem < 10:
            return torch.zeros_like(labels, dtype=torch.float32, device=points.device)
        
        stem_pts = points[stem_mask]
        dists = torch.cdist(stem_pts, stem_pts)
        dists.fill_diagonal_(1e6)
        N = stem_pts.shape[0]
        if N > 1000:
            idx = torch.randperm(N)[:1000]
            dists = dists[idx]
        knn_dists = torch.topk(dists, k=6, largest=False, dim=1).values
        avg_densities = 1.0 / (knn_dists.mean(dim=1) + 1e-6)
        mean_density = avg_densities.mean()
        estimated_radius = torch.clamp(0.02 / (mean_density + 1e-6), min=0.003, max=0.015).item()

        centroid = stem_pts.mean(dim=0, keepdim=True)
        centered_pts = stem_pts - centroid
        cov = centered_pts.T @ centered_pts
        _, _, V = torch.svd(cov, some=False)
        main_axis = V[:, 0]
        
        vecs = centered_pts
        proj_lengths = torch.abs(vecs @ main_axis)
        perp_vecs = vecs - proj_lengths.unsqueeze(1) * main_axis.unsqueeze(0)
        perp_dists = torch.norm(perp_vecs, dim=1)
        norm_dists = perp_dists / estimated_radius
        
        stem_scores = torch.sigmoid((norm_dists - 0.7) * 5.0)
        full_stem_scores = torch.zeros(points.shape[0], device=points.device)
        full_stem_scores[stem_mask] = stem_scores
        return full_stem_scores

    # 叶边缘连续性保护
    def _compute_leaf_edge_prior(self, points, labels):
        device = points.device
        leaf_mask = (labels == 1)
        num_leaf = leaf_mask.sum().item()
        if num_leaf < 30:
            return torch.ones(points.shape[0], device=device)
        
        leaf_pts = points[leaf_mask]
        centroid = leaf_pts.mean(dim=0, keepdim=True)
        centered = leaf_pts - centroid
        cov = centered.T @ centered
        _, _, V = torch.svd(cov)
        
        # 投影到主平面
        u, v = V[:, 0], V[:, 1]
        coords2d = torch.stack([centered @ u, centered @ v], dim=1)
        dists_2d = torch.norm(coords2d, dim=1)
        
        # 识别外环 15% 的点作为边缘候选集
        edge_threshold = torch.quantile(dists_2d, 0.85)
        edge_mask = dists_2d > edge_threshold
        if not edge_mask.any():
            return torch.ones(points.shape[0], device=device)
        
        edge_pts = leaf_pts[edge_mask]
        M = edge_pts.shape[0]
        
        # 逐点计算边缘密度
        k = min(8, M)
        densities = []
        chunk_size = 512
        for i in range(0, M, chunk_size):
            chunk = edge_pts[i:i + chunk_size]
            dists = torch.cdist(chunk, edge_pts)
            _, knn_idx = torch.topk(dists, k, largest=False, dim=1)
            knn_dists = torch.gather(dists, 1, knn_idx)
            rho_i = 1.0 / (knn_dists.mean(dim=1) + 1e-6)
            densities.append(rho_i)
        densities = torch.cat(densities, dim=0)
        
        rho_max = densities.max()
        
        edge_weights_raw = 2.0 + 1.5 * (1.0 - densities / (rho_max + 1e-6))
        edge_weights_clamped = torch.clamp(edge_weights_raw, min=1.5, max=3.5)
        
        # 将权重映射回全量点云
        weights = torch.ones(points.shape[0], device=device)
        edge_indices = torch.where(leaf_mask)[0][edge_mask]
        weights[edge_indices] = edge_weights_clamped
        
        return weights

    def forward(self, points, labels=None):
        N = points.shape[0]
        if N == 0:
            return torch.ones(0, 1, device=points.device)
        
        # === 基础边界响应计算 ===
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
        knn_pts = points[knn_idx]
        
        centroids = knn_pts.mean(dim=1, keepdim=True)
        cov = (knn_pts - centroids).transpose(1, 2) @ (knn_pts - centroids)
        eigvals, _ = torch.linalg.eigh(cov)
        normals = eigvals[:, 0] / (eigvals.sum(dim=1) + 1e-8)
        
        if self.training and labels is not None:
            label_i = labels.unsqueeze(1).expand(-1, self.k)
            neighbor_labels = labels[knn_idx]
            diff = (label_i != neighbor_labels).float().sum(dim=1)
            boundary_score = diff / self.k
            curvature_score = 1.0 - normals
            boundary_score = boundary_score * (0.5 + 0.5 * curvature_score)
            
            # 融合茎先验
            if self.stem_enhance:
                stem_prior = self._compute_stem_prior(points, labels)
                stem_mask = (labels == 0)
                boundary_score = torch.where(stem_mask, torch.max(boundary_score, stem_prior * self.stem_lambda), boundary_score)
        else:
            density = (dists < adaptive_radius).sum(dim=1).float()
            density_dev = torch.abs(density - density.mean()) / (density.std() + 1e-6)
            curvature_score = 1.0 - normals
            boundary_score = torch.sigmoid(density_dev) * torch.sigmoid(curvature_score * 3.0)
            
        # 注意力权重融合
        attention = 0.8 + 0.7 * boundary_score
        attention = attention.unsqueeze(1)
        
        # 叶边缘连续性保护融合
        if self.training and labels is not None and self.leaf_edge_protect:
            leaf_edge_weights = self._compute_leaf_edge_prior(points, labels)
            attention = attention.squeeze(1) * leaf_edge_weights
            attention = attention.unsqueeze(1)
            
        return attention

# KPConv 核心组件
class KPConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=15, radius=0.1, sigma=0.03, deformable=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.radius = radius
        self.sigma = sigma
        self.deformable = deformable
        
        kernel_points = np.zeros((kernel_size, 3), dtype=np.float32)
        if kernel_size > 1:
            kernel_points[1:] = np.random.randn(kernel_size - 1, 3) * radius / 2
        self.register_buffer('kernel_points', torch.from_numpy(kernel_points))
        
        self.weights = nn.Parameter(torch.randn(kernel_size, in_channels, out_channels) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        if deformable:
            self.offset_conv = nn.Sequential(nn.Linear(in_channels, 32), nn.ReLU(), nn.Linear(32, kernel_size * 3))
        else:
            self.offset_conv = None

    def forward(self, query_pts, support_pts, query_feat, support_feat):
        Nq, Ns = query_pts.shape[0], support_pts.shape[0]
        K = self.kernel_size
        dists = torch.cdist(query_pts, support_pts)
        knn_dists, knn_inds = torch.topk(dists, K, dim=1, largest=False)
        valid_mask = knn_dists < self.radius
        
        neighbor_coords = support_pts[knn_inds]
        neighbor_feats = support_feat[knn_inds]
        local_coords = neighbor_coords - query_pts.unsqueeze(1)
        
        if self.deformable and self.offset_conv is not None:
            offsets = self.offset_conv(query_feat)
            offsets = offsets.view(Nq, K, 3)
            local_coords = local_coords + offsets
            
        diffs = local_coords.unsqueeze(2) - self.kernel_points
        sq_dists = torch.sum(diffs ** 2, dim=3)
        kernel_weights = torch.exp(-sq_dists / (2 * self.sigma ** 2))
        
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
        out = self.kpconv(query_pts, support_pts, query_feat, support_feat)
        out = self.bn(out)
        return self.act(out)

# KPConvTomato 整体架构
class KPConvUNet(nn.Module):
    def __init__(self, in_features_dim=6, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        
        # 编码器
        self.enc1 = KPConvBlock(in_features_dim, 32, radius=0.015, deformable=True) 
        self.enc2 = KPConvBlock(32, 64, radius=0.03, deformable=False)
        self.enc3 = KPConvBlock(64, 128, radius=0.06, deformable=False)
        self.enc4 = KPConvBlock(128, 256, radius=0.12, deformable=False) 
        
        # 瓶颈层
        self.bottleneck = KPConvBlock(256, 256, radius=0.24)
        
        # 解码器
        self.dec3 = KPConvBlock(256 + 128, 128, radius=0.12)
        self.dec2 = KPConvBlock(128 + 64, 64, radius=0.06)
        self.dec1 = KPConvBlock(64 + 32, 32, radius=0.03) 
        
        # 分类头
        self.classifier = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes))
        
        # 体素下采样尺寸
        self.voxel_sizes = [0.015, 0.03, 0.06, 0.12] 

    def forward(self, batch):
        device = batch['points'][0].device
        all_points = torch.cat(batch['points'], dim=0)
        all_features = torch.cat(batch['features'], dim=0)
        
        current_pts = all_points
        current_feat = all_features
        skip_data = []
        
        for i, enc in enumerate([self.enc1, self.enc2, self.enc3, self.enc4]):
            subsampled_pts, subsampled_feat = batch_grid_subsampling(current_pts, current_feat, self.voxel_sizes[i])
            new_feat = enc(query_pts=subsampled_pts, support_pts=current_pts, query_feat=subsampled_feat, support_feat=current_feat)
            skip_data.append((current_pts, current_feat)) 
            current_pts, current_feat = subsampled_pts, new_feat
            
        bottleneck_out = self.bottleneck(current_pts, current_pts, current_feat, current_feat)
        x = bottleneck_out
        
        for i, dec in enumerate([self.dec3, self.dec2, self.dec1]):
            skip_pts, skip_feat = skip_data[-(i + 1)]
            dists = torch.cdist(skip_pts, current_pts)
            _, nn_idx = torch.min(dists, dim=1)
            upsampled = x[nn_idx]
            fused = torch.cat([upsampled, skip_feat], dim=1)
            x = dec(skip_pts, skip_pts, fused, fused)
            current_pts = skip_pts
            
        logits = self.classifier(x)
        return logits, current_pts

class KPConvTomato(nn.Module):
    def __init__(self, d_in=6, num_classes=2, stem_lambda=0.6):
        super().__init__()
        self.backbone = KPConvUNet(in_features_dim=d_in, num_classes=num_classes)
        self.boundary_attention = BoundaryAwareAttention(k=16, radius_factor=1.2, stem_enhance=True, leaf_edge_protect=True, stem_lambda=stem_lambda)

    def forward(self, batch):
        device = batch['points'][0].device
        logits_lowres, pts_lowres = self.backbone(batch) 
        
        orig_pts = torch.cat(batch['points'], dim=0).to(device)
        dists = torch.cdist(orig_pts, pts_lowres)
        _, nn_idx = torch.min(dists, dim=1)
        logits_interp = logits_lowres[nn_idx]
        
        if self.training and 'labels' in batch:
            labels_flat = torch.cat(batch['labels'], dim=0).to(device)
            attention = self.boundary_attention(orig_pts, labels_flat)
        else:
            attention = self.boundary_attention(orig_pts) 
        
        logits_final = logits_interp * attention
        return logits_final, orig_pts  

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

