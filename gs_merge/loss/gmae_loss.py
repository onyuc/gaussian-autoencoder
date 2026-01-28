"""
GMAE Loss: Distribution-based Loss for Gaussian Reconstruction

구성요소:
1. Density Field KL Divergence - 공간 분포 매칭
2. Random View Rendering Loss - 시각적 일관성  
3. Opacity Sparsity - 불필요한 Gaussian 억제
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


from fused_ssim import fused_ssim
from gsplat import rasterization


class GMAELoss(nn.Module):
    """
    Gaussian Merging AutoEncoder Loss
    
    복합 손실 함수:
    1. Density Field KL Divergence
    2. Random Rendering Loss (gsplat-based)
    3. Opacity Sparsity Regularization
    """
    
    def __init__(
        self, 
        lambda_density: float = 1.0,
        lambda_render: float = 1.0,
        lambda_sparsity: float = 0.05,
        n_density_samples: int = 1024,
        render_resolution: int = 64,
        debug_save_dir: str = "./debug_renders",
        debug_save_interval: int = 100,
        warmup_iterations: int = 200
    ):
        super().__init__()
        self.lambda_density = lambda_density
        self.lambda_render = lambda_render
        self.lambda_sparsity = lambda_sparsity
        self.n_samples = n_density_samples
        self.render_resolution = render_resolution
        self.debug_save_dir = debug_save_dir
        self.debug_save_interval = debug_save_interval
        self.warmup_iterations = warmup_iterations
        
        self._debug_count = 0
        self._render_call_count = 0
        self._render_save_count = 0
        self._iteration = 0

    def forward(
        self, 
        input_g: Dict[str, torch.Tensor],
        output_g: Dict[str, torch.Tensor], 
        input_mask: Optional[torch.Tensor] = None,
        output_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            input_g:  {'xyz', 'rotation', 'scale', 'opacity', ...} Input Gaussians
            output_g: {'xyz', 'rotation', 'scale', 'opacity', ...} Reconstructed
            input_mask: [B, N] Padding mask (True=Invalid)
            output_mask: [B, M] Padding mask (True=Invalid)
        
        Returns:
            total_loss: Scalar
            loss_dict: Individual components
        """
        # 1. Distribution Loss
        loss_density = self.compute_density_loss(input_g, output_g, input_mask, output_mask)
        
        # 2. Rendering Loss
        loss_render = self.compute_rendering_loss(input_g, output_g, input_mask, output_mask)
        
        # 3. Sparsity Loss
        loss_sparsity = self.compute_sparsity_loss(output_g, output_mask)
        
        # Iteration counter
        self._iteration += 1
        
        # Debug 출력
        self._debug_count += 1
        if self._debug_count % self.debug_save_interval == 0:
            warmup_status = f"WARMUP {self._iteration}/{self.warmup_iterations}" if self._iteration < self.warmup_iterations else "NORMAL"
            print(f"[{warmup_status}] Training mode")
            self._print_debug_stats(input_g, output_g, loss_density, loss_render)
        
        # Total Loss
        total_loss = (
            self.lambda_density * loss_density +
            self.lambda_render * loss_render +
            self.lambda_sparsity * loss_sparsity
        )
        
        return total_loss, {
            "loss_density": loss_density.item(),
            "loss_render": loss_render.item(),
            "loss_sparsity": loss_sparsity.item(),
            "loss_total": total_loss.item()
        }

    def _print_debug_stats(
        self, 
        input_g: Dict[str, torch.Tensor], 
        output_g: Dict[str, torch.Tensor],
        loss_density: torch.Tensor,
        loss_render: torch.Tensor
    ):
        """Debug 통계 출력 (opacity > 0.005 필터링)"""
        with torch.no_grad():
            # Raw data
            in_xyz = input_g['xyz']
            in_scale_real = torch.exp(input_g['scale'])
            in_opacity_real = torch.sigmoid(input_g['opacity'])
            
            out_xyz = output_g['xyz']
            out_scale_real = torch.exp(output_g['scale'])
            out_opacity_real = torch.sigmoid(output_g['opacity'])
            
            # Filter by opacity > 0.005
            in_valid_mask = in_opacity_real.squeeze(-1) > 0.005
            out_valid_mask = out_opacity_real.squeeze(-1) > 0.005
            
            # Count filtered
            in_total = in_xyz.numel() // 3
            out_total = out_xyz.numel() // 3
            in_removed = in_total - in_valid_mask.sum().item()
            out_removed = out_total - out_valid_mask.sum().item()
            
            # Apply filter
            in_xyz_filtered = in_xyz[in_valid_mask]
            in_scale_filtered = in_scale_real[in_valid_mask]
            in_opacity_filtered = in_opacity_real.squeeze(-1)[in_valid_mask]
            
            out_xyz_filtered = out_xyz[out_valid_mask]
            out_scale_filtered = out_scale_real[out_valid_mask]
            out_opacity_filtered = out_opacity_real.squeeze(-1)[out_valid_mask]
            
            print(f"\n{'='*60}")
            print(f"[DEBUG iter={self._debug_count}] Input vs Output 분포 (opacity > 0.005)")
            print(f"{'='*60}")
            print(f"  Filtered Gaussians:")
            print(f"    Total Input(B, N):  {in_total} → {in_total - in_removed} (removed: {in_removed}, {in_removed/max(in_total, 1)*100:.1f}%)")
            print(f"    Total Output(B, N): {out_total} → {out_total - out_removed} (removed: {out_removed}, {out_removed/max(out_total, 1)*100:.1f}%)")
            
            if in_xyz_filtered.numel() > 0 and out_xyz_filtered.numel() > 0:
                print(f"  XYZ:")
                print(f"    Input  - min: {in_xyz_filtered.min():.3f}, max: {in_xyz_filtered.max():.3f}, mean: {in_xyz_filtered.mean():.3f}")
                print(f"    Output - min: {out_xyz_filtered.min():.3f}, max: {out_xyz_filtered.max():.3f}, mean: {out_xyz_filtered.mean():.3f}")
                print(f"  Scale (실제값):")
                print(f"    Input  - min: {in_scale_filtered.min():.4f}, max: {in_scale_filtered.max():.4f}, mean: {in_scale_filtered.mean():.4f}")
                print(f"    Output - min: {out_scale_filtered.min():.4f}, max: {out_scale_filtered.max():.4f}, mean: {out_scale_filtered.mean():.4f}")
                print(f"  Opacity (0~1):")
                print(f"    Input  - min: {in_opacity_filtered.min():.3f}, max: {in_opacity_filtered.max():.3f}, mean: {in_opacity_filtered.mean():.3f}")
                print(f"    Output - min: {out_opacity_filtered.min():.3f}, max: {out_opacity_filtered.max():.3f}, mean: {out_opacity_filtered.mean():.3f}")
                print(f"  Opacity > 0.5 비율:")
                print(f"    Input:  {(in_opacity_filtered > 0.5).float().mean()*100:.1f}%")
                print(f"    Output: {(out_opacity_filtered > 0.5).float().mean()*100:.1f}%")
            else:
                print(f"  ⚠ No valid Gaussians after filtering!")
                
            print(f"  Losses: density={loss_density.item():.4f}, render={loss_render.item():.4f}")
            print(f"{'='*60}\n")

    # =========================================================================
    # [1] Density Field Matching (KLD)
    # =========================================================================
    def compute_density_loss(
        self,
        in_g: Dict[str, torch.Tensor],
        out_g: Dict[str, torch.Tensor],
        in_mask: Optional[torch.Tensor],
        out_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Importance Sampling + Uniform Sampling으로 밀도장 KL Divergence 계산"""
        B, N, _ = in_g['xyz'].shape
        device = in_g['xyz'].device
        
        n_near = int(self.n_samples * 0.7)
        n_uniform = self.n_samples - n_near
        
        # Near-Input Sampling (70%)
        rand_indices = torch.randint(0, N, (B, n_near), device=device)
        selected_xyz = self._batch_gather(in_g['xyz'], rand_indices)
        selected_scale = self._batch_gather(in_g['scale'], rand_indices)
        
        noise = torch.randn_like(selected_xyz)
        queries_near = selected_xyz + noise * selected_scale
        
        # Uniform Sampling (30%)
        queries_uniform = (torch.rand(B, n_uniform, 3, device=device) * 2.0) - 1.0
        
        query_points = torch.cat([queries_near, queries_uniform], dim=1)
        query_points = torch.clamp(query_points, -1.0, 1.0)
        
        # Density 계산
        d_in = self._get_density_field(query_points, in_g, in_mask)
        d_out = self._get_density_field(query_points, out_g, out_mask)
        
        # 안전한 정규화
        eps = 1e-8
        d_in = d_in + eps
        d_out = d_out + eps
        
        sum_in = torch.sum(d_in, dim=1, keepdim=True).clamp(min=eps)
        sum_out = torch.sum(d_out, dim=1, keepdim=True).clamp(min=eps)
        
        p_in = d_in / sum_in
        p_out = d_out / sum_out
        
        # Safe KL Divergence
        p_out_safe = p_out.clamp(min=eps)
        p_in_safe = p_in.clamp(min=eps)
        kld = torch.sum(p_in_safe * torch.log(p_in_safe / p_out_safe), dim=1)
        
        kld = torch.where(torch.isfinite(kld), kld, torch.zeros_like(kld))
        
        return kld.mean()

    def _batch_gather(self, params: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """배치 단위 인덱싱"""
        B, N, C = params.shape
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, C)
        return torch.gather(params, 1, indices_expanded)

    def _get_density_field(
        self,
        queries: torch.Tensor,
        g_dict: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """GMM Density: Sum( alpha * exp(-dist^2 / scale^2) )"""
        xyz = g_dict['xyz']
        scale = torch.exp(g_dict['scale']).clamp(min=1e-6, max=100.0)
        opacity = torch.sigmoid(g_dict['opacity']).clamp(1e-6, 1.0)
        
        if mask is not None:
            valid_mask = (~mask).unsqueeze(-1).float()
            opacity = opacity * valid_mask

        diff = queries.unsqueeze(2) - xyz.unsqueeze(1)
        scale_sq = scale.unsqueeze(1) ** 2 + 1e-8
        inv_scale_sq = 1.0 / scale_sq
        
        dist_sq = torch.sum((diff ** 2) * inv_scale_sq, dim=-1).clamp(max=50)
        
        weights = opacity.squeeze(-1).unsqueeze(1)
        density_per_gaussian = weights * torch.exp(-0.5 * dist_sq)
        total_density = torch.sum(density_per_gaussian, dim=-1)
        
        return total_density

    # =========================================================================
    # [2] Random Rendering Loss (gsplat)
    # =========================================================================
    def compute_rendering_loss(
        self,
        in_g: Dict[str, torch.Tensor],
        out_g: Dict[str, torch.Tensor],
        in_mask: Optional[torch.Tensor],
        out_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """gsplat을 사용한 렌더링 Loss 계산"""
        B = in_g['xyz'].shape[0]
        device = in_g['xyz'].device
        total_loss = 0.0
        valid_count = 0
        
        H, W = self.render_resolution, self.render_resolution
        fov_y = math.radians(60)
        
        for b in range(B):
            in_g_b = {k: v[b] for k, v in in_g.items()}
            out_g_b = {k: v[b] for k, v in out_g.items()}
            
            in_mask_b = in_mask[b] if in_mask is not None else None
            out_mask_b = out_mask[b] if out_mask is not None else None
            
            if in_mask_b is not None and (~in_mask_b).sum() == 0:
                continue
            
            view_mat, K = self._get_random_camera(device, H, W, fov_y, radius=5.0)
            
            img_in = self._render_with_gsplat(in_g_b, in_mask_b, view_mat, K, H, W)
            img_out = self._render_with_gsplat(out_g_b, out_mask_b, view_mat, K, H, W)
            
            # Debug 이미지 저장
            if b == 0:
                self._render_call_count += 1
                if self._render_call_count % self.debug_save_interval == 0:
                    n_in = (~in_mask_b).sum().item() if in_mask_b is not None else in_g_b['xyz'].shape[0]
                    n_out = out_g_b['xyz'].shape[0]
                    self._save_render_pair(img_in, img_out, n_in, n_out)
            

            if img_in.abs().sum() > 1e-6 or img_out.abs().sum() > 1e-6:
                
                # alpha = min(1.0, self._iteration / max(self.warmup_iterations, 1))
                # panalty_balck = (1 - alpha) * (1 - img_out.max()) * img_in.max()
                panalty_balck = (img_in.max() - img_out.max()).abs() + (img_in.max() - img_out.mean()).abs()

                loss_global_l1 = F.l1_loss(img_out, img_in)
                
                pixel_brightness_in = img_in.sum(dim=-1)
                mask_rendered = (pixel_brightness_in > 0.01).float()
                n_rendered = mask_rendered.sum() + 1e-6
                
                diff = (img_out - img_in).abs()
                loss_masked_l1 = (diff * mask_rendered.unsqueeze(-1)).sum() / (n_rendered * 3)
                
                loss_ssim = (1 - fused_ssim(
                    img_out.permute(2, 0, 1).unsqueeze(0),
                    img_in.permute(2, 0, 1).unsqueeze(0),
                    padding="valid"
                ))
                total_loss += 1.0 * loss_masked_l1 + 0.2 * loss_ssim + 0.1 * loss_global_l1 + panalty_balck
                valid_count += 1
        
        if valid_count == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        return total_loss / valid_count
    
    def _render_with_gsplat(
        self,
        g_dict: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor],
        view_mat: torch.Tensor,
        K: torch.Tensor,
        H: int,
        W: int
    ) -> torch.Tensor:
        """단일 Voxel에 대해 gsplat 렌더링 수행"""
        device = g_dict['xyz'].device
        
        means = g_dict['xyz'].contiguous()
        quats = g_dict['rotation'].contiguous()
        scales = torch.exp(g_dict['scale']).contiguous()
        opacities = torch.sigmoid(g_dict['opacity'].squeeze(-1)).contiguous()
        
        # SH 계수: [N, 3] + [N, 45] → [N, 16, 3]
        sh_dc = g_dict['sh_dc']
        sh_rest = g_dict['sh_rest']
        
        N = means.shape[0]
        sh_higher = sh_rest.view(N, 15, 3)
        sh_dc_expanded = sh_dc.unsqueeze(1)
        sh = torch.cat([sh_dc_expanded, sh_higher], dim=1)
        
        # 값 범위 제한
        scales = scales.clamp(min=1e-6, max=10.0)
        opacities = opacities.clamp(0.0, 1.0)
        quats = F.normalize(quats, dim=-1)
        
        # Masking
        if mask is not None:
            valid_idx = ~mask
            if valid_idx.sum() == 0:
                return torch.zeros((H, W, 3), device=device, requires_grad=True)
            
            means = means[valid_idx]
            scales = scales[valid_idx]
            quats = quats[valid_idx]
            opacities = opacities[valid_idx]
            sh = sh[valid_idx]
        
        if means.shape[0] == 0:
            return torch.zeros((H, W, 3), device=device, requires_grad=True)
        
        try:
            viewmats = view_mat.unsqueeze(0)
            Ks = K.unsqueeze(0)
            
            render_colors, render_alphas, meta = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=sh,
                viewmats=viewmats,
                Ks=Ks,
                width=W,
                height=H,
                sh_degree=3,
                near_plane=0.01,
                far_plane=100.0,
                render_mode="RGB",
            )
            
            render_colors = render_colors.squeeze(0).clamp(0.0, 1.0)
            return render_colors
        except Exception as e:
            print(f"[gsplat error] {e}")
            return torch.zeros((H, W, 3), device=device, requires_grad=True)
    
    def _get_random_camera(
        self, 
        device: torch.device, 
        H: int, 
        W: int, 
        fov_y: float, 
        radius: float = 3.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Voxel(-1~1)을 바라보는 랜덤 궤도 카메라"""
        theta = torch.rand(1).item() * 2 * math.pi
        phi = math.acos(1 - 2 * torch.rand(1).item())
        
        cx = radius * math.sin(phi) * math.cos(theta)
        cy = radius * math.sin(phi) * math.sin(theta)
        cz = radius * math.cos(phi)
        
        eye = torch.tensor([cx, cy, cz], device=device, dtype=torch.float32)
        at = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)
        up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
        
        forward = F.normalize(at - eye, dim=0)
        right = F.normalize(torch.linalg.cross(forward, up), dim=0)
        up_new = torch.linalg.cross(right, forward)
        
        R = torch.stack([right, -up_new, forward], dim=0)
        t = -torch.matmul(R, eye)
        
        view_mat = torch.eye(4, device=device)
        view_mat[:3, :3] = R
        view_mat[:3, 3] = t
        
        f_y = H / (2 * math.tan(fov_y / 2))
        K = torch.eye(3, device=device)
        K[0, 0] = f_y
        K[1, 1] = f_y
        K[0, 2] = W / 2.0
        K[1, 2] = H / 2.0
        
        return view_mat, K
    
    def _save_render_pair(self, img_in: torch.Tensor, img_out: torch.Tensor, n_in: int, n_out: int):
        """디버그용: 렌더링 쌍 이미지 저장"""
        from PIL import Image
        import numpy as np
        
        os.makedirs(self.debug_save_dir, exist_ok=True)
        
        idx = self._render_save_count
        self._render_save_count += 1
        
        img_in_np = (img_in.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
        img_out_np = (img_out.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
        combined = np.concatenate([img_in_np, img_out_np], axis=1)
        
        Image.fromarray(img_in_np).save(f"{self.debug_save_dir}/render_{idx:04d}_input.png")
        Image.fromarray(img_out_np).save(f"{self.debug_save_dir}/render_{idx:04d}_output.png")
        Image.fromarray(combined).save(f"{self.debug_save_dir}/render_{idx:04d}_combined.png")
        
        print(f"[debug] Saved render #{idx} (input: {n_in} gs, output: {n_out} gs)")

    # =========================================================================
    # [3] Sparsity Loss
    # =========================================================================
    def compute_sparsity_loss(
        self,
        out_g: Dict[str, torch.Tensor],
        out_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        출력 가우시안 Sparsity 정규화 (Annealing)
        
        1. Opacity L1: 불필요한 Gaussian 억제
        2. Volume Regularization: 큰 Scale 억제 (opacity * scale^3)
        
        초기: Sparsity 약하게 → 자유롭게 표현
        후기: Sparsity 강하게 → 효율적인 표현 유도
        """
        
        # Annealing factor
        alpha = min(1.0, self._iteration / max(self.warmup_iterations, 1))

        opacity = torch.sigmoid(out_g['opacity'])
        
        # Volume: product of 3D scale
        log_volume = out_g['scale'].sum(dim=-1, keepdim=True)
        
        if out_mask is not None:
            valid_mask = (~out_mask).unsqueeze(-1).float()
            loss_opacity = (opacity * valid_mask).sum() / (valid_mask.sum() + 1e-6)
            loss_volume = (log_volume * valid_mask).sum() / (valid_mask.sum() + 1e-6)
        else:
            loss_opacity = opacity.mean()
            loss_volume = log_volume.mean()
        
        sparsity_loss = alpha * loss_opacity + 0.02 * loss_volume

        return sparsity_loss * 0 #FIXME