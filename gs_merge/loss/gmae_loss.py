"""
GMAE Loss: Distribution-based Loss for Gaussian Reconstruction

구성요소:
1. Density Field KL Divergence - 공간 분포 매칭
2. Random View Rendering Loss - 시각적 일관성  
3. Opacity Sparsity - 불필요한 Gaussian 억제

Includes:
- GMAELoss: 메인 손실 함수
- ChamferLoss: Baseline 손실 함수
- Utility functions: parse_gaussian_tensor, model_output_to_dict
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from gsplat import rasterization
from gs_merge.utils.debug_utils import print_gaussian_stats
from fused_ssim import fused_ssim


# =============================================================================
# Utility Functions
# =============================================================================

def parse_gaussian_tensor(data: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    [B, N, 59] 또는 [N, 59] 텐서를 Gaussian 속성 딕셔너리로 분해
    
    순서: xyz(3) + rot(4) + scale(3) + opacity(1) + sh_dc(3) + sh_rest(45)
    """
    if data.dim() == 2:
        data = data.unsqueeze(0)
    
    return {
        'xyz': data[..., :3],
        'rotation': data[..., 3:7],
        'scale': data[..., 7:10],
        'opacity': data[..., 10:11],
        'sh_dc': data[..., 11:14],
        'sh_rest': data[..., 14:59]
    }


def model_output_to_dict(
    xyz: torch.Tensor, 
    rot: torch.Tensor, 
    scale: torch.Tensor, 
    opacity: torch.Tensor, 
    sh: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    모델 출력을 Gaussian 딕셔너리로 변환
    
    Args:
        xyz: [B, M, 3]
        rot: [B, M, 4]
        scale: [B, M, 3] (log space)
        opacity: [B, M, 1] (logit space)
        sh: [B, M, 48] (sh_dc + sh_rest)
    
    Returns:
        딕셔너리 형태의 Gaussian 속성
    """
    return {
        'xyz': xyz,
        'rotation': rot,
        'scale': scale,
        'opacity': opacity,
        'sh_dc': sh[..., :3],
        'sh_rest': sh[..., 3:48]
    }


def _is_main_process() -> bool:
    """분산 학습 시 메인 프로세스인지 확인"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return local_rank == 0


def _print_main(*args, **kwargs):
    """메인 프로세스에서만 출력"""
    if _is_main_process():
        print(*args, **kwargs)


# =============================================================================
# ChamferLoss (Baseline)
# =============================================================================

class ChamferLoss(nn.Module):
    """
    Chamfer Distance 기반 Baseline 손실
    
    위치 매칭 후 각 속성(rotation, scale, opacity, SH)에 대해 L1/L2 손실 적용
    """
    
    def __init__(
        self,
        pos_weight: float = 1.0,
        rot_weight: float = 0.5,
        scale_weight: float = 0.5,
        opacity_weight: float = 0.3,
        sh_weight: float = 0.2
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.rot_weight = rot_weight
        self.scale_weight = scale_weight
        self.opacity_weight = opacity_weight
        self.sh_weight = sh_weight
    
    def forward(
        self,
        input_g: Dict[str, torch.Tensor],
        output_g: Dict[str, torch.Tensor],
        input_mask: Optional[torch.Tensor] = None,
        output_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Chamfer distance + 매칭 기반 속성 손실"""
        B, N, _ = input_g['xyz'].shape
        M = output_g['xyz'].shape[1]
        device = input_g['xyz'].device
        
        # 거리 행렬
        dist = torch.cdist(output_g['xyz'], input_g['xyz'], p=2)
        if input_mask is not None:
            dist = dist.masked_fill(input_mask.unsqueeze(1), float('inf'))
        
        # 매칭
        min_out_to_in, match_idx = dist.min(dim=2)
        min_in_to_out, _ = dist.min(dim=1)
        
        # Position Loss
        if input_mask is not None:
            valid = ~input_mask
            pos_loss = min_out_to_in.mean() + (min_in_to_out * valid).sum() / valid.sum().clamp(min=1)
        else:
            pos_loss = min_out_to_in.mean() + min_in_to_out.mean()
        pos_loss = pos_loss / 2
        
        # 매칭된 속성
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, M)
        matched_rot = input_g['rotation'][batch_idx, match_idx]
        matched_scale = input_g['scale'][batch_idx, match_idx]
        matched_opacity = input_g['opacity'][batch_idx, match_idx]
        
        # Rotation Loss
        rot_dot = (output_g['rotation'] * matched_rot).sum(dim=-1).abs()
        rot_loss = (1 - rot_dot).mean()
        
        # Scale Loss
        scale_loss = F.l1_loss(output_g['scale'], matched_scale)
        
        # Opacity Loss
        opacity_loss = F.l1_loss(output_g['opacity'], matched_opacity)
        
        # SH Loss
        if 'sh_dc' in input_g and 'sh_dc' in output_g:
            matched_sh_dc = input_g['sh_dc'][batch_idx, match_idx]
            matched_sh_rest = input_g['sh_rest'][batch_idx, match_idx]
            sh_loss = F.l1_loss(output_g['sh_dc'], matched_sh_dc) + F.l1_loss(output_g['sh_rest'], matched_sh_rest)
        else:
            sh_loss = torch.tensor(0.0, device=device)
        
        # Total
        total_loss = (
            self.pos_weight * pos_loss +
            self.rot_weight * rot_loss +
            self.scale_weight * scale_loss +
            self.opacity_weight * opacity_loss +
            self.sh_weight * sh_loss
        )
        
        return total_loss, {
            'loss_pos': pos_loss.item(),
            'loss_rot': rot_loss.item(),
            'loss_scale': scale_loss.item(),
            'loss_opacity': opacity_loss.item(),
            'loss_sh': sh_loss.item() if isinstance(sh_loss, torch.Tensor) else sh_loss,
            'loss_total': total_loss.item()
        }


# =============================================================================
# GMAELoss (Main Loss)
# =============================================================================

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
        warmup_iterations: int = 200
    ):
        super().__init__()
        self.lambda_density = lambda_density
        self.lambda_render = lambda_render
        self.lambda_sparsity = lambda_sparsity
        self.n_samples = n_density_samples
        self.render_resolution = render_resolution
        self.warmup_iterations = warmup_iterations
        
        self._debug_count = 0
        self._iteration = 0

    def forward(
        self, 
        input_g: Dict[str, torch.Tensor],
        output_g: Dict[str, torch.Tensor], 
        input_mask: Optional[torch.Tensor] = None,
        output_mask: Optional[torch.Tensor] = None,
        return_debug_info: bool = False
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
        if return_debug_info:
            loss_render, debug_info = self.compute_rendering_loss(
                input_g, output_g, input_mask, output_mask, return_debug_info=True
            )
        else:
            loss_render = self.compute_rendering_loss(input_g, output_g, input_mask, output_mask)
        
        # 3. Sparsity Loss
        loss_sparsity = self.compute_sparsity_loss(output_g, output_mask)
        
        # Iteration counter
        self._iteration += 1
        
        # 가우시안 통계 수집 (입력 + 출력)
        output_stats = self._compute_gaussian_stats(output_g, output_mask, prefix='output')
        input_stats = self._compute_gaussian_stats(input_g, input_mask, prefix='input')
        
        # Total Loss
        total_loss = (
            self.lambda_density * loss_density +
            self.lambda_render * loss_render +
            self.lambda_sparsity * loss_sparsity
        )
        
        loss_dict = {
            "loss_density": loss_density.item(),
            "loss_render": loss_render.item(),
            "loss_sparsity": loss_sparsity.item(),
            "loss_total": total_loss.item(),
            **output_stats,  # 출력 가우시안 통계
            **input_stats    # 입력 가우시안 통계
        }
        
        if return_debug_info:
            # 히스토그램 데이터 추가 (입력 + 출력)
            debug_info['histograms'] = {
                **{f'output_{k}': v for k, v in self._get_histograms(output_g, output_mask).items()},
                **{f'input_{k}': v for k, v in self._get_histograms(input_g, input_mask).items()}
            }
            return total_loss, loss_dict, debug_info
        else:
            return total_loss, loss_dict

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
        """
        3DGS Density with Rotation
        exp( -0.5 * (x-mu)^T * Sigma^-1 * (x-mu) )
        """
        B, N, _ = g_dict['xyz'].shape
        device = g_dict['xyz'].device
        
        # 1. 파라미터 준비
        xyz = g_dict['xyz']      # [B, N, 3]
        scale = torch.exp(g_dict['scale']).clamp(min=1e-6, max=100.0) # [B, N, 3]
        opacity = torch.sigmoid(g_dict['opacity']).clamp(1e-6, 1.0)   # [B, N, 1]
        rot = F.normalize(g_dict['rotation'], dim=-1) # [B, N, 4] Quaternion
        
        # 2. 마스크 처리
        if mask is not None:
            valid_mask = (~mask).unsqueeze(-1).float()
            opacity = opacity * valid_mask

        # 3. Quaternion -> Rotation Matrix 변환
        # (PyTorch 내장 함수가 없다면 아래 헬퍼 함수 사용, gsplat 등 외부 라이브러리 써도 됨)
        R = self._quat_to_rotmat(rot) # [B, N, 3, 3]

        # 4. 거리 벡터 계산 (World Space)
        # queries: [B, M, 3]
        # xyz:     [B, N, 3]
        # diff:    [B, M, N, 3]
        diff = queries.unsqueeze(2) - xyz.unsqueeze(1) 

        # 5. [핵심] Local Frame으로 회전 (World Diff -> Local Diff)
        # diff_local = diff @ R (Batch Matmul)
        # R은 [B, N, 3, 3]이므로, K차원에 대해 브로드캐스팅 필요
        # 식: v_local = v_world * R  (R은 row-major 기준, 혹은 R^T 곱하기)
        # Quaternion to Rotmat 보통 R * v 형태이므로, v^T * R^T = (R * v)^T 
        # 간단히: diff 벡터를 가우시안의 회전 역방향으로 돌림
        
        # [B, 1, N, 3, 3]으로 확장하여 곱셈
        # diff: [B, K, N, 3] -> [B, K, N, 1, 3]
        # R: [B, N, 3, 3] -> [B, 1, N, 3, 3]
        # 결과: [B, K, N, 1, 3] @ [B, 1, N, 3, 3] (Transpose R) -> 복잡함.
        
        # 더 쉬운 방법: einsum 사용
        # b: batch, k: query, n: gaussian, i: world_dim, j: local_dim
        # diff[b,k,n,i], R[b,n,j,i] (R^T를 곱해야 Local로 감) -> local[b,k,n,j]
        diff_local = torch.einsum('bkni,bnji->bknj', diff, R)
        
        # 6. Scale로 나누기 (Mahalanobis Distance의 핵심)
        # 이제 로컬 좌표계이므로 그냥 스케일로 나누면 됨
        # scale: [B, N, 3] -> [B, 1, N, 3]
        inv_scale = 1.0 / (scale.unsqueeze(1) + 1e-8)
        norm_diff = diff_local * inv_scale
        
        # 7. 거리 제곱 합 (Squared Norm)
        dist_sq = torch.sum(norm_diff ** 2, dim=-1).clamp(max=50) # [B, K, N]
        
        # 8. Gaussian Density 계산
        weights = opacity.squeeze(-1).unsqueeze(1) # [B, 1, N]
        density_per_gaussian = weights * torch.exp(-0.5 * dist_sq)
        
        # 9. 최종 합산
        total_density = torch.sum(density_per_gaussian, dim=-1) # [B, K]
        
        return total_density

    def _quat_to_rotmat(self, quat):
        """Quaternion(w,x,y,z) -> Rotation Matrix(3x3)"""
        # 정규화 가정 (quat은 이미 normalize 되어야 함)
        w, x, y, z = quat.unbind(-1)
        
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z
        
        row0 = torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1)
        row1 = torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1)
        row2 = torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1)
        
        return torch.stack([row0, row1, row2], dim=-2) # [..., 3, 3]

    # =========================================================================
    # [2] Random Rendering Loss (gsplat)
    # =========================================================================
    def compute_rendering_loss(
        self,
        in_g: Dict[str, torch.Tensor],
        out_g: Dict[str, torch.Tensor],
        in_mask: Optional[torch.Tensor],
        out_mask: Optional[torch.Tensor],
        return_debug_info: bool = False
    ) -> torch.Tensor:
        """gsplat을 사용한 렌더링 Loss 계산"""
        B = in_g['xyz'].shape[0]
        device = in_g['xyz'].device
        total_loss = 0.0
        valid_count = 0
        
        H, W = self.render_resolution, self.render_resolution
        fov_y = math.radians(60)
        debug_renders = [] if return_debug_info else None
        
        for b in range(B):
            in_g_b = {k: v[b] for k, v in in_g.items()}
            out_g_b = {k: v[b] for k, v in out_g.items()}
            
            in_mask_b = in_mask[b] if in_mask is not None else None
            out_mask_b = out_mask[b] if out_mask is not None else None
            
            if in_mask_b is not None and (~in_mask_b).sum() == 0:
                continue
            
            view_mat, K = self._get_random_camera(device, H, W, fov_y, radius=5.0)
            
            # ========== 랜덤 배경 생성 ==========
            # 1. 완전 랜덤 (RGB 각 채널 독립)
            if torch.rand(1).item() < 0.5:
                backgrounds = torch.rand(3, device=device)  # [3]
            else:
                # 2. 흑백 랜덤 (그레이스케일)
                gray = torch.rand(1, device=device).item()
                backgrounds = torch.tensor([gray, gray, gray], device=device)  # [3]

            img_in, alpha_in = self._render_with_gsplat(in_g_b, in_mask_b, view_mat, K, H, W, backgrounds)
            img_out, alpha_out = self._render_with_gsplat(out_g_b, out_mask_b, view_mat, K, H, W, backgrounds)
            
            # 디버그 정보 저장 (첫 번째 배치만)
            if return_debug_info and b == 0:
                n_in = (~in_mask_b).sum().item() if in_mask_b is not None else in_g_b['xyz'].shape[0]
                n_out = (~out_mask_b).sum().item() if out_mask_b is not None else out_g_b['xyz'].shape[0]
                debug_renders.append({
                    'img_in': img_in.detach().cpu(),
                    'img_out': img_out.detach().cpu(),
                    'n_in': n_in,
                    'n_out': n_out
                })

            if img_in.abs().sum() > 1e-6 or img_out.abs().sum() > 1e-6:

                # Alpha 값 자체를 가중치로 사용 (연속값)
                # alpha_in: [1, H, W]
                alpha_mask = (alpha_in > 0.001).squeeze(0)  # [H, W] = 0~1 연속값
                n_rendered = alpha_mask.sum() + 1e-6
                
                diff = (img_out - img_in).abs()
                loss_rendered_masked_l1 = (diff * alpha_mask).sum() / (n_rendered * 3)
                loss_rendered_l1 = F.l1_loss(img_out, img_in)
                loss_rendered = 1.0 * loss_rendered_masked_l1 + 0.3 * loss_rendered_l1
                
                loss_alpha_l1 = F.l1_loss(alpha_out.squeeze(), alpha_in.squeeze())

                # RGB에만 SSIM 적용 
                loss_ssim_rendered = 1.0 - fused_ssim(img_out.permute(2,0,1).unsqueeze(0), img_in.permute(2,0,1).unsqueeze(0), padding='valid')

                total_loss += loss_rendered + loss_alpha_l1 + 0.3 * loss_ssim_rendered
                valid_count += 1
        
        if valid_count == 0:
            if return_debug_info:
                return torch.tensor(0.0, device=device, requires_grad=True), {'renders': debug_renders}
            else:
                return torch.tensor(0.0, device=device, requires_grad=True)
            
        loss = total_loss / valid_count
        if return_debug_info:
            return loss, {'renders': debug_renders}
        else:
            return loss
    
    def _render_with_gsplat(
        self,
        g_dict: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor],
        view_mat: torch.Tensor,
        K: torch.Tensor,
        H: int,
        W: int,
        backgrounds: torch.Tensor
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
        scales = scales.clamp(min=1e-6, max=20.0)
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
                backgrounds=backgrounds
            )
            
            render_colors = render_colors.squeeze(0).clamp(0.0, 1.0)
            return render_colors, render_alphas
        except Exception as e:
            _print_main(f"[gsplat error] {e}")
            return torch.zeros((H, W, 3), device=device, requires_grad=True), torch.zeros((H, W), device=device, requires_grad=True)
    
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
    
    # ==========================================================================
    # [3] Sparsity Loss
    # =========================================================================
    def compute_sparsity_loss(
        self,
        out_g: Dict[str, torch.Tensor],
        out_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        출력 가우시안 활용도 최대화 (Min-Value Penalty Loss)

        선택된 모든 가우시안의 opacity와 scale이 0에 가까워지면 강한 페널티 부여
        (quadratic penalty로 작은 값을 더 강하게 억제)
        
        Args:
            out_g: 출력 가우시안 파라미터
            out_mask: 패딩 마스크 (True=Invalid)
        """
        
        opacity = torch.sigmoid(out_g['opacity']).squeeze(-1)  # [B, M]
        log_scale = out_g['scale']  # [B, M, 3], 이미 log scale (학습 파라미터)
        
        # Valid mask 적용 (padding 제거)
        if out_mask is not None:
            valid_mask = ~out_mask
            if valid_mask.sum() == 0:
                return torch.tensor(0.0, device=opacity.device, requires_grad=True)
            valid_opacity = opacity[valid_mask]
            valid_log_scale = log_scale[valid_mask]
        else:
            valid_opacity = opacity
            valid_log_scale = log_scale
        
        # log_scale 범위: [ln(1e-8), ln(20)] = [-18.4, 2.996]
        log_scale = valid_log_scale.mean(dim=-1)  # [num_valid], 각 가우시안의 평균 scale
        
        # log_scale 공간에서 시그모이드 정규화 (각 가우시안별로 적용)
        normalized_scale = torch.sigmoid(log_scale)  # [num_valid], 각 가우시안의 정규화된 scale
                
        # 클램프된 값으로 페널티 계산
        MAX_OPACITY = 0.3
        valid_opacity_clamped = torch.clamp(valid_opacity, max=MAX_OPACITY)

        MAX_SCALE = 0.03
        normalized_scale_clamped = torch.clamp(normalized_scale, max=MAX_SCALE)
        
        valid_penalty = -1.0 * (torch.log(valid_opacity_clamped + 1e-20) - torch.log(torch.tensor(MAX_OPACITY + 1e-20))) * 5
        scale_penalty = -1.0 * (torch.log(normalized_scale_clamped + 1e-20) - torch.log(torch.tensor(MAX_SCALE + 1e-20)))
        std_penalty =  (1 - valid_opacity * normalized_scale).std() * 5
        linear_push_loss = (1.0 - valid_opacity) * 0.1 + (1.0 - normalized_scale) * 0.05
        presence_penalty = (valid_penalty + scale_penalty + linear_push_loss).mean()
        
        utilization_loss = presence_penalty + std_penalty
        
        return utilization_loss
    
    # =========================================================================
    # [4] Gaussian Statistics (for TensorBoard/Logging)
    # =========================================================================
    def _compute_gaussian_stats(
        self,
        out_g: Dict[str, torch.Tensor],
        out_mask: Optional[torch.Tensor],
        prefix: str = 'gaussian'
    ) -> Dict[str, float]:
        """가우시안 통계 계산 (opacity, scale 등)
        
        Args:
            out_g: 가우시안 파라미터
            out_mask: 패딩 마스크
            prefix: 통계 키 접두어 ('input', 'output', 'gaussian' 등)
        """
        stats = {}
        
        # Opacity 통계
        opacity_raw = out_g['opacity']  # [B, M, 1]
        opacity = torch.sigmoid(opacity_raw)
        
        if out_mask is not None:
            valid_opacity = opacity[~out_mask]
        else:
            valid_opacity = opacity.flatten()
        
        if len(valid_opacity) > 0:
            stats[f'{prefix}_opacity_mean'] = valid_opacity.mean().item()
            stats[f'{prefix}_opacity_std'] = valid_opacity.std().item()
            stats[f'{prefix}_opacity_min'] = valid_opacity.min().item()
            stats[f'{prefix}_opacity_max'] = valid_opacity.max().item()
            # 높은 opacity 가우시안 비율 (>0.5)
            stats[f'{prefix}_opacity_high_ratio'] = (valid_opacity > 0.5).float().mean().item()
        
        # Scale 통계 (x, y, z 각각)
        scale_raw = out_g['scale']  # [B, M, 3]
        scale = torch.exp(scale_raw).clamp(max=2.0)  # log scale -> real scale, 2 이상 클램프
        
        if out_mask is not None:
            valid_scale = scale[~out_mask]  # [num_valid, 3]
        else:
            valid_scale = scale.view(-1, 3)  # [B*M, 3]
        
        if len(valid_scale) > 0:
            # 전체 평균 scale
            scale_mean = valid_scale.mean(dim=-1)
            stats[f'{prefix}_scale_mean'] = scale_mean.mean().item()
            stats[f'{prefix}_scale_std'] = scale_mean.std().item()
            stats[f'{prefix}_scale_min'] = scale_mean.min().item()
            stats[f'{prefix}_scale_max'] = scale_mean.max().item()
            
            # 축별 통계
            for i, axis in enumerate(['x', 'y', 'z']):
                axis_scale = valid_scale[:, i]
                stats[f'{prefix}_scale_{axis}_mean'] = axis_scale.mean().item()
                stats[f'{prefix}_scale_{axis}_max'] = axis_scale.max().item()
                stats[f'{prefix}_scale_{axis}_min'] = axis_scale.min().item()
        
        # 유효 가우시안 개수 (mask 제외)
        if out_mask is not None:
            stats[f'{prefix}_count_valid'] = (~out_mask).sum().item() / out_mask.shape[0]  # 평균
        else:
            stats[f'{prefix}_count_valid'] = out_g['xyz'].shape[1]
        
        return stats
    
    def _get_histograms(
        self,
        out_g: Dict[str, torch.Tensor],
        out_mask: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """히스토그램용 텐서 반환 (TensorBoard)"""
        histograms = {}
        
        # Opacity
        opacity = torch.sigmoid(out_g['opacity'])
        if out_mask is not None:
            valid_opacity = opacity[~out_mask].detach().cpu()
        else:
            valid_opacity = opacity.flatten().detach().cpu()
        histograms['opacity'] = valid_opacity
        
        # Scale (평균, 2 이상 클램프)
        scale = torch.exp(out_g['scale']).clamp(max=2.0)
        scale_mean = scale.mean(dim=-1)
        if out_mask is not None:
            valid_scale = scale_mean[~out_mask].detach().cpu()
        else:
            valid_scale = scale_mean.flatten().detach().cpu()
        histograms['scale'] = valid_scale
        
        return histograms
