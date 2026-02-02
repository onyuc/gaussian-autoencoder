"""
GaussianMergingAE: Main AutoEncoder Model

Transformer Encoder-Decoder 기반 Gaussian 압축 모델
- Input: N개의 Gaussian (하나의 Voxel)
- Output: M개의 Gaussian (압축된 표현)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from gs_merge.model.encoder import PositionalEncoding, FourierPositionalEncoding, RatioEncoding
from gs_merge.model.heads import GaussianHeads


class GaussianMergingAE(nn.Module):
    """
    Gaussian Merging AutoEncoder
    
    Args:
        input_dim: Gaussian 속성 차원 (기본 59 = 3+4+3+1+48)
        latent_dim: Transformer 히든 차원
        num_queries: 출력 Gaussian 개수 (M)
        nhead: Multi-Head Attention 헤드 수
        num_enc_layers: Encoder 레이어 수
        num_dec_layers: Decoder 레이어 수
        max_octree_level: 최대 Octree 깊이 (Level Embedding용)
    """
    
    def __init__(
        self, 
        input_dim: int = 59,
        latent_dim: int = 256,
        num_inputs: int = 128,
        num_queries: int = 128,
        nhead: int = 8,
        num_enc_layers: int = 4,
        num_dec_layers: int = 4,
        max_octree_level: int = 8
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.N = num_inputs
        self.M = num_queries
        
        # ST / noise hyperparams
        self.gate_temperature = 1.0      # sigmoid temperature for ST backward path
        self._gumbel_noise_scale = 0.3   # gumbel scale (train only)
        self._gumbel_noise_scale_init = 0.3  # initial value for scheduling
        
        # ============================================
        # [1] Input Embedding
        # ============================================
        
        # Feature Projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # Positional Encoding
        self.pos_encoder = FourierPositionalEncoding(latent_dim)
        
        # Level Embedding (Voxel 크기 정보)
        self.level_embed = nn.Embedding(max_octree_level + 1, latent_dim)

        # Ratio Encoding
        self.ratio_encoder = RatioEncoding(latent_dim, num_frequencies=4)
        
        # Global Token [CLS]
        self.global_token = nn.Parameter(torch.randn(1, 1, latent_dim))
        
        # ============================================
        # [2] Transformer Backbone
        # ============================================
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=latent_dim * 4,
            dropout=0.0,  # 또는 0.05
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_enc_layers)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=latent_dim * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_dec_layers)
        
        # ============================================
        # [2.5] Query Conditioning Modules
        # ============================================
        # Query
        self.query_slots = nn.Parameter(torch.randn(1, num_queries, latent_dim))
        
        # Global Context
        self.query_film = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 2 * latent_dim)
        )
        self.query_norm = nn.LayerNorm(latent_dim)

        # Query selection score
        self.query_score = nn.Sequential(
            nn.Linear(latent_dim, latent_dim//2),
            nn.GELU(),
            nn.Linear(latent_dim//2, 1)
        )

        # ============================================
        # [3] Output Heads
        # ============================================
        self.heads = GaussianHeads(latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        voxel_levels: torch.Tensor,
        src_padding_mask: torch.Tensor = None,
        compression_ratio: float = 0.5
    ) -> tuple:
        """
        Args:
            x: [B, N, 59] 정규화된 Input Gaussians
            voxel_levels: [B] 각 Voxel의 Octree Level
            src_padding_mask: [B, N] True=패딩(무시)
            compression_ratio: 입력 대비 출력 비율 (기본 0.5)
        
        Returns:
            xyz: [B, M, 3]
            rot: [B, M, 4]
            scale: [B, M, 3]
            opacity: [B, M, 1]
            sh: [B, M, 48]
            tgt_padding_mask: [B, M] True=패딩(무시)
        """
        B, N, _ = x.shape
        
        # ============================================
        # [Step 1] Embedding
        # ============================================
        
        # Feature Projection
        features = self.input_proj(x)  # [B, N, latent]
        
        # Positional Encoding (xyz = x[:, :, :3])
        pos_feat = self.pos_encoder(x[:, :, :3])
        features = features + pos_feat
        
        # Level Embedding
        level_feat = self.level_embed(voxel_levels).unsqueeze(1)  # [B, 1, latent]
        features = features + level_feat
        
        # ============================================
        # [Step 2] Add Global Token
        # ============================================
        
        cls_tokens = self.global_token.expand(B, -1, -1)  # [B, 1, latent]
        x_with_cls = torch.cat((cls_tokens, features), dim=1)  # [B, N+1, latent]
        
        # Mask 처리
        if src_padding_mask is not None:
            cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=x.device)
            combined_mask = torch.cat((cls_mask, src_padding_mask), dim=1)
        else:
            combined_mask = None
        
        # ============================================
        # [Step 3] Encoding
        # ============================================
        
        memory = self.encoder(x_with_cls, src_key_padding_mask=combined_mask)
        
        # ============================================
        # [Step 4] Decoding
        # ============================================
        # 동적 쿼리 생성
        # memory에서 CLS 토큰 추출 (첫 번째 토큰)
        cls_token = memory[:, 0, :]  # [B, latent]
        
        queries, tgt_padding_mask, k, scores = self._condition_queries(
            cls_token=cls_token,
            compression_ratio=compression_ratio,
            voxel_levels=voxel_levels,
            src_padding_mask=src_padding_mask
        )
        
        out_feat = self.decoder(
            queries, memory,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=combined_mask
        )
        
        # ============================================
        # [Step 5] Head Prediction
        # ============================================
        
        outputs = self.heads(out_feat)

        return (*outputs, tgt_padding_mask)
    
    def _condition_queries(
        self, 
        cls_token: torch.Tensor,        # [B, latent] (Encoder의 CLS 출력)
        compression_ratio: float,       # 0.0 ~ 1.0 (타겟 프루닝 레이쇼)
        voxel_levels: torch.Tensor,    # [B] (각 Voxel의 Octree Level)
        src_padding_mask: torch.Tensor,  # [B, N] (Input 패딩 정보)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        글로벌 컨텍스트와 예산(Ratio)을 반영한 동적 쿼리 생성 및 마스킹
        
        Args:
            cls_token: Encoder에서 압축된 장면의 글로벌 특징 [B, latent]
            compression_ratio: 입력 대비 출력 목표 비율 (0.1 ~ 1.0)
            src_padding_mask: 원본 입력의 유효 길이 계산용 [B, N]
            
        Returns:
            queries: [B, M, latent] - 상황에 맞게 변형된 쿼리 뭉치
            tgt_padding_mask: [B, M] - Trt Mask-out용 마스크
        """
        B, C = cls_token.shape
        device, dtype = cls_token.device, cls_token.dtype

        # ---- ratio embed (dtype/device safe) ----
        r = float(compression_ratio)
        r = max(0.0, min(1.0, r))
        ratio_input = torch.full((B, 1), r, device=device, dtype=dtype)
        ratio_embed = self.ratio_encoder(ratio_input)            # [B,C]

        # ---- level embed ----
        level_embed = self.level_embed(voxel_levels)             # [B,C]

        # ---- FiLM ----
        film_in = cls_token + ratio_embed + level_embed          # [B,C]
        gamma_beta = self.query_film(film_in)                    # [B,2C]
        gamma, beta = gamma_beta.chunk(2, dim=-1)                # [B,C],[B,C]
        gamma = torch.tanh(gamma)

        # ---- base queries ----
        queries = self.query_slots.expand(B, -1, -1)             # [B,M,C]
        queries = queries * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        queries = self.query_norm(queries)

        # ---- valid length -> k ----
        if src_padding_mask is not None:
            valid_lengths = (~src_padding_mask).sum(dim=1)       # [B]
        else:
            valid_lengths = torch.full((B,), self.N, device=device, dtype=torch.long)

        k = torch.clamp((valid_lengths.float() * r).round().long(), min=1, max=self.M)  # [B]

        # ---- scores ----
        scores = self.query_score(queries).squeeze(-1)           # [B,M]

        # ---- noise on scores (train only) ----
        if self.training:
            noise = self._sample_gumbel(scores) * float(self._gumbel_noise_scale)
            scores_sel = scores + noise
        else:
            scores_sel = scores

        # ---- ST top-k gate (forward hard pruning, backward soft grad) ----
        w_st, keep = self._st_topk_gate(scores_sel, k, temperature=self.gate_temperature)

        # forward에서는 keep만 남고(drop은 0), backward는 soft grad가 scores로 흐름
        queries = queries * w_st[..., None]                       # [B,M,C]

        # decoder용 hard mask
        tgt_padding_mask = ~keep                                  # [B,M] True=ignore

        return queries, tgt_padding_mask, k, scores

    def _sample_gumbel(self, like: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
        U = torch.rand_like(like)
        return -torch.log(-torch.log(U + eps) + eps)

    def _st_topk_gate(self, scores: torch.Tensor, k: torch.Tensor, temperature: float):
        """
        scores: [B,M]
        k: [B]
        Returns:
        w_st: [B,M] float (forward hard, backward soft)
        keep: [B,M] bool
        """
        B, M = scores.shape
        device, dtype = scores.device, scores.dtype

        T = max(float(temperature), 1e-4)
        w_soft = torch.sigmoid(scores / T)  # backward path

        max_k = int(k.max().item())
        topk_idx = scores.topk(max_k, dim=1).indices  # [B,max_k]

        keep = torch.zeros((B, M), device=device, dtype=torch.bool)
        keep.scatter_(1, topk_idx, True)

        j = torch.arange(max_k, device=device).unsqueeze(0)  # [1,max_k]
        extra = j >= k.unsqueeze(1)                           # [B,max_k]
        if extra.any():
            b_idx, j_idx = extra.nonzero(as_tuple=True)
            keep[b_idx, topk_idx[b_idx, j_idx]] = False

        w_hard = keep.to(dtype)  # forward path (0/1)

        # ST: forward = hard, backward = soft
        w_st = w_hard + (w_soft - w_soft.detach())
        return w_st, keep

    # ============================================
    # [Gumbel Noise Scale Scheduling]
    # ============================================
    
    @property
    def gumbel_noise_scale(self) -> float:
        """현재 gumbel noise scale 값"""
        return self._gumbel_noise_scale
    
    @gumbel_noise_scale.setter
    def gumbel_noise_scale(self, value: float):
        """gumbel noise scale 설정 (0.0 ~ 1.0 권장)"""
        self._gumbel_noise_scale = max(0.0, float(value))
    
    def set_gumbel_noise_scale(self, value: float):
        """gumbel noise scale 설정 메서드 (체이닝 가능)"""
        self.gumbel_noise_scale = value
        return self
    
    def get_gumbel_noise_scale(self) -> float:
        """gumbel noise scale 조회"""
        return self._gumbel_noise_scale
    
    def reset_gumbel_noise_scale(self):
        """gumbel noise scale을 초기값으로 리셋"""
        self._gumbel_noise_scale = self._gumbel_noise_scale_init
        return self

    @property
    def num_parameters(self) -> int:
        """총 파라미터 수"""
        return sum(p.numel() for p in self.parameters())
    
    @property
    def num_trainable_parameters(self) -> int:
        """학습 가능한 파라미터 수"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

def st_topk_gate(scores: torch.Tensor, k: torch.Tensor, temperature: float = 1.0):
    """
    scores: [B, M]
    k: [B]  (batch별 k)
    Returns:
      w_st: [B, M] float, forward는 hard(0/1), backward는 soft(sigmoid) gradient
      keep: [B, M] bool, True=keep
    """
    B, M = scores.shape
    device, dtype = scores.device, scores.dtype

    T = max(float(temperature), 1e-4)
    w_soft = torch.sigmoid(scores / T)  # [B,M], (0,1)

    max_k = int(k.max().item())
    topk_idx = scores.topk(max_k, dim=1).indices  # [B,max_k]

    keep = torch.zeros((B, M), device=device, dtype=torch.bool)
    keep.scatter_(1, topk_idx, True)

    j = torch.arange(max_k, device=device).unsqueeze(0)  # [1,max_k]
    extra = j >= k.unsqueeze(1)  # [B,max_k]
    if extra.any():
        b_idx, j_idx = extra.nonzero(as_tuple=True)
        keep[b_idx, topk_idx[b_idx, j_idx]] = False

    w_hard = keep.to(dtype)  # [B,M], {0,1}

    # ST: forward는 hard, backward는 soft
    w_st = w_hard + (w_soft - w_soft.detach())
    return w_st, keep
