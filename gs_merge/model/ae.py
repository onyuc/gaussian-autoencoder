"""
GaussianMergingAE: Main AutoEncoder Model

Transformer Encoder-Decoder 기반 Gaussian 압축 모델
- Input: N개의 Gaussian (하나의 Voxel)
- Output: M개의 Gaussian (압축된 표현)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from gs_merge.model.encoder import PositionalEncoding, FourierPositionalEncoding
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
        num_queries: int = 32,
        nhead: int = 8,
        num_enc_layers: int = 4,
        num_dec_layers: int = 4,
        max_octree_level: int = 8
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_queries = num_queries
        
        # ============================================
        # [1] Input Embedding
        # ============================================
        
        # Feature Projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # Positional Encoding
        self.pos_encoder = FourierPositionalEncoding(latent_dim)
        
        # Level Embedding (Voxel 크기 정보)
        self.level_embed = nn.Embedding(max_octree_level + 1, latent_dim)
        
        # Global Token [CLS]
        self.global_token = nn.Parameter(torch.randn(1, 1, latent_dim))
        
        # ============================================
        # [2] Transformer Backbone
        # ============================================
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.0,  # 또는 0.05
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_enc_layers)
        
        # Decoder Queries (Learnable)
        self.seed_queries = nn.Parameter(torch.randn(1, num_queries, latent_dim))
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=512,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_dec_layers)
        
        # ============================================
        # [3] Output Heads
        # ============================================
        self.heads = GaussianHeads(latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        voxel_levels: torch.Tensor,
        src_padding_mask: torch.Tensor = None
    ) -> tuple:
        """
        Args:
            x: [B, N, 59] 정규화된 Input Gaussians
            voxel_levels: [B] 각 Voxel의 Octree Level
            src_padding_mask: [B, N] True=패딩(무시)
        
        Returns:
            xyz: [B, M, 3]
            rot: [B, M, 4]
            scale: [B, M, 3]
            opacity: [B, M, 1]
            sh: [B, M, 48]
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
        
        queries = self.seed_queries.expand(B, -1, -1)  # [B, M, latent]
        out_feat = self.decoder(
            queries, memory,
            memory_key_padding_mask=combined_mask
        )
        
        # ============================================
        # [Step 5] Head Prediction
        # ============================================
        
        return self.heads(out_feat)
    
    @property
    def num_parameters(self) -> int:
        """총 파라미터 수"""
        return sum(p.numel() for p in self.parameters())
    
    @property
    def num_trainable_parameters(self) -> int:
        """학습 가능한 파라미터 수"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

