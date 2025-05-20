import torch
import torch.nn as nn
import torch.nn.functional as F


class cDeepFusion(nn.Module):
    def __init__(self, 
                 embed_dim=768,
                 num_heads=8,
                 hidden_dim=256,
                 attn_drop=0.1,
                 proj_drop=0.1):
        super().__init__()
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True
        )
        
        # Post-processing
        self.attn_norm = nn.LayerNorm(embed_dim)

        # Dual Gating
        self.gate_global = nn.Sequential(
            nn.Linear(2*embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1), 
            nn.Sigmoid()
        )
        self.gate_channel = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Linear(embed_dim//2, embed_dim),
            nn.Sigmoid()
        )
        
    def forward(self, main_feat, prompt_feat):

        Q = main_feat
        K = prompt_feat
        V = K

        attn_out, _ = self.cross_attn(Q, K, V)      # [B, L, D_model]
        attn_out = self.attn_norm(attn_out)     


        global_gate = self.gate_global(
            torch.cat([attn_out.max(dim=1).values, main_feat.mean(dim=1)], dim=1)
        ).unsqueeze(1)  # [B, 1, 1]
        
        channel_gate = self.gate_channel(attn_out)  # [B, L, D_model]


        # fused = main_feat * global_gate + attn_out * channel_gate * (1-global_gate)
        fused = main_feat + attn_out * channel_gate * (1-global_gate)
        return fused

