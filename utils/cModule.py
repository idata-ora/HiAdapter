import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
import math


from pdb import set_trace as st


class CTnet(nn.Module):
    def __init__(self, embed_dim, img_size=224):
        super().__init__()

        self.down = nn.Sequential(
            nn.Conv2d(3, embed_dim//4, 3, stride=4, padding=1, bias=False),
            nn.SyncBatchNorm(embed_dim//4),
            nn.GELU(),
            nn.Conv2d(embed_dim//4, embed_dim//2, 3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim, 3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(embed_dim),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(embed_dim)


    def forward(self, x):
        c_down = self.down(x)  # [B, D, H//16, W//16]
        e = c_down.flatten(2).permute(0, 2, 1)  # [B, H*W, D]
        e = self.norm(e)

        return e, c_down



class StainNet(nn.Module):
    def __init__(self, embed_dim, img_size):
        super().__init__()
        self.net = CTnet(embed_dim, img_size)

    def forward(self, x):
        x = torch.clamp(x, 1e-6, 1.0)
        # OD = -log(I / I0)
        od = -torch.log(x) / torch.log(torch.tensor(10.0))

        e, c_down = self.net(od)  

        return e
    
