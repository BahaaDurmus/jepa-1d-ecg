import torch
import torch.nn as nn

class JEPADecoder_1D(nn.Module):
    def __init__(self, embed_dim=128, patch_size=50, in_chans=12):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, patch_size * in_chans)
        )
    def forward(self, x):
        B, N, _ = x.shape
        out = self.decoder(x)
        out = out.reshape(B, N, self.in_chans, self.patch_size)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.reshape(B, self.in_chans, N * self.patch_size)
        return out

class JEPADecoder_Conv1D(nn.Module):
    def __init__(self, embed_dim=128, in_chans=12):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=embed_dim, out_channels=64, kernel_size=5, stride=5),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=5, stride=5),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=in_chans, kernel_size=2, stride=2)
        )
    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        out = self.decoder(x)
        return out

class JEPADecoder_Robust(nn.Module):
    def __init__(self, embed_dim=128, patch_size=50, in_chans=12):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        
        # 1. Asama: Yamalarin birbiriyle iletisim kurmasini sagla (Sinirlari puruzsuzlestirir)
        self.mixer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=512, batch_first=True)
        
        # 2. Asama: Orijinal EKG'ye (yüksek boyutlara) genisle
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, patch_size * in_chans)
        )
        
    def forward(self, x):
        # x boyutu: (B, N, 128)
        x = self.mixer(x)
        out = self.proj(x) # (B, N, 600)
        
        B, N, _ = out.shape
        out = out.reshape(B, N, self.in_chans, self.patch_size)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.reshape(B, self.in_chans, N * self.patch_size)
        return out
