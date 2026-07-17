import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_single_beat import SingleBeatDataset, jepa_single_beat_collate_fn
from model import JEPA_1D
from decoder import JEPADecoder_Robust

EPOCHS = 30
BATCH_SIZE = 16
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

mimic_dir = r"C:\Users\Acer\Downloads\jepa\mimic_small"
train_ds = SingleBeatDataset(data_dir=mimic_dir, is_train=True, segment_length=300)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=jepa_single_beat_collate_fn, drop_last=True)

jepa = JEPA_1D(seq_len=300, in_chans=12, patch_size=50, embed_dim=128).to(DEVICE)
ckpt = torch.load(r"C:\Users\Acer\Downloads\jepa\checkpoints\best_jepa_single_beat.pt", map_location=DEVICE, weights_only=True)
jepa.load_state_dict(ckpt['model_state_dict'])
jepa.eval()
for param in jepa.parameters():
    param.requires_grad = False

# Robust (Transformer tabanli) Decoder
decoder = JEPADecoder_Robust(embed_dim=128, patch_size=50, in_chans=12).to(DEVICE)
optimizer = torch.optim.AdamW(decoder.parameters(), lr=LR)

# QRS tepelerini ezmemesi icin MSE yerine SmoothL1Loss (L1 + MSE hibrit) kullaniyoruz
criterion = nn.SmoothL1Loss()

print("Robust Decoder Egitimi Basliyor (30 Epoch)...")
for epoch in range(1, EPOCHS + 1):
    decoder.train()
    total_loss = 0
    for x, ctx_idx, tgt_idx in train_loader:
        x = x.to(DEVICE)
        
        with torch.no_grad():
            latents = jepa.target_encoder(x)
            # COK ONEMLI HATA DUZELTMESI: JEPA Predictor, Target'lari LayerNorm'dan gecirerek egitildi.
            # Decoder'in da ayni LayerNorm uzayini ogrenmesi lazim!
            import torch.nn.functional as F
            latents = F.layer_norm(latents, (latents.shape[-1],))
            
        reconstructed = decoder(latents)
        
        loss = criterion(reconstructed, x)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f"Epoch {epoch:2d} | Decoder SmoothL1 Loss: {total_loss / len(train_loader):.4f}")

torch.save(decoder.state_dict(), r"C:\Users\Acer\Downloads\jepa\checkpoints\best_decoder.pt")
print("Robust Decoder egitimi tamamlandi ve kaydedildi.")
