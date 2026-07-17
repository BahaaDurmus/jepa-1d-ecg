import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from dataset import MIMICDataset, jepa_collate_fn
from model import JEPA_1D

print("Tum test verileri yukleniyor...")
mimic_dir = r"C:\Users\Acer\Downloads\jepa\mimic_small"
test_ds = MIMICDataset(data_dir=mimic_dir, is_train=False, segment_length=5000)

# Batch size 1 yaparak tek tek hepsini test edelim
loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=jepa_collate_fn)

model = JEPA_1D(seq_len=5000, in_chans=12, patch_size=300, embed_dim=128)
ckpt = torch.load(r"C:\Users\Acer\Downloads\jepa\checkpoints\best_jepa.pt", weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

total_samples = len(test_ds)
all_losses = []
all_cos_sims = []

print(f"\nToplam Test Edilecek Kayit: {total_samples}")
print("-" * 65)
print(f"{'Test No':<8} | {'Gizlenen Yamalar':<20} | {'Loss (Hata)':<12} | {'Benzerlik (%)':<15}")
print("-" * 65)

with torch.no_grad():
    for idx, (x, ctx_idx, tgt_idx) in enumerate(loader):
        loss, preds, targets = model(x, ctx_idx, tgt_idx)
        
        # O kayittaki tum gizli yamalar icin ortalama benzerligi al
        cos_sim = F.cosine_similarity(preds.reshape(-1, 128), targets.reshape(-1, 128), dim=1)
        mean_cos = cos_sim.mean().item()
        
        all_losses.append(loss.item())
        all_cos_sims.append(mean_cos)
        
        tgt_list = tgt_idx[0].tolist()
        # % formati
        benzerlik_yuzde = f"%{mean_cos * 100:.1f}"
        
        print(f"Kayit {idx+1:<2} | {str(tgt_list):<20} | {loss.item():.4f}       | {benzerlik_yuzde:<15}")

print("-" * 65)
print(f"ORTALAMA : Loss = {np.mean(all_losses):.4f}       | Benzerlik = %{np.mean(all_cos_sims)*100:.1f}")
