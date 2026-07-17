import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset_micro import MIMICDataset, jepa_micro_collate_fn
from model import JEPA_1D
import matplotlib.pyplot as plt

EPOCHS = 20
BATCH_SIZE = 8
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

mimic_dir = r"C:\Users\Acer\Downloads\jepa\mimic_small"
torch.manual_seed(42)

train_ds = MIMICDataset(data_dir=mimic_dir, is_train=True, segment_length=5000)
test_ds = MIMICDataset(data_dir=mimic_dir, is_train=False, segment_length=5000)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=jepa_micro_collate_fn, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=jepa_micro_collate_fn, drop_last=True)

# patch_size=50 yapilarak model mikro yamalamaya uygun hale getiriliyor
model = JEPA_1D(seq_len=5000, in_chans=12, patch_size=50, embed_dim=128).to(DEVICE)
optimizer = torch.optim.AdamW(list(model.context_encoder.parameters()) + list(model.predictor.parameters()), lr=LR, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

train_losses = []
test_losses = []
cos_sims = []

print("Mikro Yamalama (Atim-Ici Maskeleme) Egitimi Basliyor...")
print(f"Toplam Parametre: {sum(p.numel() for p in model.parameters()):,}")

best_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    ema_m = 0.996 + (1.0 - 0.996) * (epoch - 1) / max(EPOCHS - 1, 1)
    
    # Train
    model.train()
    train_loss_sum = 0
    for x, ctx_idx, tgt_idx in train_loader:
        x, ctx_idx, tgt_idx = x.to(DEVICE), ctx_idx.to(DEVICE), tgt_idx.to(DEVICE)
        loss, _, _ = model(x, ctx_idx, tgt_idx)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        model.update_target_encoder(m=ema_m)
        train_loss_sum += loss.item()
        
    train_losses.append(train_loss_sum / max(1, len(train_loader)))
    
    # Test
    model.eval()
    test_loss_sum = 0
    test_cos_sum = 0
    with torch.no_grad():
        for x, ctx_idx, tgt_idx in test_loader:
            x, ctx_idx, tgt_idx = x.to(DEVICE), ctx_idx.to(DEVICE), tgt_idx.to(DEVICE)
            loss, preds, targets = model(x, ctx_idx, tgt_idx)
            test_loss_sum += loss.item()
            sim = F.cosine_similarity(preds.reshape(-1, 128), targets.reshape(-1, 128), dim=1).mean().item()
            test_cos_sum += sim
            
    test_losses.append(test_loss_sum / max(1, len(test_loader)))
    cos_sims.append(test_cos_sum / max(1, len(test_loader)))
    print(f"Epoch {epoch:2d} | Train Loss: {train_losses[-1]:.4f} | Test Loss: {test_losses[-1]:.4f} | Cos Sim: {cos_sims[-1]:.4f}")
    
    if test_losses[-1] < best_loss:
        best_loss = test_losses[-1]
        torch.save({'model_state_dict': model.state_dict()}, r"C:\Users\Acer\Downloads\jepa\checkpoints\best_jepa_micro.pt")
        
    scheduler.step()

# Grafik cizimi
plt.figure(figsize=(10,5))
plt.plot(range(1, EPOCHS+1), train_losses, label='Train Loss (Egitim)', marker='o')
plt.plot(range(1, EPOCHS+1), test_losses, label='Test Loss (Test)', marker='s')
plt.title('JEPA Mikro Yamalama (patch_size=50) Loss Egrisi')
plt.xlabel('Epoch')
plt.ylabel('Smooth L1 Loss')
plt.legend()
plt.grid(True)
output_path = r"C:\Users\Acer\.gemini\antigravity-ide\brain\8aa980f3-5dc6-49a2-a0da-b719fbca274e\scratch\loss_curve_micro.png"
plt.savefig(output_path)
print(f"Loss egrisi kaydedildi: {output_path}")
