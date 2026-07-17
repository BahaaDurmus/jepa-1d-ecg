"""
JEPA 1D Egitim Scripti (mimic_small EKG)
"""
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import MIMICDataset, jepa_collate_fn
from model import JEPA_1D

# --- Ayarlar ---
EPOCHS = 20
BATCH_SIZE = 8
LR = 1e-4
EMA_START = 0.996
EMA_END = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = r"C:\Users\Acer\Downloads\jepa\checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Veri ---
mimic_dir = r"C:\Users\Acer\Downloads\jepa\mimic_small"
train_ds = MIMICDataset(data_dir=mimic_dir, is_train=True, segment_length=5000)
test_ds = MIMICDataset(data_dir=mimic_dir, is_train=False, segment_length=5000)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=jepa_collate_fn, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=jepa_collate_fn, drop_last=True)

print(f"Device: {DEVICE}")
print(f"Train: {len(train_ds)} kayit | Test: {len(test_ds)} kayit")
print(f"Batch boyutu: {BATCH_SIZE} | Epoch: {EPOCHS}")

# --- Model ---
model = JEPA_1D(seq_len=5000, in_chans=12, patch_size=100, embed_dim=128).to(DEVICE)

# Sadece context_encoder + predictor egitilir, target_encoder EMA ile guncellenir
optimizer = torch.optim.AdamW(
    list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
    lr=LR, weight_decay=0.05
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

total_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Toplam param: {total_params:,} | Egitilebilir: {trainable:,}")

# --- Egitim Dongusu ---
best_test_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    # EMA momentum: epoch ilerledikce 0.996 -> 1.0 yaklasir
    ema_m = EMA_START + (EMA_END - EMA_START) * (epoch - 1) / max(EPOCHS - 1, 1)

    # --- TRAIN ---
    model.train()
    train_loss_sum = 0.0
    train_steps = 0

    for batch_idx, (x, ctx_idx, tgt_idx) in enumerate(train_loader):
        x = x.to(DEVICE)
        ctx_idx = ctx_idx.to(DEVICE)
        tgt_idx = tgt_idx.to(DEVICE)

        loss, preds, targets = model(x, ctx_idx, tgt_idx)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # EMA guncelle
        model.update_target_encoder(m=ema_m)

        train_loss_sum += loss.item()
        train_steps += 1

    train_loss_avg = train_loss_sum / max(train_steps, 1)

    # --- TEST ---
    model.eval()
    test_loss_sum = 0.0
    test_cos_sum = 0.0
    test_steps = 0

    with torch.no_grad():
        for x, ctx_idx, tgt_idx in test_loader:
            x = x.to(DEVICE)
            ctx_idx = ctx_idx.to(DEVICE)
            tgt_idx = tgt_idx.to(DEVICE)

            loss, preds, targets = model(x, ctx_idx, tgt_idx)
            cos_sim = F.cosine_similarity(
                preds.reshape(-1, 128), targets.reshape(-1, 128), dim=1
            ).mean()

            test_loss_sum += loss.item()
            test_cos_sum += cos_sim.item()
            test_steps += 1

    test_loss_avg = test_loss_sum / max(test_steps, 1)
    test_cos_avg = test_cos_sum / max(test_steps, 1)

    # --- Log ---
    lr_now = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch:2d}/{EPOCHS} | "
          f"Train Loss: {train_loss_avg:.4f} | "
          f"Test Loss: {test_loss_avg:.4f} | "
          f"Cos Sim: {test_cos_avg:+.4f} | "
          f"EMA: {ema_m:.4f} | "
          f"LR: {lr_now:.6f}")

    # --- Checkpoint ---
    if test_loss_avg < best_test_loss:
        best_test_loss = test_loss_avg
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_loss': test_loss_avg,
            'cos_sim': test_cos_avg,
        }, os.path.join(CHECKPOINT_DIR, "best_jepa.pt"))
        print(f"  -> En iyi model kaydedildi (Test Loss: {test_loss_avg:.4f})")

    scheduler.step()

print(f"\nEgitim tamamlandi. En iyi Test Loss: {best_test_loss:.4f}")
