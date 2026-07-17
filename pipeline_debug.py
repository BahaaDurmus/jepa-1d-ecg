"""
JEPA Pipeline Gozlemcisi (Observability)
-----------------------------------------
Bu script veriyi yukler, modelden gecirir ve her adimda
tensor boyutlarini, voltaj/deger araligini, hasta bilgilerini raporlar.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import MIMICDataset, jepa_collate_fn
from model import JEPA_1D
import torch.nn.functional as F

out = r"C:\Users\Acer\.gemini\antigravity-ide\brain\8aa980f3-5dc6-49a2-a0da-b719fbca274e\scratch"

def stats(name, t):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Shape     : {list(t.shape)}")
    print(f"  Dtype     : {t.dtype}")
    print(f"  Min       : {t.min().item():+.6f}")
    print(f"  Max       : {t.max().item():+.6f}")
    print(f"  Mean      : {t.mean().item():+.6f}")
    print(f"  Std       : {t.std().item():.6f}")
    print(f"  NaN var mi: {torch.isnan(t).any().item()}")
    print(f"  Inf var mi: {torch.isinf(t).any().item()}")

# ----------------------------------------------------------------
# 1. VERI SETI YUKLEME & HASTA ISTATISTIKLERI
# ----------------------------------------------------------------
print("\n" + "#"*60)
print("  ADIM 1: VERI SETI YUKLEME")
print("#"*60)

mimic_dir = r"C:\Users\Acer\Downloads\jepa\mimic_small"
ds = MIMICDataset(data_dir=mimic_dir, is_train=True, segment_length=5000)

print(f"\n  Toplam hasta  : {len(ds.patient_counts)}")
print(f"  Toplam kayit  : {len(ds.records)}")
print(f"\n  Hasta bazli kayit sayilari:")
for pid, cnt in sorted(ds.patient_counts.items(), key=lambda x: -x[1]):
    bar = "#" * min(cnt, 40)
    print(f"    {pid}: {cnt:3d} kayit  {bar}")

# ----------------------------------------------------------------
# 2. BATCH YUKLEME & HAM SINYAL INCELEME
# ----------------------------------------------------------------
print("\n" + "#"*60)
print("  ADIM 2: BATCH YUKLEME & HAM SINYAL")
print("#"*60)

loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=jepa_collate_fn)
x, ctx_idx, tgt_idx = next(iter(loader))

stats("HAM EKG SINYALI (Voltaj, mV)", x)

print(f"\n  Maskeleme Bilgileri:")
print(f"    Toplam yama sayisi : {5000 // 300}")
print(f"    Context yamalari   : {ctx_idx[0].tolist()}")
print(f"    Target yamalari    : {tgt_idx[0].tolist()}")
print(f"    Context suresi     : {len(ctx_idx[0]) * 300 / 500:.1f} sn")
print(f"    Target suresi      : {len(tgt_idx[0]) * 300 / 500:.1f} sn")

# Kanal bazli voltaj analizi
print(f"\n  Kanal Bazli Voltaj Analizi (Batch ortalamalari):")
lead_names = ['I','II','III','aVR','aVF','aVL','V1','V2','V3','V4','V5','V6']
for ch in range(12):
    ch_data = x[:, ch, :]
    print(f"    {lead_names[ch]:>4s}: min={ch_data.min().item():+.4f} mV, max={ch_data.max().item():+.4f} mV, std={ch_data.std().item():.4f}")

# ----------------------------------------------------------------
# 3. MODEL OLUSTURMA & ILERI BESLEME
# ----------------------------------------------------------------
print("\n" + "#"*60)
print("  ADIM 3: MODEL PIPELINE")
print("#"*60)

model = JEPA_1D(seq_len=5000, in_chans=12, patch_size=300, embed_dim=128)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n  Toplam parametre    : {total_params:,}")
print(f"  Egitilebilir param : {trainable:,}")
print(f"  Donuk param (EMA)  : {total_params - trainable:,}")

with torch.no_grad():
    # 3a. Patch Embedding ciktisi
    patch_tokens = model.context_encoder.patch_embed(x)
    stats("3a. PATCH EMBEDDING CIKTISI (Voltaj -> Token)", patch_tokens)

    # 3b. Context Encoder
    ctx_reps = model.context_encoder(x, mask_indices=ctx_idx)
    stats("3b. CONTEXT ENCODER CIKTISI", ctx_reps)

    # 3c. Target Encoder
    full_targets = model.target_encoder(x, mask_indices=None)
    B, N_all, D = full_targets.shape
    exp_tgt = tgt_idx.unsqueeze(-1).expand(-1, -1, D)
    target_reps = full_targets.gather(1, exp_tgt)
    target_reps = F.layer_norm(target_reps, (D,))
    stats("3c. TARGET ENCODER CIKTISI (Gercek Hedef)", target_reps)

    # 3d. Predictor
    preds = model.predictor(ctx_reps, ctx_idx, tgt_idx)
    stats("3d. PREDICTOR CIKTISI (Modelin Tahmini)", preds)

    # 3e. Loss
    loss = F.smooth_l1_loss(preds, target_reps)
    print(f"\n  LOSS (Smooth L1): {loss.item():.6f}")

    # Tahmin vs Hedef benzerlik
    cos_sim = F.cosine_similarity(preds.reshape(-1, 128), target_reps.reshape(-1, 128), dim=1)
    print(f"  Cosine Similarity (Tahmin vs Hedef): {cos_sim.mean().item():.4f}")

# ----------------------------------------------------------------
# 4. GORSEL: PIPELINE AKISI
# ----------------------------------------------------------------
print("\n" + "#"*60)
print("  ADIM 4: GORSELLESTIRME")
print("#"*60)

fig, axes = plt.subplots(5, 1, figsize=(14, 16))

# 4a. Ham sinyal + maskeler
sig = x[0, 0, :].numpy()
t = np.arange(len(sig)) / 500.0
axes[0].plot(t, sig, color='black', linewidth=0.5)
ps = 300
for c in ctx_idx[0].numpy():
    axes[0].axvspan((c*ps)/500, ((c+1)*ps)/500, color='#2196F3', alpha=0.08)
for ti in tgt_idx[0].numpy():
    axes[0].axvspan((ti*ps)/500, ((ti+1)*ps)/500, color='#F44336', alpha=0.25)
axes[0].set_title(f"1. Ham Sinyal (Lead I) | Min={sig.min():.3f} mV, Max={sig.max():.3f} mV | Mavi=Context, Kirmizi=Target")
axes[0].set_ylabel("mV")

# 4b. Patch Tokens
im1 = axes[1].imshow(patch_tokens[0].numpy().T, aspect='auto', cmap='coolwarm', interpolation='nearest')
axes[1].set_title(f"2. Patch Embedding Ciktisi | {patch_tokens.shape[1]} yama x {patch_tokens.shape[2]} dim | [{patch_tokens.min():.2f}, {patch_tokens.max():.2f}]")
fig.colorbar(im1, ax=axes[1], shrink=0.8)

# 4c. Context temsili
im2 = axes[2].imshow(ctx_reps[0].numpy().T, aspect='auto', cmap='viridis', interpolation='nearest')
axes[2].set_title(f"3. Context Encoder | {ctx_reps.shape[1]} yama x {ctx_reps.shape[2]} dim | [{ctx_reps.min():.2f}, {ctx_reps.max():.2f}]")
fig.colorbar(im2, ax=axes[2], shrink=0.8)

# 4d. Target (gercek cevap)
im3 = axes[3].imshow(target_reps[0].numpy().T, aspect='auto', cmap='magma', interpolation='nearest')
axes[3].set_title(f"4. Hedef (Gercek Cevap) | {target_reps.shape[1]} yama x {target_reps.shape[2]} dim | [{target_reps.min():.2f}, {target_reps.max():.2f}]")
fig.colorbar(im3, ax=axes[3], shrink=0.8)

# 4e. Tahmin
im4 = axes[4].imshow(preds[0].numpy().T, aspect='auto', cmap='magma', interpolation='nearest')
axes[4].set_title(f"5. Predictor Tahmini | cos_sim={cos_sim.mean():.3f} | [{preds.min():.2f}, {preds.max():.2f}]")
fig.colorbar(im4, ax=axes[4], shrink=0.8)

plt.tight_layout()
plt.savefig(os.path.join(out, "pipeline_full.png"), dpi=150)
print(f"  Kaydedildi: pipeline_full.png")

print("\n" + "="*60)
print("  PIPELINE TESTI TAMAMLANDI")
print("="*60)
