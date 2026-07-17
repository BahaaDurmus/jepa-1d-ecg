import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_single_beat import SingleBeatDataset, jepa_single_beat_collate_fn
from model import JEPA_1D
from decoder import JEPADecoder_1D, JEPADecoder_Conv1D, JEPADecoder_Robust

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Sabit tohum, her seferinde ayni seyi gostersin diye
torch.manual_seed(42)

mimic_dir = r"C:\Users\Acer\Downloads\jepa\mimic_small"
test_ds = SingleBeatDataset(data_dir=mimic_dir, is_train=False, segment_length=300)
# Shuffle=True yapiyoruz ki her calistirdigimizda baska EKG'ler (baska hastalar) gelsin
test_loader = DataLoader(test_ds, batch_size=1, shuffle=True, collate_fn=jepa_single_beat_collate_fn)

# 1. Modelleri Yukle
jepa = JEPA_1D(seq_len=300, in_chans=12, patch_size=50, embed_dim=128).to(DEVICE)
jepa.load_state_dict(torch.load(r"C:\Users\Acer\Downloads\jepa\checkpoints\best_jepa_single_beat.pt", map_location=DEVICE, weights_only=True)['model_state_dict'])
jepa.eval()

decoder = JEPADecoder_Robust(embed_dim=128, patch_size=50, in_chans=12).to(DEVICE)
decoder.load_state_dict(torch.load(r"C:\Users\Acer\Downloads\jepa\checkpoints\best_decoder.pt", map_location=DEVICE, weights_only=True))
decoder.eval()

import torch.nn.functional as F

# 3 FARKLI HASTA/ORNEK ICIN DONGU
for i, (x, ctx_idx, tgt_idx) in enumerate(test_loader):
    if i >= 3:
        break # Sadece 3 ornek goster
        
    with torch.no_grad():
        x = x.to(DEVICE)
        ctx_idx = ctx_idx.to(DEVICE)
        tgt_idx = tgt_idx.to(DEVICE)
        
        true_latents = jepa.target_encoder(x)
        true_latents = F.layer_norm(true_latents, (true_latents.shape[-1],))
        
        loss, preds, targets = jepa(x, ctx_idx, tgt_idx)
        
        hybrid_latents = true_latents.clone()
        masked_patch_index = tgt_idx[0, 0].item()
        hybrid_latents[0, masked_patch_index, :] = preds[0, 0, :]
        
        reconstructed_x = decoder(hybrid_latents)

    x_np = x[0].cpu().numpy()
    rec_np = reconstructed_x[0].cpu().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    fig.suptitle(f"Örnek {i+1} - JEPA ile EKG Bosluk Doldurma (Inpainting)\nGizlenen Bolge: {masked_patch_index}. Yama ({masked_patch_index*0.1:.1f}sn - {(masked_patch_index+1)*0.1:.1f}sn arasi)", fontsize=14, fontweight='bold')

    channel = 1
    time_axis = np.linspace(0, 0.6, 300)

    # A) Orjinal Sinyal
    axes[0].plot(time_axis, x_np[channel], color='black', label="Orjinal EKG (Gercek)")
    axes[0].axvspan(masked_patch_index*0.1, (masked_patch_index+1)*0.1, color='red', alpha=0.3, label="Modele Gizlenen 'Kor Nokta'")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Orijinal Sinyal (Eksiksiz)")
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # B) Boslugu Doldurulmus Sinyal (Inpainting)
    infilled_np = np.copy(x_np)
    start_step = masked_patch_index * 50
    end_step = (masked_patch_index + 1) * 50
    infilled_np[:, start_step:end_step] = rec_np[:, start_step:end_step]

    axes[1].plot(time_axis, infilled_np[channel], color='black', label="Bildiğimiz Kısımlar (Gerçek)")
    axes[1].plot(time_axis[start_step:end_step], infilled_np[channel, start_step:end_step], color='blue', linewidth=2, label="AI Tarafindan Tamamlanan Kısım")
    axes[1].axvspan(masked_patch_index*0.1, (masked_patch_index+1)*0.1, color='red', alpha=0.3)
    axes[1].legend(loc="upper right")
    axes[1].set_title("Bildiğimiz Sinyal + Yapay Zekanın Tamamladığı Sinyal")
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # C) Ust Uste Karsilastirma (Sadece Kör Nokta)
    axes[2].plot(time_axis[start_step:end_step], x_np[channel, start_step:end_step], color='black', alpha=0.5, label="Gerçekte Olan", linewidth=4)
    axes[2].plot(time_axis[start_step:end_step], rec_np[channel, start_step:end_step], color='blue', linestyle='--', label="Yapay Zekanın Tahmini", linewidth=2)
    axes[2].axvspan(masked_patch_index*0.1, (masked_patch_index+1)*0.1, color='red', alpha=0.2)
    axes[2].legend(loc="upper right")
    axes[2].set_title("Sadece Kör Noktaya (Yakından) Bakış")
    axes[2].grid(True, linestyle='--', alpha=0.5)
    axes[2].set_xlabel("Zaman (Saniye)", fontsize=12)

    plt.tight_layout()
    out_path = rf"C:\Users\Acer\.gemini\antigravity-ide\brain\8aa980f3-5dc6-49a2-a0da-b719fbca274e\scratch\reconstruction_{i+1}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Ornek {i+1} gorsellestirme kaydedildi: {out_path}")
