import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import pearsonr

from dataset_single_beat import SingleBeatDataset, jepa_single_beat_collate_fn
from model import JEPA_1D
from decoder import JEPADecoder_Robust

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

mimic_dir = r"C:\Users\Acer\Downloads\jepa\mimic_small"
test_ds = SingleBeatDataset(data_dir=mimic_dir, is_train=False, segment_length=300)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=True, collate_fn=jepa_single_beat_collate_fn)

jepa = JEPA_1D(seq_len=300, in_chans=12, patch_size=50, embed_dim=128).to(DEVICE)
jepa.load_state_dict(torch.load(r"C:\Users\Acer\Downloads\jepa\checkpoints\best_jepa_single_beat.pt", map_location=DEVICE, weights_only=True)['model_state_dict'])
jepa.eval()

decoder = JEPADecoder_Robust(embed_dim=128, patch_size=50, in_chans=12).to(DEVICE)
decoder.load_state_dict(torch.load(r"C:\Users\Acer\Downloads\jepa\checkpoints\best_decoder.pt", map_location=DEVICE, weights_only=True))
decoder.eval()

print("JEPA HATA ANALIZI (MATEMATIKSEL KARSILASTIRMA)\n" + "-"*50)

for i, (x, ctx_idx, tgt_idx) in enumerate(test_loader):
    if i >= 3:
        break
        
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

    # Matematiksel Analiz Icin Numpy'a cevir
    x_np = x[0, 1].cpu().numpy() # Sadece Lead II
    rec_np = reconstructed_x[0, 1].cpu().numpy()
    
    start = masked_patch_index * 50
    end = (masked_patch_index + 1) * 50
    
    true_patch = x_np[start:end]
    pred_patch = rec_np[start:end]
    
    # 1. Mutlak Hata (MSE ve MAE)
    mse = np.mean((true_patch - pred_patch)**2)
    mae = np.mean(np.abs(true_patch - pred_patch))
    
    # 2. DC Offset (Dikey Kayma / Bazal Kaymasi)
    true_mean = np.mean(true_patch)
    pred_mean = np.mean(pred_patch)
    dc_shift = pred_mean - true_mean
    
    # 3. Sekil Benzerligi (Pearson Korelasyonu - Dikey kaymayi gormezden gelir, sadece sekle bakar)
    # Eger DC kaymasi varsa ama sekil ayniysa, Pearson 1.0 cikar!
    correlation, _ = pearsonr(true_patch, pred_patch)
    
    # 4. Latent Uzay (Yapay Zeka Dili) Benzerligi
    latent_true = targets[0, 0].cpu().numpy()
    latent_pred = preds[0, 0].cpu().numpy()
    latent_cosine = np.dot(latent_true, latent_pred) / (np.linalg.norm(latent_true) * np.linalg.norm(latent_pred))
    
    print(f"\nORNEK {i+1} - Yama {masked_patch_index} ({masked_patch_index*0.1:.1f}s - {(masked_patch_index+1)*0.1:.1f}s)")
    print(f"  * Latent Uzay (Fikir) Benzerligi (Cosine): %{latent_cosine*100:.1f}")
    print(f"  * Sekil ve Ritim Benzerligi (Pearson Cor): %{correlation*100:.1f}")
    print(f"  * Dikey Kayma Miktari (DC Shift)         : {dc_shift:.4f} Volt")
    print(f"  * Ortalama Karesel Hata (MSE)            : {mse:.4f}")
    
    if correlation > 0.8 and abs(dc_shift) > 0.05:
        print("  -> TESPIT: Model sekli MUKEMMEL yakalamis ama dikey eksende (DC Offset) kayma yapmis! Bu bir 'Representation' (Anlam) modelinin dogasidir.")
    elif correlation < 0.5:
        print("  -> TESPIT: Model bu bölgede şekli tamamen yanlış hayal etmiş (Halüsinasyon).")
    else:
        print("  -> TESPIT: Model hem şekli hem de konumu oldukça iyi tahmin etmiş.")
