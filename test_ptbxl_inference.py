import os
import glob
import wfdb
import random
import torch
import numpy as np
import torch.nn.functional as F
from scipy.signal import butter, filtfilt, resample
from model import JEPA_1D

random.seed(42)

def butter_highpass_filter(data, cutoff=0.5, fs=500, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data, axis=0)
    return y

ptbxl_dir = r"C:\Users\Acer\Downloads\jepa\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
all_heas = glob.glob(os.path.join(ptbxl_dir, "**", "*.hea"), recursive=True)

# Test etmek icin rastgele 15 kayit secelim
sample_heas = random.sample(all_heas, min(15, len(all_heas)))

print("Model Yukleniyor (MIMIC-small'da Egitilen Tek-Atim Modeli)...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = JEPA_1D(seq_len=300, in_chans=12, patch_size=50, embed_dim=128).to(DEVICE)
ckpt = torch.load(r"C:\Users\Acer\Downloads\jepa\checkpoints\best_jepa_single_beat.pt", map_location=DEVICE, weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

print("\nPTB-XL Verileri Uzerinde 'Zero-Shot' (Hic Gormedigi Veri) Testi Sonuclari:")
print("-" * 65)
print(f"{'PTB-XL Kaydi':<20} | {'Gizlenen Yama':<15} | {'Loss':<10} | {'Benzerlik (%)':<10}")
print("-" * 65)

total_loss = 0
total_cos = 0
valid_samples = 0

with torch.no_grad():
    for hea_path in sample_heas:
        record_path = hea_path[:-4]
        try:
            record = wfdb.rdrecord(record_path)
            sig = record.p_signal
            fs = record.fs
        except:
            continue
            
        # Sinyali parazitten temizle
        sig = butter_highpass_filter(sig, cutoff=0.5, fs=fs)
        sig = np.nan_to_num(sig)
        
        # Modelimiz 500 Hz'e gore tasarlandigi (300 adim = 0.6s) icin, PTB-XL 100 Hz ise ornekleme hizini 500'e cikartiyoruz.
        if fs != 500:
            target_length = int(sig.shape[0] * (500 / fs))
            sig = resample(sig, target_length, axis=0)
        
        # Ortadan rastgele 0.6 saniyelik (300 adim) tam bir atim penceresi kopar
        total_len = sig.shape[0]
        steps = 300
        if total_len > steps:
            # Kenar hatalarini onlemek icin ortalardan secelim
            start_idx = random.randint(int(500 * 2), total_len - steps)
            beat = sig[start_idx:start_idx+steps, :]
        else:
            continue
            
        # Tensoru PyTorch'a uygun [1, 12, 300] formatina getir
        beat = beat.transpose(1, 0)
        x = torch.tensor(beat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # 6 yamadan (0,1,2,3,4,5) rastgele 1'ini gizle
        tgt = random.randint(0, 5)
        ctx = [p for p in range(6) if p != tgt]
        
        ctx_idx = torch.tensor([ctx]).to(DEVICE)
        tgt_idx = torch.tensor([[tgt]]).to(DEVICE)
        
        # Modeli tahmin icin calistir
        loss, preds, targets = model(x, ctx_idx, tgt_idx)
        cos_sim = F.cosine_similarity(preds.reshape(-1, 128), targets.reshape(-1, 128), dim=1).mean().item()
        
        total_loss += loss.item()
        total_cos += cos_sim
        valid_samples += 1
        
        rec_name = os.path.basename(record_path)
        print(f"{rec_name:<20} | Yama No {tgt:<9} | {loss.item():.4f}     | %{cos_sim*100:.1f}")

if valid_samples > 0:
    print("-" * 65)
    print(f"{'ORTALAMA':<20} | {'':<15} | {total_loss/valid_samples:.4f}     | %{(total_cos/valid_samples)*100:.1f}")
