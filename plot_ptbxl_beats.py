import os
import glob
import wfdb
import matplotlib.pyplot as plt
import numpy as np
import random

# Sabit tohum verelim ki ayni guzel atimlari ceksin
random.seed(42)

ptbxl_dir = r"C:\Users\Acer\Downloads\jepa\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
all_heas = glob.glob(os.path.join(ptbxl_dir, "**", "*.hea"), recursive=True)

if not all_heas:
    print("PTB-XL dizininde hic .hea dosyasi bulunamadi!")
    exit()

# Rastgele 4 farkli hastanin kaydini secelim
sample_heas = random.sample(all_heas, min(4, len(all_heas)))

fig, axes = plt.subplots(4, 1, figsize=(10, 10))
fig.suptitle('PTB-XL Veri Setinden Farkli Hastalara Ait Tek-Atim (Single-Beat) Ornekleri', fontsize=14, fontweight='bold')

for i, hea_path in enumerate(sample_heas):
    record_path = hea_path[:-4]
    try:
        record = wfdb.rdrecord(record_path)
        sig = record.p_signal
        fs = record.fs
    except Exception as e:
        print(f"Hata: {record_path} okunamadi. {e}")
        continue
        
    # Bizim mimarimizdeki 0.6 saniyelik "Tek-Atim" penceresini hesapla
    # PTB-XL'de genelde fs=100 veya fs=500 olur. 
    steps = int(fs * 0.6) 
    
    # Atimin net gorunmesi icin sinyalin ortalarindan (mesela 3. saniyeden) 0.6 saniyelik bir parca koparalim
    start_idx = int(fs * 3.5) 
    if start_idx + steps > sig.shape[0]:
        start_idx = 0
        
    # Genellikle kalp ritmini en net gosteren kanal Lead II'dir (indeks 1)
    beat = sig[start_idx:start_idx+steps, 1] 
    
    time_axis = np.linspace(0, 0.6, steps)
    
    axes[i].plot(time_axis, beat, color='darkred', linewidth=2)
    axes[i].set_title(f"Kayit: {os.path.basename(record_path)} | Ornekleme Hizi: {fs}Hz | {steps} Adim", fontsize=10)
    axes[i].set_ylabel("Voltaj (mV)")
    axes[i].grid(True, linestyle='--', alpha=0.7)
    
    # X eksenini sadece son grafikte gosterelim
    if i < 3:
        axes[i].set_xticklabels([])

axes[3].set_xlabel("Zaman (Saniye)", fontsize=12)
plt.tight_layout()

output_path = r"C:\Users\Acer\.gemini\antigravity-ide\brain\8aa980f3-5dc6-49a2-a0da-b719fbca274e\scratch\ptbxl_beats.png"
plt.savefig(output_path, dpi=150)
print(f"Gorsel basariyla olusturuldu: {output_path}")
