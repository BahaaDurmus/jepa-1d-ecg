import os
import glob
import wfdb
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt

def butter_highpass_filter(data, cutoff=0.5, fs=500, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data, axis=0)
    return y

class SingleBeatDataset(Dataset):
    def __init__(self, data_dir, is_train=True, segment_length=300, fs=500):
        # segment_length = 300 adim (0.6 saniye) -> Tam 1 atim uzunlugu
        self.segment_length = segment_length
        self.fs = fs

        all_heas = glob.glob(os.path.join(data_dir, "**", "*.hea"), recursive=True)

        patient_records = {}
        for hea in all_heas:
            parts = os.path.normpath(hea).split(os.sep)
            patient_id = None
            for p in parts:
                if p.startswith('p') and len(p) > 5:
                    patient_id = p
                    break
            if patient_id not in patient_records:
                patient_records[patient_id] = []
            patient_records[patient_id].append(hea[:-4])

        patient_ids = list(patient_records.keys())
        patient_ids.sort()
        rng = random.Random(42)
        rng.shuffle(patient_ids)

        split_idx = int(len(patient_ids) * 0.8)
        selected_patients = patient_ids[:split_idx] if is_train else patient_ids[split_idx:]

        self.records = []
        for pid in selected_patients:
            for rec in patient_records[pid]:
                self.records.append(rec)

    def __len__(self):
        # 1 kayittan sadece 1 atim cekmek yerine, veri setini buyutmek icin epoch basina daha cok cekebiliriz.
        # Basitlik ve hiz adina yine kayit sayisi kadar dondurelim, icerde rastgele bir atim koparalim.
        return len(self.records)

    def __getitem__(self, idx):
        record_path = self.records[idx]
        try:
            record = wfdb.rdrecord(record_path)
            sig = record.p_signal
        except Exception:
            return torch.zeros(12, self.segment_length)

        sig = butter_highpass_filter(sig, cutoff=0.5, fs=self.fs)
        sig = np.nan_to_num(sig)

        total_len = sig.shape[0]
        # Toplam sinyalden rastgele 0.6 saniyelik bir atim kopariyoruz
        if total_len > self.segment_length:
            start = random.randint(0, total_len - self.segment_length)
            sig = sig[start:start + self.segment_length, :]
        else:
            pad_len = self.segment_length - total_len
            sig = np.pad(sig, ((0, pad_len), (0, 0)), mode='constant')

        sig = sig.transpose(1, 0)
        return torch.tensor(sig, dtype=torch.float32)


def jepa_single_beat_collate_fn(batch):
    x = torch.stack(batch, dim=0)
    B = x.shape[0]
    
    total_patches = 6 # 300 / 50 = 6 yamacik
    
    # 6 yamadan sadece 1 tanesi rastgele secilip hedefleniyor (maskeleniyor)
    target_indices = []
    context_indices = []
    
    for i in range(B):
        # Her bir ornek icin rastgele 1 yama gizle
        tgt = random.randint(0, 5)
        target_indices.append([tgt])
        
        ctx = [p for p in range(6) if p != tgt]
        context_indices.append(ctx)
        
    tgt_idx = torch.tensor(target_indices)
    ctx_idx = torch.tensor(context_indices)
    
    return x, ctx_idx, tgt_idx
