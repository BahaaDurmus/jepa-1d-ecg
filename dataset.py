import os
import glob
import wfdb
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt

def butter_highpass_filter(data, cutoff=0.5, fs=500, order=3):
    """Baseline Wander temizlemek icin High-Pass Filter"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data, axis=0)
    return y

class MIMICDataset(Dataset):
    def __init__(self, data_dir, is_train=True, segment_length=5000, fs=500):
        self.segment_length = segment_length
        self.fs = fs

        all_heas = glob.glob(os.path.join(data_dir, "**", "*.hea"), recursive=True)

        # Patient-wise split
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
        self.patient_map = {}  # kayit -> hasta eslestirmesi
        for pid in selected_patients:
            for rec in patient_records[pid]:
                self.records.append(rec)
                self.patient_map[rec] = pid

        self.patient_counts = {}
        for pid in selected_patients:
            self.patient_counts[pid] = len(patient_records[pid])

    def __len__(self):
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
        if total_len > self.segment_length:
            start = random.randint(0, total_len - self.segment_length)
            sig = sig[start:start + self.segment_length, :]
        else:
            pad_len = self.segment_length - total_len
            sig = np.pad(sig, ((0, pad_len), (0, 0)), mode='constant')

        sig = sig.transpose(1, 0)
        return torch.tensor(sig, dtype=torch.float32)


def jepa_collate_fn(batch):
    x = torch.stack(batch, dim=0)
    B = x.shape[0]
    total_patches = 5000 // 100

    tgt_len = random.choice(range(15, 21))  # 15-20 target patches (30%-40% mask ratio)
    start_idx = random.randint(0, total_patches - tgt_len)
    tgt_idx = torch.arange(start_idx, start_idx + tgt_len).unsqueeze(0).repeat(B, 1)

    avail_ctx = [i for i in range(total_patches) if i < start_idx or i >= start_idx + tgt_len]
    ctx_idx = torch.tensor(sorted(avail_ctx)).unsqueeze(0).repeat(B, 1)

    return x, ctx_idx, tgt_idx
