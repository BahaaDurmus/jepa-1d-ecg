import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import wfdb

class PTBXLClassificationDataset(Dataset):
    def __init__(self, data_dir, is_train=True, segment_length=300, max_samples=None):
        self.data_dir = data_dir
        self.segment_length = segment_length
        
        # 1. Superclass etiketlerini oku
        agg_df = pd.read_csv(os.path.join(data_dir, 'scp_statements.csv'), index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]
        
        def aggregate_diagnostic(y_dict):
            tmp = []
            for key in y_dict.keys():
                if key in agg_df.index:
                    tmp.append(agg_df.loc[key].diagnostic_class)
            return list(set(tmp))
            
        # 2. Ana veri setini oku
        Y = pd.read_csv(os.path.join(data_dir, 'ptbxl_database.csv'))
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
        Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)
        
        # Superclass bos olanlari (sadece ritim veya form etiketi olanlari) at
        Y = Y[Y.diagnostic_superclass.apply(lambda x: len(x) > 0)]
        
        # 3. Ikili Siniflandirma Etiketleri (Binary Labels)
        # NORM (Saglikli) ise 0, diger (MI, STTC, CD, HYP) ise 1
        def get_binary_label(classes):
            if 'NORM' in classes and len(classes) == 1:
                return 0 # Sadece Normal
            elif 'NORM' not in classes:
                return 1 # Tamamen Anormal
            else:
                return -1 # Hem normal hem anormal (karisik, bunlari atalim)
                
        Y['binary_label'] = Y.diagnostic_superclass.apply(get_binary_label)
        Y = Y[Y.binary_label != -1]
        
        # 4. Train / Test Ayrimi (PTB-XL onerisi: strat_fold 10 test icindir, 1-9 train icindir)
        if is_train:
            Y = Y[Y.strat_fold != 10]
        else:
            Y = Y[Y.strat_fold == 10]
            
        # Veri setini hizli test etmek icin limit koy
        if max_samples is not None and len(Y) > max_samples:
            Y = Y.sample(n=max_samples, random_state=42)
            
        self.data_info = Y
        print(f"PTB-XL Dataset Yuklendi (is_train={is_train}): {len(self.data_info)} kayit.")
        print(f"  - Saglikli (0): {len(Y[Y.binary_label == 0])}")
        print(f"  - Hastalikli (1): {len(Y[Y.binary_label == 1])}")

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        row = self.data_info.iloc[idx]
        
        # Yüksek çözünürlüklü (500Hz) dosyayi kullan (filename_hr)
        record_path = os.path.join(self.data_dir, row['filename_hr'])
        
        # Sinyali wfdb ile oku
        record = wfdb.rdrecord(record_path)
        sig = record.p_signal # (5000, 12) boyutunda (10 saniye * 500 Hz = 5000)
        
        # 10 saniyeyi (5000 adim), 0.6 saniyelik (300 adim) parcalara (segmentlere) bolelim.
        # 5000 // 300 = 16 adet tam parca cikar. Sondaki 200 adimi atariz.
        num_segments = sig.shape[0] // self.segment_length
        usable_length = num_segments * self.segment_length
        
        sig_chopped = sig[:usable_length, :] # (4800, 12)
        
        # (16, 300, 12) boyutuna yeniden sekillendir
        segments = sig_chopped.reshape(num_segments, self.segment_length, 12)
        
        # JEPA icin kanal boyutunu one alalim: (16, 12, 300)
        segments = segments.transpose(0, 2, 1)
        
        # Her bir segmenti kendi icinde kanal bazinda (Z-score) normalize et
        mean = np.mean(segments, axis=2, keepdims=True)
        std = np.std(segments, axis=2, keepdims=True) + 1e-8
        segments = (segments - mean) / std
        
        x = torch.tensor(segments, dtype=torch.float32) # (16, 12, 300)
        y = torch.tensor(row['binary_label'], dtype=torch.long)
        
        return x, y
