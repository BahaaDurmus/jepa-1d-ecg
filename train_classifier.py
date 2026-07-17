import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_ptbxl_class import PTBXLClassificationDataset
from model import JEPA_1D
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3

# 1. Veri Setlerini Yukle (Hizli test icin train=2000, test=500 ile limitliyoruz)
ptbxl_dir = r"C:\Users\Acer\Downloads\jepa\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"

train_ds = PTBXLClassificationDataset(data_dir=ptbxl_dir, is_train=True, max_samples=2000)
test_ds = PTBXLClassificationDataset(data_dir=ptbxl_dir, is_train=False, max_samples=500)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# 2. Dondurulmus JEPA (Feature Extractor) Yukle
jepa = JEPA_1D(seq_len=300, in_chans=12, patch_size=50, embed_dim=128).to(DEVICE)
jepa.load_state_dict(torch.load(r"C:\Users\Acer\Downloads\jepa\checkpoints\best_jepa_single_beat.pt", map_location=DEVICE, weights_only=True)['model_state_dict'])

# JEPA'NIN AGIRLIKLARINI DONDUR (FROZEN) - Sadece ogrendigi dili kullanacagiz
for param in jepa.parameters():
    param.requires_grad = False
jepa.eval()

# 3. Basit Linear Sınıflandırıcı (Linear Probing)
# JEPA (B, 6, 128) cikti uretir. Bunu (B, 768) yapip tek katmana veriyoruz.
class JEPALinearClassifier(nn.Module):
    def __init__(self, embed_dim=128, num_patches=6, num_classes=2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(embed_dim * num_patches, num_classes)
        
    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

classifier = JEPALinearClassifier().to(DEVICE)
optimizer = torch.optim.AdamW(classifier.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

print("\nJEPA Saha Testi (Downstream Classification) Basliyor...")
print("JEPA donduruldu, sadece sondaki 1-katmanli lineer ag egitiliyor.\n")

for epoch in range(1, EPOCHS + 1):
    start_time = time.time()
    
    # Train
    classifier.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        # JEPA'dan ozellik (feature) cikar
        with torch.no_grad():
            features = jepa.target_encoder(x) # (B, 6, 128)
            
        # Sınıflandır
        optimizer.zero_grad()
        outputs = classifier(features)
        
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
        
    train_loss = total_loss / total
    train_acc = 100. * correct / total
    
    # Eval
    classifier.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            features = jepa.target_encoder(x)
            outputs = classifier(features)
            
            loss = criterion(outputs, y)
            test_loss += loss.item() * x.size(0)
            
            _, predicted = outputs.max(1)
            test_total += y.size(0)
            test_correct += predicted.eq(y).sum().item()
            
    test_loss = test_loss / test_total
    test_acc = 100. * test_correct / test_total
    
    elapsed = time.time() - start_time
    print(f"Epoch {epoch:2d}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: %{train_acc:.1f} | Test Loss: {test_loss:.4f} Acc: %{test_acc:.1f} | Sure: {elapsed:.1f}s")

print("\nSaha Testi Tamamlandi!")
