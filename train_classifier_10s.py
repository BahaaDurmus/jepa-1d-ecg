import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_ptbxl_class import PTBXLClassificationDataset
from model import JEPA_1D
import time
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
BATCH_SIZE = 16 # Hafiza sikintisi olmamasi icin 16 yaptik
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

# 3. 10 Saniyelik (16 Segmentli) Linear Sınıflandırıcı
class JEPA10sClassifier(nn.Module):
    def __init__(self, embed_dim=128, num_patches=6, num_classes=2):
        super().__init__()
        self.flatten = nn.Flatten()
        # Artik her segment icin (6*128) = 768 boyutlu vektör gelecek.
        self.fc = nn.Linear(embed_dim * num_patches, num_classes)
        
    def forward(self, features):
        # features: (B, 16, 6, 128)
        B, num_segments, num_patches, embed_dim = features.shape
        
        # 1. Adim: Her segmenti ayri ayri flatten yap
        # (B, 16, 6*128) -> (B, 16, 768)
        x = features.view(B, num_segments, num_patches * embed_dim)
        
        # 2. Adim: 16 vagonun (segmentin) Özellik Ortalamasını al (Mean Pooling)
        # Bu sayede 10 saniyenin TAMAMINI temsil eden TEK bir 768 boyutlu vektor elde ederiz!
        # (B, 16, 768) -> (B, 768)
        x = torch.mean(x, dim=1)
        
        # 3. Adim: Sınıflandırma
        return self.fc(x)

classifier = JEPA10sClassifier().to(DEVICE)
optimizer = torch.optim.AdamW(classifier.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

print("\n--- 10 Saniyelik Tam EKG Saha Testi (Mean Pooling) ---")
print("JEPA donduruldu. 10 sn'lik kayit 16 parcaya bolunup islenecek.\n")

for epoch in range(1, EPOCHS + 1):
    start_time = time.time()
    
    # Train
    classifier.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for x, y in train_loader:
        # x boyutu: (B, 16, 12, 300)
        x, y = x.to(DEVICE), y.to(DEVICE)
        B, num_segments, C, L = x.shape
        
        # JEPA'dan gecirmek icin batch ile segmenti katla: (B*16, 12, 300)
        x_folded = x.view(B * num_segments, C, L)
        
        with torch.no_grad():
            features_folded = jepa.target_encoder(x_folded) # (B*16, 6, 128)
            
        # Katlamayi ac: (B, 16, 6, 128)
        features = features_folded.view(B, num_segments, 6, 128)
            
        # Sınıflandır
        optimizer.zero_grad()
        outputs = classifier(features)
        
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * B
        _, predicted = outputs.max(1)
        total += B
        correct += predicted.eq(y).sum().item()
        
    train_loss = total_loss / total
    train_acc = 100. * correct / total
    
    # Eval
    classifier.eval()
    test_loss = 0
    
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            B, num_segments, C, L = x.shape
            
            x_folded = x.view(B * num_segments, C, L)
            features_folded = jepa.target_encoder(x_folded)
            features = features_folded.view(B, num_segments, 6, 128)
            
            outputs = classifier(features)
            loss = criterion(outputs, y)
            test_loss += loss.item() * B
            
            _, predicted = outputs.max(1)
            
            all_targets.extend(y.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            
    test_loss = test_loss / len(all_targets)
    
    # Metrikleri Scikit-learn ile hesapla
    from sklearn.metrics import accuracy_score
    test_acc = accuracy_score(all_targets, all_preds) * 100
    
    elapsed = time.time() - start_time
    print(f"Epoch {epoch:2d}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: %{train_acc:.1f} | Test Loss: {test_loss:.4f} Acc: %{test_acc:.1f} | Sure: {elapsed:.1f}s")

print("\n================ FINAL RAPOR ================")
print(classification_report(all_targets, all_preds, target_names=["SAGLIKLI (0)", "HASTALIKLI (1)"]))

cm = confusion_matrix(all_targets, all_preds)
print("Karmaşıklık Matrisi (Confusion Matrix):")
print(f"  Gerçek Sağlıklı : {cm[0][0]} Doğru | {cm[0][1]} Yanlış (Hasta dendi)")
print(f"  Gerçek Hastalıklı: {cm[1][0]} Yanlış (Sağlıklı dendi) | {cm[1][1]} Doğru")
print("=============================================")
