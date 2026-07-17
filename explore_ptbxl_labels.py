import pandas as pd
import ast

ptbxl_dir = r"C:\Users\Acer\Downloads\jepa\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"

print("1. Etiket Haritasini (scp_statements.csv) Yukluyoruz...")
agg_df = pd.read_csv(f"{ptbxl_dir}/scp_statements.csv", index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1] # Sadece teshis kodlarini al
print("   - Superclass Kategorileri:", agg_df['diagnostic_class'].unique())

def aggregate_diagnostic(y_dict):
    tmp = []
    for key in y_dict.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

print("\n2. EKG Kayitlarini (ptbxl_database.csv) Yukluyoruz...")
Y = pd.read_csv(f"{ptbxl_dir}/ptbxl_database.csv")

# String olarak gelen sozluk yapisini gercek sozluge cevir
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Sozlukteki kodlari ust siniflara (Superclass) donustur
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

print("\n3. Ilk 10 Hastanin Etiketleri:")
with open("labels_output.txt", "w", encoding="utf-8") as f:
    for i in range(10):
        ecg_id = Y.iloc[i]['ecg_id']
        raw_codes = Y.iloc[i]['scp_codes']
        superclass = Y.iloc[i]['diagnostic_superclass']
        
        # Eger NORM (Normal) varsa Saglikli, yoksa Hasta diyelim
        durum = "SAGLIKLI" if "NORM" in superclass else "HASTALIKLI"
        
        f.write(f"Hasta ID {ecg_id}:\n")
        f.write(f"  - Ham SCP Kodlari : {raw_codes}\n")
        f.write(f"  - Hastalik Sinifi : {superclass} -> {durum}\n")
print("Sonuclar labels_output.txt dosyasina kaydedildi.")
