import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


# Veriyi okuma
data = pd.read_csv("C:/Users/ASAF/Desktop/Datasets/heart.csv")
df = data.copy()  

# Hedef ve özelliklerin ayrılması
y = df["output"]  
x = df.drop(columns=["output"])  

# Özelliklerin standartlaştırılması
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# 1'den 20'ye kadar n_splits değerlerini kullanarak çapraz doğrulama
for n_splits in range(2, 21): 
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1) 
    model = LogisticRegression(max_iter=100)  
    
    # Sonuçları hesapla
    results = cross_val_score(model, x_scaled, y, cv=kf, scoring='accuracy')  # Modelin çapraz doğrulama sonuçlarını hesapla
    
    # Sonuçları yazdır
    print(f"\n{n_splits}. iterasyon:")  
    for fold_idx, score in enumerate(results, start=1):
        print(f"  Fold {fold_idx}: {score:.4f}")  
    print(f"  Ortalama doğruluk skoru: {results.mean():.4f}")  

# Tekrar veri okuma ve kopyalama (bu kodun amacı muhtemelen hata yapmış olabilir, tekrar okunmasına gerek yok)
data = pd.read_csv("C:/Users/ASAF/Desktop/Datasets/heart.csv")
df = data.copy()

# Hedef ve özelliklerin tekrar ayrılması
y = df["output"]
x = df.drop(columns=["output"])

# Özelliklerin standartlaştırılması
scaler = StandardScaler()
x_scaler = scaler.fit_transform(x)  

# 2'den 19'a kadar n_splits değerlerini kullanarak çapraz doğrulama
for n_splits in range(2, 20):  
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)  
    model = LogisticRegression(max_iter=100)  

    results = cross_val_score(model, x_scaler, y, cv=kf, scoring="accuracy")  # Modelin çapraz doğrulama sonuçlarını hesapla
    
    # Sonuçları yazdır
    print(f"n_splits: {n_splits} - Ortalama doğruluk skoru: {results.mean()}") 

model1 = LogisticRegression(x_scaled,y)
score = model1.score(x_scaled,y)

# Özelliklerin standartlaştırılması
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# KFold oluşturma (8 katman)
kf = KFold(n_splits=8, shuffle=True, random_state=1)

# 8. iterasyon için eğitim ve test setlerini ayırma
for fold_idx, (train_index, test_index) in enumerate(kf.split(x_scaled)):
    if fold_idx == 7:  # 8. iterasyon (0 tabanlı indeksleme)
        x_train, x_test = x_scaled[train_index], x_scaled[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Model oluşturma ve eğitme
        model = LogisticRegression(max_iter=100)
        model.fit(x_train, y_train)
        
        # Tahmin yapma
        y_pred = model.predict(x_test)
        
        # Sonuçları yazdırma
        print(f"8. iterasyon sonuçları:")
        print(f"Doğruluk skoru: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Sınıflandırma raporu:\n{classification_report(y_test, y_pred)}")
        
        break

sns.pairplot(df)
plt.show()

