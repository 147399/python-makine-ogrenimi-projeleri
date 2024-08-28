import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Veriyi CSV dosyasından yükle
data = pd.read_csv("C:/Users/ASAF/Desktop/Datasets/diabetes.csv")

# Veri çerçevesinin bir kopyasını oluştur
df = data.copy()

# 'BMI' sütununu veri setinden çıkar
df = df.drop(columns=["BMI"])

# 'Outcome' sütununu hedef değişken (y) olarak ayarla
y = df["Outcome"].values.reshape(-1, 1)

# 'Outcome' sütunu hariç tüm sütunları özellikler (x) olarak ayarla
x = df.drop(columns=["Outcome"]).values

# Veriyi eğitim ve test setlerine ayır (eğitim için %70, test için %30)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=20, train_size=0.7)

# Veriyi standardize etmek için StandardScaler nesnesi oluştur
sc = StandardScaler()
 
# Eğitim verilerini standardize et 
x_train1 = sc.fit_transform(x_train)

# Test verilerini standardize et
x_test1 = sc.fit_transform(x_test)

# KNN sınıflandırıcı nesnesi oluştur (komşu sayısı 3)
knn = KNeighborsClassifier(n_neighbors=3)

# Modeli eğitim verileriyle eğit
model = knn.fit(x_train1, y_train)

# Modelin test setindeki doğruluğunu hesapla ve yazdır
score = model.score(x_test1, y_test)
print(score)

# Belirli bir veri noktası için tahminde bulun (örneğin: [0, 137, 40, 35, 168, 2.228, 33])
tahmin = model.predict([[0, 137, 40, 35, 168, 2.228, 33]])
print(tahmin)

# 'Outcome' sütunundaki belirli bir satırı yazdır
reel = df.iloc[4]["Outcome"]
print(reel)

# StandardScaler nesnesini pickle dosyasına kaydet
scaler_dosyası = "sc.pickle"
pickle.dump(sc, open(scaler_dosyası, "wb"))

# KNN modelini pickle dosyasına kaydet
model1 = "knn.model.pickle"
pickle.dump(knn, open(model1, "wb"))
#ANP