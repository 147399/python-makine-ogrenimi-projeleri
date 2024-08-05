import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 

# Veriyi CSV dosyasından okuma
data = pd.read_csv("C:/Users/ASAF/Desktop/Datasets/iris.csv")
df = data.copy()

# 'species' sütunundaki kategorik verileri sayısal verilere dönüştürme
df["species"] = df["species"].map({
    "setosa": "0",
    "versicolor": "1",
    "virginica": "2"
})

# Özellikler (X) ve hedef değişkeni (y) ayırma
y = df["species"]
x = df.drop(columns=["species"])

# Veriyi eğitim ve test setlerine ayırma
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=30, train_size=0.7)

# Karar ağacı modelini oluşturma ve eğitim verisi ile eğitme
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# Test verisi üzerindeki başarı oranını hesaplama
score = model.score(x_test, y_test)

# Kullanıcıdan 4 tane değer alma
degerler = []

for x in range(1, 5):
    while True:  # Sonsuz döngü başlatıyoruz
        girdi = input(f"{x}. değeri girin: ")
        # Kullanıcının girdiği değerin sayı olup olmadığını kontrol etme (negatif sayılar da dahil)
        if girdi.replace('.', '', 1).isdigit() or (girdi[0] == '-' and girdi[1:].replace('.', '', 1).isdigit()):
            sayi = float(girdi)  
            degerler.append(sayi) 
            break  
        else:
            print("Lütfen geçerli bir sayı girin.")  # Hatalı giriş için uyarı
            continue

# Girilen değerleri ekrana yazdırma
for i, deger in enumerate(degerler, start=1):
    print(f"{i}. degeriniz : {deger}")

# Girilen değerleri numpy dizisine dönüştürme ve modelin tahmin yapabilmesi için yeniden şekillendirme
girdi_dizisi = np.array(degerler).reshape(1, -1)

# Tahmin yapma
tahmin = model.predict(girdi_dizisi)

# Tahmin sonucuna göre çiçek türünü ekrana yazdırma
if tahmin == '0':
    print("Çiçek: setosa")
elif tahmin == '1':
    print("Çiçek: versicolor")
elif tahmin =="2":
    print("Çiçek: virginica")
else :
    print("Çiçeğin türünü bilmiyoruz")

print(score)

# Veriyi yeniden hazırlama
y2 = df["species"]
x2 = df.drop(columns=["species"])

# Veriyi ölçeklendirme
sc = StandardScaler()
x1 = sc.fit_transform(x2)

# PCA ile boyut indirgeme
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x1)

# PCA sonuçlarını DataFrame'e dönüştürme
principalDf = pd.DataFrame(data=principalComponents, columns=["p1", "p2"])

print(principalDf.head(4))

# Orijinal tür bilgileri ile birleştirme
finalDf = pd.concat([principalDf, data[["species"]]], axis=1)

# Türlere göre veriyi ayırma
setosa = finalDf[data.species=="setosa"]
versicolor = finalDf[data.species=="versicolor"]
virginica = finalDf[data.species=="virginica"]

# PCA sonuçlarını görselleştirme
plt.xlabel("p1")
plt.ylabel("p2")
plt.scatter(setosa["p1"], setosa["p2"], color="blue")
plt.scatter(versicolor["p1"], versicolor["p2"], color="green")
plt.scatter(virginica["p1"], virginica["p2"], color="red")
plt.show()