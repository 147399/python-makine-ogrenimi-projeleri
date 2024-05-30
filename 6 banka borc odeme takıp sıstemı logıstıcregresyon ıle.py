import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Veriyi yükle
data = pd.read_csv("C:/Users/ASAF/Desktop/Datasets/UCI_Credit_Card.csv")
veri = data.copy()  # Orijinal veriyi yedekle

# Bağımlı ve bağımsız değişkenleri ayır
y = veri["default.payment.next.month"]  # Hedef değişken (bağımlı değişken)
x = veri.drop(columns=["default.payment.next.month"], axis=1)  # Özellikler (bağımsız değişkenler)

# Veriyi eğitim ve test setlerine ayır
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.77, random_state=6)

# Logistic Regression modelini oluştur ve eğit
log = LogisticRegression(max_iter=10000)  # Daha fazla iterasyona izin veriyoruz
model = log.fit(x_train, y_train)

# Modelin test seti üzerindeki skorunu hesapla
skor = model.score(x_test, y_test)
print(f"Model skoru: {skor}")

# Test etmek için belirli bir veri noktası al
denemex = np.array(x.iloc[1903])  # 1903. satırdaki veriyi al

# Seçilen veri noktası için tahmin yap
tahmin = model.predict([denemex])  # Veri noktasını modelde kullanarak tahmin yap
cevap = y.iloc[1903]  # Gerçek değeri al

# Tahmin ve gerçek değeri yazdır
print(f" tahmin = {tahmin}\n gercek deger = {cevap}")