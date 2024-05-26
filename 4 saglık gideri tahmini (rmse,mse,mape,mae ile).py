import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Veriyi yükle
data = pd.read_csv("C:/Users/ASAF/Desktop/Datasets/insurance.csv")
veri = data.copy()

# Kategorik verileri one-hot encoding ile sayısal değerlere dönüştür
veri = pd.get_dummies(veri, columns=["sex", "smoker", "region"], drop_first=True)

# Hedef değişkeni (charges) ve özellikleri ayır
y = veri["charges"]
x = veri.drop(columns="charges", axis=1)

# Linear Regression modelini oluştur ve eğit
lr = LinearRegression()
model = lr.fit(x, y)

# Örnek bir veri ile tahmin yap
tahmin = model.predict([[19, 34, 2, False, True, False, True, False]])
oran = model.score(x, y)

# Tahmin sonucunu ve modelin skoru (R^2) ekrana yazdır
print(tahmin)
print(oran)

# Tahmin edilen ve gerçek değerleri içeren bir DataFrame oluştur
veri_hata = pd.DataFrame()
veri_hata["y"] = y
y_tahmin = model.predict(x)
veri_hata["tahmin"] = y_tahmin

# Tahmin ve gerçek değerleri ekrana yazdır
print(veri_hata)
print(y_tahmin)

# Hata metriklerini hesapla
mae = mean_absolute_error(y, y_tahmin)
mape = mean_absolute_percentage_error(y, y_tahmin)
mse = mean_squared_error(y, y_tahmin)

# Hata metriklerini ekrana yazdır
print("mae = {}\n mape = {}\n mse ={}".format(mae, mape, mse))
