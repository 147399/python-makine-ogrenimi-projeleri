import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Veri setinin belirtilen dosya yolundan okunması
data = pd.read_csv("C:/Users/ASAF/Desktop/Datasets/Audi_A1_listings.csv")

# Orijinal veri setinin kopyalanması
veri = data.copy()

# İlgili sütunların silinmesi ve yeniden adlandırılması
veri = veri.drop(columns=["Type", "index", "href", "MileageRank", "PriceRank", "PPYRank", "Score", "Number_of_Owners"], axis=1)
veri = veri.rename(columns={
    "Year": "Yıl",
    "Mileage(miles)": "Mil",
    "Engine": "Motor",
    "Transmission": "Şanzıman",
    "Fuel": "Yakıtı",
    "Price(£)": "Fiyat"
})

# 'Motor' sütununun temizlenmesi ve sayısal türe dönüştürülmesi
veri["Motor"] = veri["Motor"].str.replace("L", "")
veri["Motor"] = pd.to_numeric(veri["Motor"], errors='coerce')

# 'Şanzıman' ve 'Yakıtı' sütunlarının one-hot encoding işlemi ile dönüştürülmesi
veri = pd.get_dummies(veri, columns=["Şanzıman","Yakıtı"])

# Bağımlı değişken (hedef) ve bağımsız değişkenlerin ayrılması
y = veri["Fiyat"]
x = veri.drop(columns=["Fiyat"], axis=1)

# Lineer regresyon modelinin oluşturulması ve eğitilmesi
lr = LinearRegression()
model = lr.fit(x, y)

# Yeni bir veri noktası için tahmin yapılması
tahmin = model.predict([[2018, 44000, 1.6, 115, 2400, True, False, True, False]]) 
# tahmin degerinin dogruluk yüzdesi
oran = model.score(x,y)
print(tahmin)
print(oran)
