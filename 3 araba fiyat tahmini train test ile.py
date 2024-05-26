import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as mt

# Veriyi yükle
data = pd.read_csv("C:/Users/ASAF/Desktop/Datasets/Audi_A1_listings.csv")
veri = data.copy()

# Gereksiz sütunları çıkar
veri = veri.drop(columns=["Score","href","MileageRank" ,"PriceRank","PPYRank","PPY","index","Type"])

# Sütun adlarını Türkçeleştir
veri = veri.rename(columns={"Year":"yıl",
                            "Mileage(miles)":"Mil",
                            "Engine":"Motor",
                            "Transmission":"Vites",
                            "Fuel":"Yakıt",
                            "Number_of_Owners":"Sahip sayısı",
                            "Price(£)":"Fiyat"})

# Motor verisini düzenle, 'L' harfini kaldır ve sayısal değerlere dönüştür
veri["Motor"] = veri["Motor"].str.replace("L","")
veri["Motor"] = pd.to_numeric(veri["Motor"], errors='coerce')

# Kategorik verileri one-hot encoding ile sayısal değerlere dönüştür
veri = pd.get_dummies(veri, columns=["Yakıt", "Vites"])

# Hedef değişkeni (Fiyat) ve özellikleri (diğer sütunlar) ayır
y = veri["Fiyat"]
x = veri.drop(columns=["Fiyat"], axis=1)

# Veriyi eğitim ve test setlerine ayır
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Linear Regression modelini oluştur ve eğitim verisi ile eğit
lr = LinearRegression()
model = lr.fit(x_train, y_train)

# Modelin test verisi üzerindeki skoru (R^2) hesapla
print(f'Model skoru (test seti): {model.score(x_test, y_test)}')

# Yeni bir veri seti ile tahmin yap
tahmin = model.predict([[2018, 44000, 1.6, 115, 2400, True, False, True, False]]) 

# Tahmin sonucunu ekrana yazdır
print(f'Tahmin edilen fiyat: {tahmin[0]}')

# Modelin test verisi üzerindeki skoru (R^2) hesapla ve ekrana yazdır
oran = model.score(x_test, y_test)
print(f'Model skoru (test seti): {oran}')
