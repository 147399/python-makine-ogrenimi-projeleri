import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Verisetini oku
data = pd.read_csv("C:/Users/ASAF/Desktop/Datasets/insurance.csv")

# Orijinal verisetinin bir kopyasını oluştur
veri = data.copy()

# Kategorik değişkenleri dummy değişkenlere dönüştür
veri = pd.get_dummies(veri, columns=["smoker", "region", "sex"], drop_first=True)

# Hedef değişkeni (y) ve özellikleri (x) tanımla
y = veri["charges"]
x = veri.drop(columns="charges", axis=1)

# Veriyi eğitim ve test setlerine ayır
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=22)

# Doğrusal Regresyon modelini başlat
lr = LinearRegression()

# Modeli eğitim verisiyle eğit
model = lr.fit(x_train, y_train)

# Modelin test verisi üzerindeki R^2 skorunu hesapla
score = model.score(x_test, y_test)

# Random Forest Regressor modelini başlat
rf = RandomForestRegressor()

# Random Forest modelini eğitim verisiyle eğit
model2 = rf.fit(x_train, y_train)

# Random Forest modelinin test verisi üzerindeki R^2 skorunu hesapla
score2 = model2.score(x_test, y_test)

# Yeni bir veri noktası için Doğrusal Regresyon modeli ile tahmin yap
# [age, bmi, children, smoker_yes, region_northwest, region_southeast, region_southwest, sex_male]
tahmin = model.predict([[20, 25, 0, False, False, False, True, True]])
print("Yeni veri noktası için tahmin:", tahmin)

deneme = np.array(x.iloc[1033])

tahmin2 = model2.predict([deneme])
print(tahmin2)

gercek = y.iloc[1033]
print(gercek)


















