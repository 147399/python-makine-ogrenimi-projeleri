# Gerekli kütüphaneleri import et
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.linear_model import LinearRegression

# Verisetini oku
data = pd.read_csv("C:/Users/ASAF/Desktop/Datasets/UCI_Credit_Card.csv")

# Orijinal verisetinin bir kopyasını oluştur
veri = data.copy()

# Modelde kullanılmayacak olan sütunları çıkar
veri = veri.drop(columns=["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
                          "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","ID"])

# Hedef değişkeni (y) ve özellikleri (x) tanımla
y = veri["default.payment.next.month"]
x = veri.drop(columns="default.payment.next.month", axis=1)

# Veriyi eğitim ve test setlerine ayır
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.77, random_state=42)

# Doğrusal Regresyon modelini başlat
lr = LinearRegression()

# Modeli eğitim verisiyle eğit
model = lr.fit(x_train, y_train)

# Modelin test verisi üzerindeki R^2 skorunu yazdır
print("Model Score (R^2):", model.score(x_test, y_test))

# Yeni bir veri noktası için tahmin yap
# [LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6]
tahmin = model.predict([[20000, 2, 2, 1, 25, 1, 2, 1, 0, 0, 0]])
print("Yeni veri noktası için tahmin:", tahmin)

# Veri setindeki bir veri noktası kullanarak tahmin yap
denemex = np.array(x.iloc[29996])
tahmin2 = model.predict([denemex])
print("Mevcut veri noktası için tahmin:", tahmin2)
