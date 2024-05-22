from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import warnings

# Uyarıları görmezden gel
#warnings.filterwarnings(action='ignore', category=UserWarning)

# Veriyi yükle
data = pd.read_csv("C:/Users/ASAF/Desktop/Datasets/Student_Marks.csv")
veri = data.copy()

#print(veri[:3]) # İlk 3 satırı yazdırma

# Bağımlı ve bağımsız değişkenleri ayır
y = veri["Marks"]  # Bağımlı değişken (hedef değişken)
X = veri[["number_courses", "time_study"]]  # Bağımsız değişkenler (özellikler)

# Veriyi eğitim ve test setlerine ayır
X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluştur ve eğit
lr = LinearRegression()
model = lr.fit(X, y)

# Tahmin yap
predictions = model.predict(np.array([[4, 5]]))  # 4 ders ve 5 saat çalışma süresi için tahmin yap
print(predictions)

# 'Marks' sütunundaki en yüksek değeri bul
max_mark = data["Marks"].max()

# Sonucu yazdır
print(f'Marks sütunundaki en yüksek değer: {max_mark}')

# Modelin performansını değerlendir
modelb = model.score(X, y)
print(modelb)

# Yeni bir tahmin yap
result = model.predict([[3, 4.508]])  # 3 ders ve 4.508 saat çalışma süresi için tahmin yap
print(result)