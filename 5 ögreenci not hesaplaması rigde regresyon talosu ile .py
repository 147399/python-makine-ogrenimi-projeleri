import pandas as pd 
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt

# Veriyi yükle
data = pd.read_csv("C:/Users/ASAF/Desktop/Datasets/student_scores.csv")
veri = data.copy()

# Bağımlı ve bağımsız değişkenleri ayır
y = veri["Scores"]
x = veri[["Hours"]]  # x'i DataFrame olarak geçiyoruz

# Linear Regression modelini oluştur ve eğit
lr = LinearRegression()
model = lr.fit(x, y)
print(f'Model skoru: {model.score(x, y)}')

# Ridge Regresyonu için farklı alfa değerlerini dene
alfalar = [1, 10, 100, 150, 200]
for a in alfalar:
    r = Ridge(alpha=a)
    modelr = r.fit(x, y)
    skor = modelr.score(x, y)  # Model skoru hesaplanıyor
    print(f'Ridge Regression alpha={a}, skor={skor}')

# Veri noktalarını scatter plot olarak göster
plt.style.use("fivethirtyeight")
plt.figure(figsize=(10, 8))
plt.scatter(x, y, label='Veri noktaları')

# Linear Regression doğrusunu çiz
plt.plot(x, model.predict(x), color='red', label='Linear Regresyon doğrusu')
plt.xlabel('Hours')  # X ekseni etiketi
plt.ylabel('Scores')  # Y ekseni etiketi
plt.title('Hours vs Scores')  # Grafik başlığı
plt.legend()  # Grafik üzerindeki etiketleri göster
plt.show()  # Grafik gösterimi



