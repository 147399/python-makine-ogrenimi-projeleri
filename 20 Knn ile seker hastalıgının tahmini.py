import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Verinin okunması ve kopyalanması
data = pd.read_csv("C:/Users/ASAF/Desktop/Datasets/diabetes.csv")
df = data.copy()

# Sağlıklı ve şeker hastası olan kişilerin ayrılması
saglıklı = df[df.Outcome == 0]
sekerhastasi = df[df.Outcome == 1]

# Yaş ve glikoz değerlerine göre scatter plot oluşturulması
plt.scatter(saglıklı.Age, saglıklı.Glucose, color="green", label="sağlıklı", alpha=0.4)
plt.scatter(sekerhastasi.Age, sekerhastasi.Glucose, color="red", label="hasta", alpha=0.4)
plt.xlabel("yaş")
plt.ylabel("glikoz")
plt.legend()
plt.show()

# Bağımsız değişkenlerin (X) ve bağımlı değişkenin (Y) ayrılması
x = data.iloc[:, 0:-1]
y = data.iloc[:, -1].values

# Verinin eğitim ve test setlerine ayrılması
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=19, train_size=0.70)

# Verinin standartlaştırılması (ölçeklenmesi)
sc = StandardScaler()
x_train1 = sc.fit_transform(x_train)
x_test1 = sc.transform(x_test)

# K-En Yakın Komşu (KNN) sınıflandırıcısının oluşturulması ve eğitilmesi
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(x_train1, y_train)

# Test seti üzerindeki doğruluk skorunun hesaplanması
score = knn.score(x_test1, y_test)
print(score)

# Farklı komşu sayıları için doğruluk oranlarının saklanacağı liste
scorelist = []

# 1'den 30'a kadar olan komşu sayıları için KNN modeli oluşturma ve doğruluk skorlarının hesaplanması
for i in range(1, 30):
    knn2 = KNeighborsClassifier(n_neighbors=i)
    knn2.fit(x_train1, y_train)
    scorelist.append(knn2.score(x_test1, y_test))

# Komşu sayısına karşılık doğruluk oranlarının çizdirilmesi
plt.plot(range(1, 30), scorelist)
plt.xlabel("komşu sayısı")
plt.ylabel("doğruluk oranı")
plt.show()