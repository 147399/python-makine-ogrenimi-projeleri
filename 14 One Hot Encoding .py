import pandas as pd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

# Veri setini okuma
data = pd.read_csv("C:/Users/ASAF/Desktop/Datasets/Churn_Modelling.csv")
veri = data.copy()

# Gereksiz sütunları kaldırma
veri = veri.drop(columns=["Surname", "RowNumber", "CustomerId"])

# Kategorik verileri sayısal verilere dönüştürme
veri = pd.get_dummies(veri, columns=["Geography", "Gender"], drop_first=True)

# OneHotEncoder kullanarak 'Geography' ve 'Gender' sütunlarını dönüştürme
one = OneHotEncoder()
xd = one.fit_transform(veri[["Geography", "Gender"]]).toarray()

# Kodun bu kısmında OneHotEncoder ile elde edilen özellik isimlerini almak istiyorsunuz
xd1 = one.get_feature_names_out(["Geography", "Gender"])

print(xd1)

# Hedef değişken (y) ve özellikler (x) ayırma
y = veri["Exited"]
x = veri.drop(columns=["Exited"], axis=1)

# Veriyi eğitim ve test setlerine ayırma
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.70, random_state=21)

# RandomForestClassifier modelini oluşturma ve eğitme
rf = RandomForestClassifier()
model = rf.fit(x_train, y_train)

# Modelin test setindeki başarımını hesaplama
score = model.score(x_test, y_test)          

# Modelin başarımını yazdırma
print("score : ", score)

# Yeni bir veri noktası ile tahmin yapma
tahmin = model.predict([[340, 21, 1, 450, 1, 0, 0, 10122, 0, 0, 1]])
print(tahmin)

