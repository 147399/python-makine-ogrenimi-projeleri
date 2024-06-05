import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import numpy as np 

# Veriyi yükle
data = pd.read_csv("C:/Users/ASAF/Desktop/Datasets/heart.csv")
veri = data.copy()

# Bağımlı ve bağımsız değişkenleri ayır
y = veri["output"]
x = veri.drop(columns="output", axis=1)

# Veriyi eğitim ve test setlerine ayır
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.70, random_state=30)

# Karar ağacı sınıflandırıcısını oluştur ve eğit
tree = DecisionTreeClassifier()
model = tree.fit(x_train, y_train)

# Test seti üzerinde modelin doğruluk skorunu hesapla
test_score = model.score(x_test, y_test)
print(f'Model skoru (test seti): {test_score}')

# Tüm veri üzerinde modelin doğruluk skorunu hesapla
oran = model.score(x, y)
print(f'Tüm veri üzerindeki model skoru: {oran}')

# Karar ağacını görselleştirmek için DOT formatında export et
dot = export_graphviz(model, feature_names=x.columns, filled=True, rounded=True, special_characters=True)
gorsel = graphviz.Source(dot)

# Karar ağacının görselleştirilmesini sağlar
gorsel.render("karar_agaci")  # Bu satır, karar ağacını bir dosya olarak kaydeder
gorsel.view()  # Bu satır, karar ağacını bir pencerede görüntüle
