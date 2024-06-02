import pandas as pd
from  sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split


# Veriyi yükle
data = pd.read_csv("C:/Users/ASAF/Desktop/Datasets/heart.csv")
veri = data.copy()

# Veri hakkında bilgi edin
print(veri.info())

# Bağımlı ve bağımsız değişkenleri ayır
y = veri["output"]  # Hedef değişken (output)
x = veri.drop(columns="output", axis=1)  # Özellikler (output sütunu hariç diğer sütunlar)

# Karar ağacı sınıflandırıcısını oluştur ve tüm veriyle eğit
tree = DecisionTreeClassifier()
model = tree.fit(x, y)
print(f'Model skoru (tüm veri): {model.score(x, y)}')

# Veriyi eğitim ve test setlerine ayır
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.70, random_state=16)

# Karar ağacı sınıflandırıcısını oluştur ve eğitim seti ile eğit
tree = DecisionTreeClassifier()
model = tree.fit(x_train, y_train)

# Test seti üzerinde modelin doğruluk skorunu hesapla
test_score = model.score(x_test, y_test)
print(f'Model skoru (test seti): {test_score}')

# Yeni bir veri örneği üzerinde tahmin yap
tahmin = model.predict([[31, 1, 2, 130, 240, 0, 0, 150, 0, 2, 0, 0, 2]])
print(f'Tahmin: {tahmin}')

x_train , x_test , y_train, y_test = train_test_split(x,y,train_size=0.70 , random_state= 16)

tree = DecisionTreeClassifier()
model = tree.fit(x_train,y_train)
model.score(x_test,y_test)

tahmin = model.predict([[31,1,2,130,240,0,0,150,0,2,0,0,2]])
print(tahmin)






