import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Veriyi CSV dosyasından okuma
data = pd.read_csv("C:/Users/ASAF/Desktop/Datasets/banka.csv")

# Orijinal veriyi kopyalama
veri = data.copy()

# Gerekli sütunları seçme
veri = veri[["sorgu", "label"]]

# Stop words (gereksiz kelimeler) listesi
stapworts = ['fakat','lakin','ancak','acaba', 'ama', 'aslında', 'az', 'bazı',
             'belki', 'biri', 'birkaç', 'birşey', 'biz', 'bu', 'çok', 'çünkü', 
             'da', 'daha', 'de', 'defa', 'diye', 'eğer', 'en', 'gibi', 'hem', 
             'hep', 'hepsi', 'her', 'hiç', 'için', 'ile', 'ise', 'kez', 'ki', 
             'kim', 'mı', 'mu', 'mü', 'nasıl', 'ne', 'neden', 'nerde', 'nerede',
             'nereye', 'niçin', 'niye', 'o', 'sanki', 'şey', 'siz', 'şu', 'tüm',
             've', 'veya', 'ya', 'yani']

# Kullanıcıdan bir mesaj girişi al
mesaj = input("Yapmak istediğiniz işlemi giriniz: ")

# Yeni mesajı veri çerçevesine ekleme
mesajveri = pd.DataFrame({"sorgu": mesaj, "label": 0}, index=[42])
veri = pd.concat([veri, mesajveri], ignore_index=True)

# Stop words'leri sorgulardan temizleme
for word in stapworts:
    word = " " + word + " "
    veri["sorgu"] = veri["sorgu"].str.replace(word, " ")

# CountVectorizer kullanarak kelime sayım matrisi oluşturma
cv = CountVectorizer(max_features=20)
x = cv.fit_transform(veri["sorgu"]).toarray()
y = veri["label"]

# Tahmin edilecek mesajı ayırma
tahmin = x[-1].copy()

# Eğitim verisini ve etiketleri ayırma
x = x[0:-1]
y = y[0:-1]

# Veriyi eğitim ve test setlerine ayırma
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=30)

# RandomForestClassifier modeli oluşturma ve eğitme
rf = RandomForestClassifier()
model = rf.fit(x_train, y_train)

# Modelin test seti üzerindeki başarımını ölçme
score = model.score(x_test, y_test)

# Yeni mesajın etiketini tahmin etme
sonuc = model.predict([tahmin])

# Sonucu ve modeli başarı skorunu yazdırma
print("Sonuç:", sonuc, " Skor:", score)