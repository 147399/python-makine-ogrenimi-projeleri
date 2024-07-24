import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns 
import matplotlib.pyplot as plt
# Veriyi CSV dosyasından oku
data = pd.read_csv("C:/Users/ASAF/Desktop/Datasets/customer_booking.csv", encoding='ISO-8859-1')
veri = data.copy()  # Orijinal veriyi yedekle

# Belirtilen kategorik sütunları dummy değişkenlere dönüştür (one-hot encoding)
veri = pd.get_dummies(veri, columns=["sales_channel", "trip_type", 'flight_day', 'route', 'booking_origin'], drop_first=True)

# Hedef değişkeni (y) ve özellikler (X) belirle
y = veri["wants_extra_baggage"] 
x = veri.drop("wants_extra_baggage", axis=1) 

# Veriyi eğitim ve test setlerine ayır
x_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.70, random_state=42)

# Rastgele Orman (Random Forest) sınıflandırıcısını oluştur ve eğit
rf = RandomForestClassifier()
model = rf.fit(x_train, y_train)

# Test setinde modelin doğruluk skorunu hesapla
score = model.score(X_test, y_test)

# Çapraz doğrulama (cross-validation) ile modelin performansını değerlendirme
crossval = cross_val_score(model, x, y, cv=4)
cv = crossval.mean()  # Çapraz doğrulama skorlarının ortalamasını al

# Skorları yazdır
print(score)
print(cv)






sns.pairplot(veri)

plt.show
