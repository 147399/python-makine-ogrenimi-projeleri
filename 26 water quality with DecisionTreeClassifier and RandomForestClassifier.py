import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, confusion_matrix
import plotly.express as px
import missingno as msno

# Veriyi yükle ve bir kopyasını oluştur
data = pd.read_csv("C:/Users/ASAF/Desktop/Datasets/water_potability.csv")
df = data.copy()

# Bu adım, verinin içilebilir (1) ve 5içilemez (0) olan su örneklerinin sayısını belirler.
new = pd.DataFrame(df["Potability"].value_counts())
new.columns = ["count"]  
print(new)

# İçilebilir ve içilemez su örneklerinin oranını göstermek için pasta grafiği kullanıyoruz.
fig = px.pie(new, values="count", names=["icilebilir", "icilemez"], hole=0.4, opacity=0.8,
             labels={"label": "Potability", "count": "numune sayısı"})
  
# Bu adımda grafiğe başlık ekliyoruz ve sonra grafiği gösteriyoruz.
fig.update_layout(title=dict(text="İçilebilirlik"))
fig.show()  

# Veriler arasındaki ilişkiyi incelemek için pairplot oluştur
sns.pairplot(new)

# Veri setindeki değişkenler arasındaki korelasyonu incelemek için kullanılır.
sns.clustermap(df.corr(), cmap="vlag", dendrogram_ratio=(0.1, 0.2), annot=True, linewidth=0.8, figsize=(9, 10))
plt.show()  

# İçilebilir ve içilemez su örnekleri için KDE (Kernel Density Estimate) plot oluştur
non_potable = df.query("Potability==0") 
potable = df.query("Potability==1")  

plt.figure(figsize=(15, 15))  

# Bu döngü, her sütun için ayrı bir KDE grafiği oluşturur.
for ax, col in enumerate(df.columns[:9]):
    plt.subplot(3, 3, ax + 1)  # 3x3 matris içinde her bir sütun için grafik oluşturur.
    plt.title(col)  # Grafiğin başlığı olarak sütun ismini kullanır.
    sns.kdeplot(x=non_potable[col], label="içilmez")  
    sns.kdeplot(x=potable[col], label="icilebilir")  
    plt.legend()  # Grafik açıklamalarını ekler.

plt.tight_layout()  # Alt grafiklerin düzenini sıkılaştırır ve çakışmaları önler.
plt.show()  

# Veri setinde eksik değerler olup olmadığını görmek için bu grafiği kullanıyoruz.
msno.matrix(df, color=(0, 0, 0)) 

# Veri setindeki eksik değerlerin sayısını yazdır
print(df.isnull().sum())

# Hedef değişken (y) ve özellikler (x) olarak veriyi ayır
y = df["Potability"]
x = df.drop(columns=["Potability"], axis=1)

# Veriyi eğitim ve test setlerine ayır
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=20, train_size=0.70)

# Eğitim ve test setlerinin boyutlarını göster
print(x_train.shape)
print(x_test.shape)

# Eğitim ve test setleri için min-max normalizasyonu uygula
x_trainmax = np.max(x_train)
x_trainmin = np.min(x_train)
x_train = (x_train - x_trainmin) / (x_trainmax - x_trainmin) 
x_test = (x_test - x_trainmin) / (x_trainmax - x_trainmin)  

# Karar ağacı (Decision Tree) ve Rastgele Orman (Random Forest) modellerini kullanacağız.
models = [("DTC", DecisionTreeClassifier(max_depth=3)),
          ("RF", RandomForestClassifier())]

# Sonuçları ve karışıklık matrislerini kaydetmek için listeler oluştur
sonuc = []  
cmlist = [] 

# Bu döngü, modelleri sırasıyla eğitir, test setinde tahminler yapar ve sonuçları toplar.
for name, ml in models:
    ml.fit(x_train, y_train)  # Modeli eğitim verisi üzerinde eğit
    model_result = ml.predict(x_test)  
    score = precision_score(y_test, model_result)  
    cm = confusion_matrix(y_test, model_result) 
    sonuc.append((name, score)) 
    cmlist.append((name, cm)) 

# Sonuçları yazdır
print(sonuc)


# Her bir model için karışıklık matrisini bir heatmap (ısı haritası) olarak çizdirir.
for name, i in cmlist:
    plt.figure()  # Yeni bir figür oluştur
    sns.heatmap(i, annot=True, linewidths=0.8, fmt=".1f")  # Karışıklık matrisini ısı haritası olarak çiz
    plt.title(name) 
    plt.show()  

