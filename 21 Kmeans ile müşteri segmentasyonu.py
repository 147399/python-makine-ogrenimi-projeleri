import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Veriyi CSV dosyasından okuma
data = pd.read_csv("C:/Users/ASAF/Desktop/Datasets/Avm_Musterileri.csv")
df = data.copy()

# Veriyi ölçeklendirme (MinMaxScaler ile 0-1 aralığına getirme)
sc = MinMaxScaler()
sc.fit(df[["Annual Income (k$)"]])
df[["Annual Income (k$)"]] = sc.transform(df[["Annual Income (k$)"]])

sc.fit(df[["Spending Score (1-100)"]])
df[["Spending Score (1-100)"]] = sc.transform(df[["Spending Score (1-100)"]])

# Dirsek yöntemini kullanarak uygun küme sayısını belirleme
dirsek = range(1, 11)
list = []

for k in dirsek:
    km = KMeans(n_clusters=k)
    km.fit(df[["Annual Income (k$)", "Spending Score (1-100)"]])
    list.append(km.inertia_)

# Dirsek grafiğini çizme
plt.xlabel("k")
plt.ylabel("Dirsek")
plt.plot(dirsek, list)
plt.show()

# K-means modelini 5 küme ile oluşturma ve eğitme
kson = KMeans(n_clusters=5)
y_pred = kson.fit_predict(df[["Annual Income (k$)", "Spending Score (1-100)"]])

# Tahmin edilen küme etiketlerini ekrana yazdırma
print(y_pred)

# Veri setine küme etiketlerini ekleme
df["cluster"] = y_pred
print(df.head(3))

# Küme merkezlerini elde etme
kson.cluster_centers_

# Her bir küme için veri setini ayırma
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]
df4 = df[df.cluster == 3]
df5 = df[df.cluster == 4]

# Kümeleri görselleştirme
plt.xlabel("Gelir")
plt.ylabel("Skor")
plt.scatter(df1["Annual Income (k$)"], df1["Spending Score (1-100)"], label="Cluster 0")
plt.scatter(df2["Annual Income (k$)"], df2["Spending Score (1-100)"], label="Cluster 1")
plt.scatter(df3["Annual Income (k$)"], df3["Spending Score (1-100)"], label="Cluster 2")
plt.scatter(df4["Annual Income (k$)"], df4["Spending Score (1-100)"], label="Cluster 3")
plt.scatter(df5["Annual Income (k$)"], df5["Spending Score (1-100)"], label="Cluster 4")
# Küme merkezlerini (centroid'leri) görselleştirme
plt.scatter(kson.cluster_centers_[:,0,],kson.cluster_centers_[:,1] ,   color="black",marker="X",label="centroid")
plt.legend()
plt.show()