import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# Veriyi yükleme ve kopyalama
data = pd.read_csv("C:/Users/ASAF/Desktop/Datasets/heart.csv")
df = data.copy()

# "sex" sütununun değer sayımlarını yazdırma
valuecounts = df["sex"].value_counts()
print("Sex column value counts:\n", valuecounts)

# Sütun isimlerini yazdırma
columns_name = df.columns
print("Columns in DataFrame:\n", columns_name)

# Her sütunun benzersiz değer sayısını yazdırma
for i in list(df.columns):
    print("{} --- {}".format(i, df[i].value_counts().shape[0]))

# Kategorik sütunları belirleme
kategorik = ["sex", "fbs", "restecg", "exng", "slp", "thall", "caa", "cp"]

# Kategorik sütunları içeren yeni bir DataFrame oluşturma
df_kat = df.loc[:, kategorik]
print("Categorical columns DataFrame:\n", df_kat)

# Grafik penceresi oluşturma
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 20))
axes = axes.flatten()

# Her kategorik sütun için sayım grafiği oluşturma
for ax, i in zip(axes, df_kat.columns):
    sns.countplot(x=i, data=df, hue="output", ax=ax)
    ax.set_title(i)

plt.tight_layout()
plt.show()

# Sayısal sütunlar
sayisal = ["age", "trtbps", "chol", "thalachh", "oldpeak"]
df_say = df.loc[:, sayisal + ["output"]]  # output sütununu da ekleyin

print("Numerical columns DataFrame:\n", df_say.head(5))

# Pairplot oluşturma
sns.pairplot(df_say, hue="output", diag_kind="kde")
plt.show()

# Sayısal verilerin standartlaştırılması
scaler = StandardScaler()
scaled_array = scaler.fit_transform(df[sayisal[:-1]])

print("Scaled numerical data:\n", scaled_array)

# Kategorik verilerin one-hot encoding ile dönüştürülmesi
df1 = data.copy()
df1 = pd.get_dummies(df1, columns=kategorik[:-1], drop_first=True)

# Bağımlı ve bağımsız değişkenlerin ayrılması
y = df1["output"]
x = df1.drop(columns=["output"], axis=1)

# Sayısal verilerin yeniden standartlaştırılması
scaler = StandardScaler()
x[sayisal[:-1]] = scaler.fit_transform(x[sayisal[:-1]])

# Eğitim ve test setlerine ayırma
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=5, train_size=0.70)

# Lojistik regresyon modelinin oluşturulması ve eğitilmesi
lg = LogisticRegression()
model = lg.fit(x_train, y_train)

# Modelin test setindeki doğruluğunu hesaplama
score = model.score(x_test, y_test)
print("Test Accuracy:", score)

# Test seti üzerindeki tahmin olasılıklarını hesaplama
ypred_prob = model.predict_proba(x_test)
print("Predicted probabilities:\n", ypred_prob)

# Tahmin sonuçlarının maksimum olasılıkla sınıflandırılması
ypred = np.argmax(ypred_prob, axis=1)

# Tahmin sonuçlarını içeren bir DataFrame oluşturma
dummy = pd.DataFrame(ypred_prob)
dummy["ypred"] = ypred  

print("Predictions DataFrame:\n", dummy)

# Tahminlerin gerçek değerlerle karşılaştırılması
print("Test Accuracy:", accuracy_score(ypred, y_test))