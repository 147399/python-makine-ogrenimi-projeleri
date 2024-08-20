import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# MNIST datasetini indir
mnist = fetch_openml('mnist_784', version=1)

def resim(dframe, index):
    # Veriyi numpy array'e dönüştür
    numara = dframe.to_numpy()[index]
    # Görüntüyü 28x28 şekline dönüştür
    numara_resim = numara.reshape(28, 28)
    # Görüntüyü çiz
    plt.imshow(numara_resim, cmap="binary")
    plt.axis("off")
    plt.show()

# Veriyi eğitim ve test kümelerine ayırma
train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, random_state=20, train_size=0.70)
test_img_kopya = test_img.copy()

# Standartlaştırma
sc = StandardScaler()
train_img = sc.fit_transform(train_img)
test_img = sc.transform(test_img)

# PCA ile boyut azaltma
pca = PCA(n_components=0.95)
train_img = pca.fit_transform(train_img)
test_img = pca.transform(test_img)

# Yeni sütun sayısını yazdırma
n = pca.n_components_
print("Sütun sayımız:", n)

# Logistic Regression modeli oluşturma ve eğitme
lr = LogisticRegression(solver="lbfgs", max_iter=10000)
model = lr.fit(train_img, train_lbl)

# Modelin test kümesi üzerindeki skoru
score = model.score(test_img, test_lbl)
tahmin = model.predict(test_img[4].reshape(1, -1))

#  indexlerdeki resimleri görüntüleme
resim(test_img_kopya, 13)

# Tahmin ve skorları yazdırma
print("Tahmin:", tahmin)
print("Skor:", score)