import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error , mean_squared_error ,mean_absolute_percentage_error

data = pd.read_csv("C:/Users/ASAF/Desktop/Datasets/insurance.csv")
veri = data.copy()

veri = pd.get_dummies(veri,columns=["sex","smoker","region"],drop_first=True)

y = veri["charges"]
x = veri.drop(columns="charges",axis=1)

lr = LinearRegression()
model = lr.fit(x,y)

tahmin = model.predict([[19,34,2,False,True,False,True,False]])
oran = model.score(x,y)

print(tahmin)
print(oran)


veri_hata = pd.DataFrame()
veri_hata["y"] = y
y_tahmin = model.predict(x)
veri_hata["tahmin"] = y_tahmin

print(veri_hata)
print(y_tahmin)


mae = mean_absolute_error(y,y_tahmin)
mape = mean_absolute_percentage_error(y,y_tahmin)
mse = mean_squared_error(y,y_tahmin)




print(" mae = {}\n mape = {}\n mse ={}".format(mae,mape,mse))