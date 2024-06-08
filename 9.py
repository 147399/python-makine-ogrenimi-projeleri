import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.linear_model import LinearRegression


data = pd.read_csv("C:/Users/ASAF/Desktop/Datasets/UCI_Credit_Card.csv")
veri = data.copy()

veri = veri.drop(columns=["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","ID"])


y = veri["default.payment.next.month"]
x = veri.drop(columns="default.payment.next.month", axis=1 )

x_train , x_test , y_train, y_test = train_test_split(x,y,train_size= 0.77 , random_state= 42)


lr = LinearRegression()
model = lr.fit(x_train,y_train)
model.score(x_test,y_test)

tahmin  = model.predict([[20000,2,2,1,25,1,2,1,0,0,0]])
print(tahmin)

denemex = np.array(x.iloc[29996])

tahmin2 = model.predict([denemex])

print(tahmin2)



