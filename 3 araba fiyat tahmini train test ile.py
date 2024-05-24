import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as mt

data = pd.read_csv("C:/Users/ASAF/Desktop/Datasets/Audi_A1_listings.csv")
veri = data.copy()


veri = veri.drop(columns=["Score","href","MileageRank" ,"PriceRank","PPYRank","PPY","index","Type"])


veri = veri.rename( columns={"Year":"yıl",
                             "Mileage(miles)":"Mil",
                             "Engine":"Motor",
                             "Transmission":"Vites",
                             "Fuel":"Yakıt",
                             "Number_of_Owners":"Sahip sayısı",
                             "Price(£)":"Fiyat"

})

veri["Motor"] = veri["Motor"].str.replace("L","")
veri["Motor"] = pd.to_numeric(veri["Motor"],errors='coerce')

veri = pd.get_dummies(veri , columns=["Yakıt","Vites"])

y = veri["Fiyat"]
x = veri.drop(columns=["Fiyat"],axis=1)

x_test , x_train ,y_test ,y_train = train_test_split(x,y,test_size=0.2,random_state=42 )

lr = LinearRegression()
model = lr.fit(x_train,y_train)

model.score(x_test,y_test)

tahmin = model.predict([[2018, 44000, 1.6, 115, 2400, True, False, True, False]]) 

print(tahmin)
oran = model.score(x_test,y_test)
print(oran)