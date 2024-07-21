import yfinance as yf
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt  # matplotlib'i import edin


# Veriyi indirme
df = yf.download('NVDA', "2014-01-01", "2024-04-26")

# Sadece 'Close' sütununu kullanma
df = df[["Close"]]

# DataFrame'i sıfırlama ve yeniden adlandırma
df = df.reset_index()
df = df.rename(columns={'Date': 'ds', 'Close': 'y'})

# Modeli oluşturma ve eğitme
model = Prophet()
model.fit(df)

# Gelecekteki tarihleri içeren DataFrame oluşturma
gelecek = model.make_future_dataframe(periods=360)

# Tahmin yapma
tahmin = model.predict(gelecek)

# Tahminleri görselleştirme
fig1 = model.plot(tahmin)
fig2 = model.plot_components(tahmin)

# Grafikleri gösterme
plt.show()  # Bu komutu ekleyin

# Veriyi yazdırma
print(df)