import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

data = pd.DataFrame()

data["cumleler"] = ["ali bak ","bak ali ata ","ali ata bak ", " bak ali ata ","sevil sen kes","ipek bık iki ","ipek" ]

cv = CountVectorizer(max_features=4)

a = cv.fit_transform(data["cumleler"])

torray = a.toarray()  # yogun olan kelımelerı bulup kacıncı ındekste oldugunu gosterırır 
print(torray)

tekrarlayanlar = cv.get_feature_names_out() # tekrarlayan kelimeleri gösgterir
print(tekrarlayanlar)


# stop wors kelimiler = cumlede kullanıldıgında anlamı degıstırmeyen sadece cumle kurmaya yardımcı kelımelerdır " ne, gibi ,mesela ,fakat vb. "

