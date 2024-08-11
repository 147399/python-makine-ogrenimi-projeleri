"""




def uzun_kelime_ters_cevir(cumle):""
    # Cümleyi kelimelere ayır
    kelimeler = cumle.split()

    # Cümledeki her kelimeyi kontrol et
    for i in range(len(kelimeler)):
        # Eğer kelimenin beş veya daha fazla harfi varsa, ters çevir
        if len(kelimeler[i]) >= 5:
            kelimeler[i] = kelimeler[i][::-1]  # Dilimleme yöntemiyle kelimeyi ters çevir

    # Kelimeleri tekrar birleştirerek tek bir cümle haline getir
    ters_cümle = ' '.join(kelimeler)

    return ters_cümle

# Fonksiyonu test et
print(uzun_kelime_ters_cevir("selam ne yapıyordun"))  
print(uzun_kelime_ters_cevir("kurt agızlı olmayı bırakmalısın gus"))  

"""

# kullanuıcıdan cumle alan versıyonu


def uzun_kelime_ters_cevir():
    # Kullanıcıdan bir cümle girdisi al
    cumle = input("Lütfen bir cümle girin: ")

    # Cümleyi kelimelere ayır
    kelimeler = cumle.split()

    # Cümledeki her kelimeyi kontrol et
    for i in range(len(kelimeler)):
        # Eğer kelimenin beş veya daha fazla harfi varsa, ters çevir
        if len(kelimeler[i]) >= 5:
            kelimeler[i] = kelimeler[i][::-1]  # Dilimleme yöntemiyle kelimeyi ters çevir

    # Kelimeleri tekrar birleştirerek tek bir cümle haline getir
    ters_cümle = ' '.join(kelimeler)

    return ters_cümle

# Fonksiyonu çağır ve sonucu yazdır
print(uzun_kelime_ters_cevir())