import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Ses dosyasını oku
sample_rate, data = wavfile.read('kayit.wav')

# Bilgileri ekrana yazdır
print("Örnekleme frekansı:", sample_rate, "Hz")
print("Toplam örnek sayısı:", len(data))
print("Süre:", len(data) / sample_rate, "saniye")

# Zaman eksenini oluştur
t = np.arange(len(data)) / sample_rate

# Grafiği çiz
plt.plot(t, data)
plt.xlabel('Zaman (saniye)')
plt.ylabel('Genlik')
plt.title('Ses Sinyali - Dalga Formu')
plt.show()

# Frekans analizi (FFT)
fft_sonucu = np.fft.rfft(data)
frekanslar = np.fft.rfftfreq(len(data), d=1/sample_rate)
genlik_db = 20 * np.log10(np.abs(fft_sonucu) + 1e-10)

# Frekans spektrumunu çiz
plt.figure()
plt.plot(frekanslar, genlik_db) 
plt.xlabel('Frekans (Hz)')
plt.ylabel('Genlik (dB)')
plt.title('Frekans Spektrumu (FFT)')
plt.show()

from scipy.signal import butter, filtfilt

# Filtre tasarımı (1000 Hz üstünü kes)
kesim_frekansi = 1000
nyquist = sample_rate / 2
normalize_kesim = kesim_frekansi / nyquist
b, a = butter(5, normalize_kesim, btype='low')

# Filtreyi uygula
filtrelenmis_data = filtfilt(b, a, data)

# Orijinal ve filtrelenmiş sesi karşılaştır
plt.figure()
plt.plot(t, data, label='Orijinal', alpha=0.7)
plt.plot(t, filtrelenmis_data, label='Filtrelenmiş', alpha=0.7)
plt.xlabel('Zaman (saniye)')
plt.ylabel('Genlik')
plt.title('Filtre Öncesi ve Sonrası Karşılaştırma')
plt.legend()
plt.show()

# Filtrelenmiş sesi yeni bir dosya olarak kaydet
wavfile.write('kayit_filtrelenmis.wav', sample_rate, filtrelenmis_data.astype(data.dtype))
print("Filtrelenmiş ses 'kayit_filtrelenmis.wav' olarak kaydedildi.")


from scipy.signal import spectrogram

# Spektrogram hesapla
frekanslar_s, zamanlar_s, guc = spectrogram(data, sample_rate, nperseg=1024)
guc_db = 10 * np.log10(guc + 1e-10)

# Spektrogramı çiz
plt.figure()
plt.pcolormesh(zamanlar_s, frekanslar_s, guc_db, shading='gouraud', cmap='magma')
plt.xlabel('Zaman (saniye)')
plt.ylabel('Frekans (Hz)')
plt.title('Spektrogram')
plt.colorbar(label='Güç (dB)')
plt.show()
