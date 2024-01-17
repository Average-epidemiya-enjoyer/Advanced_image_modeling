import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift


def create_subject(lines, N):
    subject = np.zeros((N, N))
    line_width = N // (lines * 2)
    for i in range(lines):
        subject[:, i * 2 * line_width:(i * 2 + 1) * line_width] = 1
    return subject


def calculate_fpm_noncoherent(subject, pupil):
    intensity_subject = np.abs(subject) ** 2
    fourier_intensity = fftshift(fft2(intensity_subject))
    convolved = fourier_intensity * pupil
    image = np.abs(ifft2(convolved)) ** 2
    contrast = (np.max(image) - np.min(image)) / (np.max(image) + np.min(image))
    return contrast


def calculate_fpm_coherent(subject, pupil):
    fourier_subject = fftshift(fft2(subject))
    convolved = fourier_subject * pupil
    image = np.abs(ifft2(convolved)) ** 2
    contrast = (np.max(image) - np.min(image)) / (np.max(image) + np.min(image))
    return contrast


N = 1024
Dzr = 100

X, Y = np.meshgrid(np.linspace(-N / 2, N / 2, N), np.linspace(-N / 2, N / 2, N))
R = np.sqrt(X ** 2 + Y ** 2)
pupil = R <= N / Dzr

spatial_frequencies = []
contrast_noncoherent = []
contrast_coherent = []

for lines in range(1, 100, 2):
    subject = create_subject(lines, N)
    contrast_nc = calculate_fpm_noncoherent(subject, pupil)
    contrast_c = calculate_fpm_coherent(subject, pupil)
    spatial_frequency = lines / N
    spatial_frequencies.append(spatial_frequency)
    contrast_noncoherent.append(contrast_nc)
    contrast_coherent.append(contrast_c)

plt.figure(figsize=(12, 6))
plt.plot(spatial_frequencies, contrast_noncoherent, label='Некогерентное')
plt.plot(spatial_frequencies, contrast_coherent, label='Когерентное')
plt.xlabel('Пространственная частота')
plt.ylabel('Контраст')
plt.title('Функция передачи модуляции (ФПМ)')
plt.legend()
plt.grid(True)
plt.show()

sharp_drop_frequency = next(
    (sp_freq for sp_freq, contrast in zip(spatial_frequencies, contrast_coherent) if contrast < 0.3), None)
print(f"Пространственная частота с резким падением ФПМ: {sharp_drop_frequency}")
