import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift
from scipy.ndimage import zoom


def comb(lines, N):
    w = N / (lines * 2)
    f_x = np.arange(-w, N + 1)
    ww = np.arange(0, N + 1, w * 2)
    f_y = np.zeros_like(f_x)
    for w_i in ww:
        f_y[(f_x >= w_i) & (f_x <= w_i + w)] = 1

    func = np.zeros((N, N))
    for i in range(N // 2):
        func[:, i] = f_y[i]
        func[:, N - i - 1] = f_y[i]

    for j in range(N // 2):
        if not np.array_equal(func[:, j], func[:, N - j - 1]):
            print('Не соблюдение симметрии')

    return func


N = 2048
lambda_ = 0.5
Dzr = 100
A = 0.5
count = 1

dp = Dzr / N
dn = 1 / (N * dp)
dx = dn * (lambda_ / A)

n_max = dn * N / 2
p_max = dp * N / 2
x_max = dx * N / 2

nx, ny = np.meshgrid(np.arange(0, n_max, dn), np.arange(1, n_max + dn, dn))
px, py = np.meshgrid(np.arange(0, p_max, dp), np.arange(1, p_max + dp, dp))
x, y = np.meshgrid(np.arange(0, x_max, dx), np.arange(1, x_max + dx, dx))

X, Y = np.meshgrid(np.arange(-N // 2, N // 2), np.arange(-N // 2, N // 2))
R = np.sqrt(X ** 2 + Y ** 2)
pupil = R <= N / Dzr
Fzr = np.pad(pupil.astype(float), ((0, 1), (0, 1)), mode='constant', constant_values=0)
target_shape = (N, N)
Fzr = zoom(Fzr, (target_shape[0] / Fzr.shape[0], target_shape[1] / Fzr.shape[1]))

k = []
kk = []
s = []

for i in range(1, 93, 3):
    Subject = comb(i, N)
    FUp = (dn / dp) * (fftshift(fft2(fftshift(np.abs(Subject) ** 2))) / N)
    hn = (dp / dn) * (fftshift(ifft2(fftshift(Fzr))) * N)
    hn = np.abs(hn) ** 2 / np.pi ** 2
    Fhp = (dn / dp) * (fftshift(fft2(fftshift(hn))) / N * np.pi)

    FUp = FUp[:Fhp.shape[0], :Fhp.shape[1]]
    FUpzr = FUp * Fhp
    Ink = (dp / dn) * (fftshift(ifft2(fftshift(FUpzr))) * N)
    if np.max(Ink) != np.min(Ink):
        contrast_nk = (np.max(Ink) - np.min(Ink)) / (np.max(Ink) + np.min(Ink))
    else:
        contrast_nk = 0
    k.append(contrast_nk)

    Up = (dn / dp) * (fftshift(fft2(fftshift(Subject))) / N)
    Up = zoom(Up, (target_shape[0] / Up.shape[0], target_shape[1] / Up.shape[1]))
    Upzr = np.multiply(Up, Fzr)

    Upzr = Upzr[:FUpzr.shape[0], :FUpzr.shape[1]]

    Unzr = (dp / dn) * (fftshift(ifft2(fftshift(Upzr))) * N)
    Ik = np.abs(Unzr) ** 2

    if np.max(Ik) != np.min(Ik):
        contrast_k = (np.max(Ik) - np.min(Ik)) / (np.max(Ik) + np.min(Ik))
    else:
        contrast_k = 0
    kk.append(contrast_k)

    n_val = i / (2 * n_max)
    s_val = n_val * (lambda_ / A)
    s.append(s_val)
    count += 1

sharp_drop_index = np.argmax(np.array(s) > 0.3)  # Пространственная частота 0.3 была выбрана в качестве примера

plt.figure()
plt.plot(s[:sharp_drop_index], k[:sharp_drop_index], color='blue', linewidth=2)
plt.xlim([0, 2.5])
plt.ylim([0, 1.0])
plt.grid(True)
plt.title('Функция передачи модуляции для некогерентных изображений')
plt.xlabel('s, к.е.')

plt.figure()
plt.plot(s, kk, color='blue', linewidth=2)
plt.xlim([0, 2.5])
plt.ylim([0, 1.0])
plt.grid(True)
plt.title('Функция передачи модуляции для когерентных изображений (контраст)')
plt.xlabel('s, к.е.')

print(f"Пространственная частота с резким падением ФПМ: {s[sharp_drop_index]} к.е.")

plt.show()
