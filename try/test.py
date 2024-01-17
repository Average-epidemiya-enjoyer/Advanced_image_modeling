import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift


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

nx, ny = np.meshgrid(np.arange(-n_max, n_max, dn), np.arange(-n_max, n_max, dn))
px, py = np.meshgrid(np.arange(-p_max, p_max, dp), np.arange(-p_max, p_max, dp))
x, y = np.meshgrid(np.arange(-x_max, x_max, dx), np.arange(-x_max, x_max, dx))

X, Y = np.meshgrid(np.arange(-N // 2, N // 2), np.arange(-N // 2, N // 2))
R = np.sqrt(X ** 2 + Y ** 2)
pupil = R <= N / Dzr
Fzr = pupil.astype(float)

k = []
kk = []
s = []

for i in range(1, 93, 3):
    Subject = comb(i, N)
    FUp = 2 * (dn / dp) * fftshift(fft2(fftshift(np.abs(Subject) ** 2))) / N

    FUp = FUp * (1 - 0.25 * np.abs(x - 1))

    hn = (dp / dn) * fftshift(ifft2(fftshift(Fzr))) * N
    hn = np.abs(hn) ** 2 / np.pi ** 2
    Fhp = (dn / dp) * fftshift(fft2(fftshift(hn))) / N * np.pi
    FUpzr = FUp * Fhp
    Ink = (dp / dn) * fftshift(ifft2(fftshift(FUpzr))) * N
    if np.max(Ink) != np.min(Ink):
        contrast_nk = (np.max(Ink) - np.min(Ink)) / (np.max(Ink) + np.min(Ink))
    else:
        contrast_nk = 0
    k.append(contrast_nk)

    Up = (dn / dp) * fftshift(fft2(fftshift(Subject))) / N
    Upzr = Up * Fzr
    Unzr = (dp / dn) * fftshift(ifft2(fftshift(Upzr))) * N
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

plt.figure()
plt.plot(s, k, color='blue', linewidth=2)
plt.xlim([0, 2.5])
plt.grid(True)
plt.title('Функция передачи модуляции для некогерентных изображений')
plt.xlabel('s, к.е.')

sharp_drop_index = np.argmax(np.array(s) > 0.3)  # Пространственная частота 0.3 была выбрана в качестве примера

plt.figure()
plt.plot(s, kk, color='blue', linewidth=2)
plt.xlim([0, 2.5])
plt.grid(True)
plt.title('Функция передачи модуляции для когерентных изображений (контраст)')
plt.xlabel('s, к.е.')

# Вывод значения пространственной частоты, после которой ФПМ резко падает
print(f"Пространственная частота с резким падением ФПМ: {s[sharp_drop_index]} к.е.")

plt.show()
