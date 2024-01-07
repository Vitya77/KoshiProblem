import numpy as np
from matplotlib import pyplot as plt


#Задаємо задачу Коші
def f(x, y):
    return 2 * y + 10 * (3 * np.pi * np.cos(3 * np.pi * x) - 2 * np.sin(3 * np.pi * x))

def ytrue(x):
    return np.exp(2 * x) + 10 * np.sin(3 * np.pi * x) - 0.5

y0 = 1/2


#Метод Рунге-Кутта 3 порядку, формула А
def runge_kutta_3A(x, y0, func):
    y = np.asarray([y0])
    for x_prev, x_next in zip(x[:-1:], x[1:]):
        h = x_next - x_prev
        y_prev = y[-1]
        k1 = h*func(x_prev, y_prev)
        k2 = h*func(x_prev + h/2, y_prev + k1/2)
        k3 = h*func(x_prev + h, y_prev - k1 + 2*k2)

        y_next = y_prev + (k1 + 4*k2 + k3)/6
        y = np.append(y, y_next)

    return y

#Метод Рунге-Кутта 4 порядку, формула В
def runge_kutta_4B(x, y0, func):
    y = np.asarray([y0])
    for x_prev, x_next in zip(x[:-1:], x[1:]):
        h = x_next - x_prev
        y_prev = y[-1]
        k1 = h*func(x_prev, y_prev)
        k2 = h*func(x_prev + h/4, y_prev + k1/4)
        k3 = h*func(x_prev + h/2, y_prev + k2/2)
        k4 = h*func(x_prev + h, y_prev + k1 - 2*k2 + 2*k3)

        y_next = y_prev + (k1 + 4*k3 + k4)/6
        y = np.append(y, y_next)

    return y

#Визначення таблиці скінченних різниць
def finiteDifference(y):
    if len(y) > 1:
        fd = finiteDifference(np.diff(y))
        fd.append(y)
        return fd
    else:
        return [y]

#Екстраполяційний метод Адамса 4 порядку
def adams_ecs4(x, y, func):
    a1 = 1/2
    a2 = 5/12
    a3 = 3/8

    for i in range(3, len(x)-1):
        h = x[i+1] - x[i]
        fd = finiteDifference(func(x[0:i+1], y))[::-1]
        y_next = y[i] + h*(fd[0][i] + a1*fd[1][i-1] + a2*fd[2][i-2] + a3*fd[3][i-3])
        y = np.append(y, y_next)

    return y

#Інтерполяційний метод Адамса 3 порядку
def adams_inter3(x, y, func):
    a1 = -1/2
    a2 = -1/12

    for i in range(1, len(x)-1):
        h = x[i+1] - x[i]
        y = np.append(y, y[-1])

        for j in range(10):
            fd = finiteDifference(func(x[0:i+2], y))[::-1]
            y[i+1] = y[i] + h*(fd[0][i+1] + a1*fd[1][i] + a2*fd[2][i-1])

    return y


#Визначаємо ікси з кроком 1/N
N = 40
x = np.linspace(0, 1, N)

solve_runge_3A = runge_kutta_3A(x, y0, f)

plt.title(label='Розв\'язок методом Рунге-Кутта 3 порядку, формулою А')
plt.plot(x, solve_runge_3A, label='Знайдена функція y')
plt.plot(x, ytrue(x), label='Справжня функція y')
plt.legend()
plt.show()
plt.cla()

print(f'RMSE для розв\'язку методом Рунге-Кутта 3 порядку, формулою А: {np.linalg.norm(solve_runge_3A - ytrue(x))}')
print(f'|D(xn)| для розв\'язку методом Рунге-Кутта 3 порядку, формулою А: {np.abs(solve_runge_3A[-1] - ytrue(x[-1]))}')

solve_runge_4B = runge_kutta_4B(x, y0, f)

plt.title(label='Розв\'язок методом Рунге-Кутта 4 порядку, формулою B')
plt.plot(x, solve_runge_4B, label='Знайдена функція y')
plt.plot(x, ytrue(x), label='Справжня функція y')
plt.legend()
plt.show()
plt.cla()

print(f'RMSE для розв\'язку методом Рунге-Кутта 4 порядку, формулою B: {np.linalg.norm(solve_runge_4B - ytrue(x))}')
print(f'|D(xn)| для розв\'язку методом Рунге-Кутта 4 порядку, формулою B: {np.abs(solve_runge_4B[-1] - ytrue(x[-1]))}')

y = solve_runge_3A[:4]
solve_adans_ecs = adams_ecs4(x, y, f)

plt.title(label='Розв\'язок екстраполяційним методом Адамса 4 порядку')
plt.plot(x, solve_adans_ecs, label='Знайдена функція y')
plt.plot(x, ytrue(x), label='Справжня функція y')
plt.legend()
plt.show()
plt.cla()

print(f'RMSE для розв\'язку екстраполяційним методом Адамса 4 порядку: {np.linalg.norm(solve_adans_ecs - ytrue(x))}')
print(f'|D(xn)| для розв\'язку екстраполяційним методом Адамса 4 порядку: {np.abs(solve_adans_ecs[-1] - ytrue(x[-1]))}')

y = solve_runge_3A[:2]
solve_adams_inter = adams_inter3(x, y, f)

plt.title(label='Розв\'язок інтерполяційним методом Адамса 3 порядку')
plt.plot(x, solve_adams_inter, label='Знайдена функція y')
plt.plot(x, ytrue(x), label='Справжня функція y')
plt.legend()
plt.show()
plt.cla()

print(f'RMSE для розв\'язку інтерполяційним методом Адамса 3 порядку: {np.linalg.norm(solve_adams_inter - ytrue(x))}')
print(f'|D(xn)| для розв\'язку інтерполяційним методом Адамса 3 порядку: {np.abs(solve_adams_inter[-1] - ytrue(x[-1]))}')