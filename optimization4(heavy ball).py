import numpy as np
import matplotlib.pyplot as plt


def f(x1, x2):
    return 100 * (1 - x1 ** 2 - x2 ** 2) ** 2 + (x1 - 1) ** 2 + x2 ** 2


def gradient(x1, x2):
    dfdx1 = -400 * x1 * (1 - x1 ** 2 - x2 ** 2) + 2 * x1 - 2
    dfdx2 = -400 * x2 * (1 - x1 ** 2 - x2 ** 2) + 2 * x2
    return np.array([dfdx1, dfdx2])


def heavy_ball_method(initial_point, learning_rate, momentum, num_iterations):
    x = initial_point
    v = np.zeros_like(x)
    trajectory = [x.copy()]

    for i in range(num_iterations):
        grad = gradient(x[0], x[1])
        v_prev = v
        v = momentum * v - learning_rate * grad
        x = x - v_prev + (1 + momentum) * v
        trajectory.append(x.copy())

    return x, np.array(trajectory)


initial_point = np.array([0.55, 0.11])  # начальная точка
learning_rate = 0.001  # скорость обучения
momentum = 0.5  # параметр инерции
num_iterations = 1000  # количество итераций

result, trajectory = heavy_ball_method(initial_point, learning_rate, momentum, num_iterations)

print("Минимум функции достигается в точке:", result)
print("Значение функции в минимуме:", f(result[0], result[1]))

x1 = np.linspace(-1.5, 1.5, 100)
x2 = np.linspace(-1.5, 1.5, 100)
X, Y = np.meshgrid(x1, x2)
Z = f(X, Y)

plt.figure(figsize=(8, 6))
plt.contourf(X, Y, Z, levels=np.logspace(0, 5, 35), cmap='gray', alpha=0.3)
plt.plot(trajectory[:, 0], trajectory[:, 1], '-o', color='red')  # Траектория движения к минимуму
plt.title('Траектория движения к минимуму')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()