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

x1 = np.linspace(-1.1, 1.1, 400)
x2 = np.linspace(-1.1, 1.1, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = f(X1, X2)

plt.figure(figsize=(10, 6))
plt.contour(X1, X2, Z, levels=50)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('График функции')
plt.grid(True)

x1_values = trajectory[:, 0]
x2_values = trajectory[:, 1]
plt.plot(x1_values, x2_values, '-o', markersize=3, color='red')

plt.show()
