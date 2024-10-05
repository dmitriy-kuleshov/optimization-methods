import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


def f(x):
    return (x[0] + x[1]) ** 4 + 4 * x[1] ** 2


def grad_f(x):
    return np.array([4 * (x[0] + x[1]) ** 3, 4 * (x[0] + x[1]) ** 3 + 8 * x[1]])


def conjugate_direction(x, grad_prev, grad_curr):
    beta = np.dot(grad_curr, grad_curr) / np.dot(grad_prev, grad_prev)
    return -grad_curr + beta * x


constraints = ({'type': 'ineq', 'fun': lambda x: np.array([x[0] + 2]), 'jac': lambda x: np.array([1.0, 0.0])},
               {'type': 'ineq', 'fun': lambda x: np.array([-x[0]]), 'jac': lambda x: np.array([-1.0, 0.0])},
               {'type': 'ineq', 'fun': lambda x: np.array([x[1] - 1]), 'jac': lambda x: np.array([0.0, 1.0])},
               {'type': 'ineq', 'fun': lambda x: np.array([-x[1] + 3]), 'jac': lambda x: np.array([0.0, -1.0])})

# Начальная точка в допустимом диапазоне
x0 = np.array([0.05, 0.889])

max_iter = 100
x = x0
grad_prev = grad_f(x)
k = 0

trajectory = [x]

while k < max_iter:
    pk = conjugate_direction(x, grad_prev, grad_f(x))

    res = minimize_scalar(lambda alpha: f(x + alpha * pk), bounds=(0, 1), method='bounded')
    alpha = res.x

    x_new = x + alpha * pk
    x_new[1] += 0.6
    x_new[0] += 0.1

    # Ограничение на диапазон
    x_new = np.clip(x_new, [-2, 1], [0, 3])

    if np.linalg.norm(x_new - x) < 1e-6:
        break

    trajectory.append(x_new)
    grad_prev = grad_f(x_new)
    x = x_new
    k += 1

trajectory = np.array(trajectory)

x1 = np.linspace(-2, 0, 400)
x2 = np.linspace(1, 3, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = f([X1, X2])

plt.contour(X1, X2, Z, levels=50, cmap='jet')
plt.colorbar(label='f(x)')
plt.xlabel('x1')
plt.ylabel('x2')

plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', color='red', label='Траектория движения')
plt.plot(x[0], x[1], marker='o', color='green', markersize=10, label='Точка минимума')

plt.title('Траектория движения к точке минимума')
plt.legend()
plt.grid(True)
plt.show()

print("Минимум найден в точке:", x)
print("Значение функции в этой точке:", f(x))
print("Число итераций:", k)

# Вывод координат каждой точки из траектории
print("Точки из траектории:")
for i, point in enumerate(trajectory):
    print(f"Точка {i + 1}: {point}")
