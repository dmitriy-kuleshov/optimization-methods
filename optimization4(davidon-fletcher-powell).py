import numpy as np
import matplotlib.pyplot as plt


def f(x1, x2):
    return 100 * (1 - x1 ** 2 - x2 ** 2) ** 2 + (x1 - 1) ** 2 + x2 ** 2


def grad_f(x1, x2):
    df_dx1 = -400 * x1 * (1 - x1 ** 2 - x2 ** 2) - 2 * (1 - x1)
    df_dx2 = -400 * x2 * (1 - x1 ** 2 - x2 ** 2) + 2 * x2
    return np.array([df_dx1, df_dx2])


def DFP(f, grad_f, x0, epsilon=1e-5, max_iterations=100):
    n = len(x0)
    H = np.eye(n)

    x = x0
    iteration = 0
    trajectory = [x]
    while np.linalg.norm(grad_f(*x)) > epsilon and iteration < max_iterations:
        gradient = grad_f(*x)

        p = -np.dot(H, gradient)

        alpha = 0.01
        while f(*(x + alpha * p)) > f(*x):
            alpha *= 0.5

        s = alpha * p
        x_new = x + s

        y = grad_f(*x_new) - grad_f(*x)

        Hy = np.dot(H, y)
        yHy = np.dot(y, Hy)

        if yHy == 0:
            break

        H = H - np.outer(s, np.dot(H, s)) / np.dot(s, Hy) + np.outer(y, y) / yHy

        x = x_new
        iteration += 1
        trajectory.append(x)

    return x, f(*x), trajectory


x0 = np.array([0.88, 0.02])

minimum, min_value, trajectory = DFP(f, grad_f, x0)
print("Минимум найден в точке:", minimum)
print("Значение функции в минимуме:", min_value)

x1_vals = np.linspace(-1, 1, 400)
x2_vals = np.linspace(-1, 1, 400)
x1_mesh, x2_mesh = np.meshgrid(x1_vals, x2_vals)
z_mesh = f(x1_mesh, x2_mesh)

plt.figure()

plt.contour(x1_mesh, x2_mesh, z_mesh, levels=50)

trajectory = np.array(trajectory)
plt.plot(trajectory[:, 0], trajectory[:, 1], color='r', marker='o', label='Trajectory')

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Траектория движения к минимуму')

plt.legend()
plt.grid(True)
plt.show()
