import numpy as np
import matplotlib.pyplot as plt


def function(x1, x2):
    return 100 * (1 - x1 ** 2 - x2 ** 2) ** 2 + (x1 - 1) ** 2 + x2 ** 2


def gradient(x):
    x1, x2 = x
    df_x1 = 2 * (x1 - 1) - 400 * x1 * (-x1 ** 2 - x2 ** 2 + 1)
    df_x2 = 2 * (x2 - 1) - 400 * x2 * (-x2 ** 2 - x1 ** 2 + 1)
    return np.array([df_x1, df_x2])


def hessian_inverse(x):
    x1, x2 = x
    h11 = 2 + 1200 * x1 ** 2 - 400 * x2 ** 2
    h12 = -400 * x1 * x2
    h21 = -400 * x1 * x2
    h22 = 2 + 1200 * x2 ** 2 - 400 * x1 ** 2
    determinant = h11 * h22 - h12 * h21
    h_inv = np.array([[h22, -h12], [-h21, h11]]) / determinant
    return h_inv


def linear_search(x, dk):
    alpha = 1.0
    c = 0.5
    rho = 0.5
    while function(x[0] + alpha * dk[0], x[1] + alpha * dk[1]) > function(x[0], x[1]) + c * alpha * np.dot(gradient(x),
                                                                                                           dk):
        alpha *= rho
    return alpha


def newton_method(x0, epsilon_1, epsilon_2, iteration):
    x_k = np.array(x0)
    trajectory = [x_k]
    k = 0
    while True:
        grad = gradient(x_k)
        if np.linalg.norm(grad) < epsilon_1:
            x_star = x_k
            break
        if k >= iteration:
            x_star = x_k
            break
        h_inv = hessian_inverse(x_k)
        if np.all(np.linalg.eigvals(h_inv) > 0):
            d_k = -np.dot(h_inv, grad)
        else:
            d_k = -grad
        alpha_k = linear_search(x_k, d_k)
        x_k1 = x_k + alpha_k * d_k
        trajectory.append(x_k1)
        if np.linalg.norm(x_k1 - x_k) <= epsilon_2 and abs(function(*x_k1) - function(*x_k)) <= epsilon_2:
            x_star = x_k1
            break
        x_k = x_k1
        k += 1
    return x_star, trajectory


x0 = [0.3, 0.5]
epsilon_1 = 0.001
epsilon_2 = 0.002
iteration = 3

result, trajectory = newton_method(x0, epsilon_1, epsilon_2, iteration)
print(result, function(*result))


x1 = np.linspace(-2, 2, 100)
x2 = np.linspace(-2, 2, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = function(X1, X2)

plt.contour(X1, X2, Z, levels=50)
plt.plot(*zip(*trajectory), marker='o', color='r', label='Траектория')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Траектория движения к экстремуму и график функции')
plt.legend()
plt.colorbar()
plt.show()
