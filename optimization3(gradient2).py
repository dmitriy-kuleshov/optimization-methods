import math
import numpy as np
import matplotlib.pyplot as plt


def function(x1, x2):
    global function_calls
    function_calls += 1
    return x1 ** 2 - x1 * x2 + x2 ** 2 + 3 * x1 - 2 * x2 + 1


def partial_1(x1, x2):
    return 2 * x1 - x2 + 3


def partial_2(x1, x2):
    return -x1 + 2 * x2 - 2


def gradient_descent(x0, epsilon_1, epsilon_2, iteration):
    global function_calls
    k = 0
    x = x0
    trajectory = [x]
    while k < iteration:
        grad_f_x = [partial_1(x[0], x[1]), partial_2(x[0], x[1])]
        if math.sqrt(grad_f_x[0] ** 2 + grad_f_x[1] ** 2) < epsilon_1:
            return trajectory
        if k >= iteration - 1:
            return trajectory
        gamma_star = golden_section_search(x, grad_f_x)
        x_next = [x[i] - gamma_star * grad_f_x[i] for i in range(len(x))]
        if abs(x_next[0] - x[0]) <= epsilon_2 and abs(x_next[1] - x[1]) <= epsilon_2 and abs(
                function(x_next[0], x_next[1]) - function(x[0], x[1])) <= epsilon_2:
            trajectory.append(x_next)
            return trajectory
        trajectory.append(x_next)
        k += 1
        x = x_next
    return trajectory


def golden_section_search(x, grad_f_x):
    global function_calls
    a, b = 0, 1
    tau = (math.sqrt(5) - 1) / 2
    gamma_a = a + (1 - tau) * (b - a)
    gamma_b = a + tau * (b - a)
    while abs(b - a) > 0.001:
        f_gamma_a = function(x[0] - gamma_a * grad_f_x[0], x[1] - gamma_a * grad_f_x[1])
        f_gamma_b = function(x[0] - gamma_b * grad_f_x[0], x[1] - gamma_b * grad_f_x[1])
        if f_gamma_a < f_gamma_b:
            b = gamma_b
            gamma_b = gamma_a
            gamma_a = a + (1 - tau) * (b - a)
        else:
            a = gamma_a
            gamma_a = gamma_b
            gamma_b = a + tau * (b - a)
    return (a + b) / 2


x0 = [1, 0]
epsilon_1 = 0.000001
epsilon_2 = 0.000002
iteration = 500

function_calls = 0
trajectory = gradient_descent(x0, epsilon_1, epsilon_2, iteration)

trajectory = np.array(trajectory)

x1 = np.linspace(-2, 2, 400)
x2 = np.linspace(-2, 2, 400)
X1, X2 = np.meshgrid(x1, x2)
Y = function(X1, X2)

plt.contour(X1, X2, Y, levels=50)
plt.plot(trajectory[:, 0], trajectory[:, 1], color='red', marker='o')
plt.title('Траектория движения к экстремуму и график функции')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.show()

result = trajectory[-1]
print(result, function(result[0], result[1]))
print("Количество вычислений функции:", function_calls)
