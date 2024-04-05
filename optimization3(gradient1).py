import matplotlib.pyplot as plt
import numpy as np


def function(x1, x2):
    return x1 ** 2 - x1 * x2 + x2 ** 2 + 3 * x1 - 2 * x2 + 1


def partial_1(x1, x2):
    return 2 * x1 - x2 + 3


def partial_2(x1, x2):
    return -x1 + 2 * x2 - 2


iteration = 500
epsilon_1 = 0.0001
epsilon_2 = 0.0002
k = 0
x0 = [0.5, 0]
x_history = [x0]

gradient = [partial_1(x0[0], x0[1]), partial_2(x0[0], x0[1])]

while True:
    if abs(gradient[0]) < epsilon_1 and abs(gradient[1]) < epsilon_1:
        x_end = x0
        break
    else:
        if k >= iteration:
            x_end = x0
            break
        else:
            gamma = 0.01
            x1_new = x0[0] - gamma * gradient[0]
            x2_new = x0[1] - gamma * gradient[1]
            x_new = [x1_new, x2_new]
            if function(x_new[0], x_new[1]) - function(x0[0], x0[1]) < 0:
                if abs(x_new[0] - x0[0]) <= epsilon_2 and abs(x_new[1] - x0[1]) <= epsilon_2 and \
                        abs(function(x_new[0], x_new[1]) - function(x0[0], x0[1])) <= epsilon_2:
                    x_end = x_new
                    break
                else:
                    x0 = x_new
                    x_history.append(x_new)
                    gradient = [partial_1(x0[0], x0[1]), partial_2(x0[0], x0[1])]
                    k += 1
            else:
                x_end = x0
                break
print(x_end, function(x_end[0], x_end[1]))
num_function_evaluations = len(x_history)
print("Количество вычислений функции:", num_function_evaluations)

x1 = np.linspace(-2, 2, 100)
x2 = np.linspace(-2, 2, 100)
X1, X2 = np.meshgrid(x1, x2)
Y = function(X1, X2)

plt.contour(X1, X2, Y, levels=50)
plt.plot([point[0] for point in x_history], [point[1] for point in x_history], '-ro')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Траектория движения к экстремуму и график функции')
plt.show()
