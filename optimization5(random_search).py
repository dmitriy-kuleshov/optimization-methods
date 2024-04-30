import numpy as np
import matplotlib.pyplot as plt

def f(x1, x2):
    return 100 * (1 - x1 ** 2 - x2 ** 2) ** 2 + (x1 - 1) ** 2 + x2 ** 2

def random_search_with_backtracking(initial_point, delta=0.1, max_iterations=1000, tolerance=1e-6):
    current_point = initial_point
    num_evaluations = 0
    trajectory = [initial_point]

    for _ in range(max_iterations):
        delta_x1 = np.random.uniform(-1, 1) * delta
        delta_x2 = np.random.uniform(-1, 1) * delta

        new_point = (current_point[0] + delta_x1, current_point[1] + delta_x2)
        if f(*new_point) < f(*current_point):
            current_point = new_point
            num_evaluations += 1
            trajectory.append(new_point)
        else:
            delta *= 0.9

        if delta < tolerance:
            break

    return current_point, f(*current_point), num_evaluations, trajectory

initial_point = (0.5, 0.15)
minimum_point, minimum_value, num_evaluations, trajectory = random_search_with_backtracking(initial_point)

print("Минимальное значение функции:", minimum_value)
print("Координаты минимума:", minimum_point)
print("Количество вычислений функции:", num_evaluations)

# Построение графика функции и траектории движения
x1 = np.linspace(-1.1, 1.1, 400)
x2 = np.linspace(-1.1, 1.1, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = f(X1, X2)

plt.figure(figsize=(10, 6))
plt.contour(X1, X2, Z, levels=50)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('График функции и траектория движения к минимуму')
plt.grid(True)

x1_values = [point[0] for point in trajectory]
x2_values = [point[1] for point in trajectory]
plt.plot(x1_values, x2_values, '-o', markersize=3, color='red')
plt.plot(x1_values[-1], x2_values[-1], 'go', markersize=8, label='Конечная точка')

plt.show()
