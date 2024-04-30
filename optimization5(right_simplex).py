import numpy as np
import matplotlib.pyplot as plt


def objective_function(x):
    return 100 * (1 - x[0] ** 2 - x[1] ** 2) ** 2 + (x[0] - 1) ** 2 + x[1] ** 2


eps = 1e-6
x0 = np.array([0.8, 0.08])
a = 1.0

n = len(x0)

d1 = a * (np.sqrt(n + 1) - 1) / (n * np.sqrt(2))
d2 = a * (np.sqrt(n + 1) + n - 1) / (n * np.sqrt(2))
vertices = [x0]
for i in range(n):
    vertex = x0.copy()
    vertex[i] += d1
    vertices.append(vertex)
vertex_0 = x0.copy()
vertex_0[0] += d2
vertices[0] = vertex_0

values = [objective_function(vertex) for vertex in vertices]

trajectory = [vertices[0].copy()]
iteration = 0
while True:
    sorted_indices = np.argsort(values)
    vertices = [vertices[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]

    sum_sq_diff = np.sum([(values[i] - values[0]) ** 2 for i in range(1, n + 1)])
    if (sum_sq_diff / n) ** 0.5 < eps:
        break

    centroid = np.mean(vertices[:-1], axis=0)

    x_r = centroid + (centroid - vertices[-1])
    f_r = objective_function(x_r)

    if values[0] <= f_r < values[-2]:
        vertices[-1] = x_r
        values[-1] = f_r
    elif f_r < values[0]:
        x_e = centroid + 2 * (x_r - centroid)
        f_e = objective_function(x_e)
        if f_e < f_r:
            vertices[-1] = x_e
            values[-1] = f_e
        else:
            vertices[-1] = x_r
            values[-1] = f_r
    else:
        x_c = centroid + 0.5 * (vertices[-1] - centroid)
        f_c = objective_function(x_c)
        if f_c < values[-1]:
            vertices[-1] = x_c
            values[-1] = f_c
        else:
            for i in range(1, n + 1):
                vertices[i] = vertices[0] + 0.5 * (vertices[i] - vertices[0])
                values[i] = objective_function(vertices[i])

    trajectory.append(vertices[0].copy())
    iteration += 1

x1 = np.linspace(-2, 2, 400)
x2 = np.linspace(-2, 2, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = np.zeros_like(X1)
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        Z[i, j] = objective_function([X1[i, j], X2[i, j]])

trajectory = np.array(trajectory)
min_point = vertices[0]

plt.figure(figsize=(10, 6))
plt.contour(X1, X2, Z, levels=50)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('График функции и траектория движения к минимуму')
plt.grid(True)

plt.plot(trajectory[:, 0], trajectory[:, 1], '-o', markersize=3, color='red', label='Траектория')
plt.plot(min_point[0], min_point[1], 'go', markersize=8, label='Минимум')

plt.legend()
plt.show()

print("\nРезультаты:")
print("Минимум функции:", min_point)
print("Значение функции в минимуме:", values[0])
