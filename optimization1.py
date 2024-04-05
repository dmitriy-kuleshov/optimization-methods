import math


def f(x):
    return -x * math.exp(-x)


a = -2
b = 6
eps = 0.001

# 1 Метод (метод равномерного поиска)
L = [a, b]
N = math.ceil((b - a) / eps)
print("Число итераций: %d" % (N + 1))
print("Количество вычислений функции: %d" % (N + 1))


def xi(i):
    return a + i * (b - a) / N


k = 0
f_min = f(a)

for i in range(N + 1):
    if f(xi(i)) < f_min:
        f_min = f(xi(i))
        k = i

print("Найденное решение: ", xi(k))
print("Значение функции: ", f_min)
print()


# 3 Метод (метод деления отрезка пополам)
def second(a, b, eps):
    iterations = 0
    count_z = 0

    while (b - a) > eps:
        L = b - a
        xm = (a + b) / 2
        x1 = a + L / 4
        x2 = b - L / 4

        f_xm = f(xm)
        f_x1 = f(x1)
        f_x2 = f(x2)

        count_z += 3

        if f_x1 < f_xm:
            b = xm
            xm = x1
        elif f_x2 < f_xm:
            a = xm
            xm = x2
        else:
            a = x1
            b = x2

        iterations += 1

    min_point = (a + b) / 2
    min_value = f(min_point)

    return min_point, min_value, iterations, count_z


min_point, min_value, iterations, count_z = second(a, b, eps)

print("Число итераций:", iterations)
print("Количество вычислений функции:", count_z)
print("Найденное решение", min_point)
print("Значение функции", min_value)
print()


# 8 Метод (метод парабол)
def parabolic(a, b, eps):
    g = 0  # Число итераций
    call_count = 0

    while abs(b - a) > eps:

        x1 = a
        x2 = (a + b) / 2
        x3 = b

        f1 = f(x1)
        f2 = f(x2)
        f3 = f(x3)

        a0 = f1
        a1 = (f2 - f1) / (x2 - x1)
        a2 = ((f3 - f1) / (x3 - x1) - (f2 - f1) / (x2 - x1)) / (x3 - x2)

        x_min = 0.5 * (x1 + x2 - a1 / a2)

        f_minimal = f(x_min)
        call_count += 1

        if x_min < x2:
            if f_minimal < f2:
                b = x2
            else:
                a = x_min
        else:
            if f_minimal < f2:
                a = x2
            else:
                b = x_min

        g += 1

    return x_min, f_minimal, g, call_count


min_point, min_value, iterations, function_evaluations = parabolic(a, b, eps)

print("Число итераций:", iterations)
print("Количество вычислений функции:", function_evaluations)
print("Найденное решение:", min_point)
print("Значение функции:", min_value)
