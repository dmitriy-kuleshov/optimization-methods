import math


# method1(middle_point)
def f(x):
    return math.exp(1 / x) + math.log(x)


def midpoint_search(a, b, eps):
    k = 0

    while True:
        x_ = (a + b) / 2
        f_ = derivative(x_)

        if abs(f_) <= eps:
            x_star, f_star = x_, f(x_)
            break
        elif f_ > 0:
            b = x_
        else:
            a = x_

        k += 1

    return x_star, f_star, k


def derivative(x):
    h = 1e-5
    return (f(x + h) - f(x - h)) / (2 * h)


a_initial = 1
b_initial = 3
epsilon = 0.0012

result_x, result_f, num_iterations = midpoint_search(a_initial, b_initial, epsilon)

print(f"Минимум функции {f.__name__} = {result_f} при x = {result_x}")
print(f"Количество итераций: {num_iterations}")
print(f"Количество вычислений равно количеству итераций в данном методе их: {num_iterations}")


# method2(hoard)
def derivative_f(x):
    return -(math.exp(1 / x) / (x ** 2)) + 1 / x


def chord_method(a, b, eps):
    L = [a, b]
    k = 0
    function_evaluations = 0

    while True:
        fk_a = derivative_f(L[0])
        fk_b = derivative_f(L[1])
        function_evaluations += 2

        if fk_a * fk_b < 0:
            xk = L[0] - (fk_a * (L[1] - L[0])) / (fk_b - fk_a)
            fk_xk = derivative_f(xk)
            function_evaluations += 1

            if abs(fk_xk) <= eps:
                print(f"Минимум функции: f = {f(xk)} при x = {xk}")
                break
            elif fk_xk > eps:
                L = [L[0], xk]
            elif fk_xk <= 0:
                L = [xk, L[1]]
        else:
            if fk_a > 0 and fk_b > 0:
                xk = L[0]
            elif fk_a < 0 and fk_b < 0:
                xk = L[1]
            else:
                xk = L[0] if fk_a == 0 else L[1]

            print(f"Минимум функции: f = {f(xk)} при x = {xk}")
            break

        k += 1

    print(f"Количество итераций: {k}")
    print(f"Количество вычислений функции: {function_evaluations}")


a = 1
b = 3
epsilon = 0.0012

chord_method(a, b, epsilon)
