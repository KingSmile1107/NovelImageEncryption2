import numpy as np
import matplotlib.pyplot as plt


def cubic_map(x):
    return x ** 3 - x  # Cubic 映射公式


def lyapunov_exponent(f, x0, n_iter=1000):
    x = x0
    exponents = []

    # 初始斜率
    dx = 1e-6

    for _ in range(n_iter):
        # 计算函数和其导数
        x_next = f(x)
        f_prime = 3 * x ** 2 - 1  # 求导数
        exponent = np.log(abs(f_prime))  # Lyapunov 指数
        exponents.append(exponent)

        # 更新 x
        x = x_next

    return np.mean(exponents)  # 返回 Lyapunov 指数的平均值


# 绘制 Lyapunov 指数图
x_vals = np.linspace(-2, 2, 1000)  # x 值范围
lyapunov_vals = [lyapunov_exponent(cubic_map, x) for x in x_vals]  # 计算每个 x 值的 Lyapunov 指数

plt.plot(x_vals, lyapunov_vals, label="Lyapunov Exponent")
plt.title("Lyapunov Exponent of Cubic Map (x^3 - x)")
plt.xlabel("x")
plt.ylabel("Lyapunov Exponent")
plt.grid(True)
plt.show()
