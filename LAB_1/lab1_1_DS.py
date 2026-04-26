import numpy as np
import matplotlib.pyplot as plt

N = 200
scale = 2.0
a = 0.01
b = 0.3
c = 10

np.random.seed(42)
error_sample = np.random.exponential(scale=scale, size=N)
x = np.arange(N)
trend = a * x**2 + b * x + c
centered_error = error_sample - np.mean(error_sample)
additive_sample = trend + centered_error

def print_stats(name, data):
    mean_value = np.mean(data)
    variance_value = np.var(data, ddof=1)
    std_value = np.std(data, ddof=1)

    print(f"\n{name}")
    print(f"Mean: {mean_value:.6f}")
    print(f"Variance: {variance_value:.6f}")
    print(f"Standard deviation: {std_value:.6f}")

print_stats("Exponential random variable", error_sample)
print_stats("Additive sample (quadratic trend + error)", additive_sample)

plt.figure(figsize=(8, 5))
plt.hist(error_sample, bins=15, edgecolor="black")
plt.title("Histogram of exponential random variable")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(additive_sample, bins=15, edgecolor="black")
plt.title("Histogram of additive sample")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(x, trend, label="Quadratic trend")
plt.plot(x, additive_sample, label="Additive sample", alpha=0.8)
plt.title("Quadratic trend and additive model")
plt.xlabel("Observation number")
plt.ylabel("Value")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()