import pandas as pd
import matplotlib.pyplot as plt

FILE_NAME = "Oschadbank - Архів курсу валюти для відділень 12-04-2026 (USD) (2).xlsx"
COLUMN_NUMBER = 5

df = pd.read_excel(FILE_NAME)
data = pd.to_numeric(df.iloc[:, COLUMN_NUMBER - 1], errors="coerce").dropna()
column_name = df.columns[COLUMN_NUMBER - 1]

mean_value = data.mean()
variance_value = data.var(ddof=1)
std_value = data.std(ddof=1)

print(f"Selected column: {column_name}")
print(f"Number of values: {len(data)}")
print(f"Mean: {mean_value:.6f}")
print(f"Variance: {variance_value:.6f}")
print(f"Standard deviation: {std_value:.6f}")

plt.figure(figsize=(8, 5))
plt.hist(data, bins=15, edgecolor="black")
plt.title(f"Distribution histogram for column: {column_name}")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()