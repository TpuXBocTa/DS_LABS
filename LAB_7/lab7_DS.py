import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_excel("Data_Set_9.xlsx")

os.makedirs("results", exist_ok=True)

data["Amount"] = data["Quantity"] * data["Price"]

status_probability = {
    "won": 1.0,
    "presented": 0.6,
    "pending": 0.4,
    "declined": 0.0
}

data["Probability"] = data["Status"].map(status_probability)
data["ExpectedAmount"] = data["Amount"] * data["Probability"]

summary = pd.DataFrame({
    "Indicator": [
        "Total potential amount",
        "Won amount",
        "Expected forecast amount",
        "Number of records",
        "Number of companies"
    ],
    "Value": [
        data["Amount"].sum(),
        data[data["Status"] == "won"]["Amount"].sum(),
        data["ExpectedAmount"].sum(),
        len(data),
        data["Name"].nunique()
    ]
})

status_analysis = data.groupby("Status").agg({
    "Status": "count",
    "Quantity": "sum",
    "Amount": "sum",
    "ExpectedAmount": "sum"
})

status_analysis = status_analysis.rename(columns={
    "Status": "Records"
})

product_analysis = data.groupby("Product").agg({
    "Product": "count",
    "Quantity": "sum",
    "Amount": "sum",
    "ExpectedAmount": "sum"
})

product_analysis = product_analysis.rename(columns={
    "Product": "Records"
})

rep_analysis = data.groupby("Rep").agg({
    "Rep": "count",
    "Amount": "sum",
    "ExpectedAmount": "sum"
})

rep_analysis = rep_analysis.rename(columns={
    "Rep": "Records"
})

x = np.arange(1, len(data) + 1)
y = data["ExpectedAmount"].values

trend_coefficients = np.polyfit(x, y, 1)
trend_line = np.poly1d(trend_coefficients)

future_x = np.arange(len(data) + 1, len(data) + 6)
future_y = trend_line(future_x)
future_y = np.maximum(future_y, 0)

forecast = pd.DataFrame({
    "Observation": future_x,
    "ForecastExpectedAmount": future_y
})

data.to_csv("results/prepared_data.csv", index=False, encoding="utf-8-sig")
summary.to_csv("results/summary.csv", index=False, encoding="utf-8-sig")
status_analysis.to_csv("results/status_analysis.csv", encoding="utf-8-sig")
product_analysis.to_csv("results/product_analysis.csv", encoding="utf-8-sig")
rep_analysis.to_csv("results/rep_analysis.csv", encoding="utf-8-sig")
forecast.to_csv("results/forecast.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(9, 5))
plt.bar(status_analysis.index, status_analysis["Amount"])
plt.title("Amount by Status")
plt.xlabel("Status")
plt.ylabel("Amount")
plt.tight_layout()
plt.savefig("results/status_amount.png", dpi=300)
plt.close()

plt.figure(figsize=(9, 5))
plt.bar(product_analysis.index, product_analysis["Amount"])
plt.title("Amount by Product")
plt.xlabel("Product")
plt.ylabel("Amount")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("results/product_amount.png", dpi=300)
plt.close()

plt.figure(figsize=(9, 5))
plt.bar(rep_analysis.index, rep_analysis["ExpectedAmount"])
plt.title("Expected Amount by Representative")
plt.xlabel("Representative")
plt.ylabel("Expected Amount")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("results/rep_expected_amount.png", dpi=300)
plt.close()

plt.figure(figsize=(9, 5))
plt.plot(x, y, marker="o", label="Actual expected amount")
plt.plot(future_x, future_y, marker="o", linestyle="--", label="Forecast")
plt.title("Forecast of Expected Amount")
plt.xlabel("Observation")
plt.ylabel("Expected Amount")
plt.legend()
plt.tight_layout()
plt.savefig("results/forecast.png", dpi=300)
plt.close()

print("Initial data")
print(data)

print("\nSummary")
print(summary)

print("\nStatus analysis")
print(status_analysis)

print("\nProduct analysis")
print(product_analysis)

print("\nRepresentative analysis")
print(rep_analysis)

print("\nForecast")
print(forecast)