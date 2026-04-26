import numpy as np
import matplotlib.pyplot as plt

RANDOM_SEED = 42
N = 100
TRUE_LEVEL = 50.0
NOISE_AMPLITUDE = 2.0
ANOMALY_COUNT = 8
ANOMALY_MIN_SHIFT = 12.0
ANOMALY_MAX_SHIFT = 20.0
MAX_POLY_DEGREE = 5
TRAIN_RATIO = 0.7
MAD_THRESHOLD = 3.5
MODEL_TOLERANCE = 0.05

np.random.seed(RANDOM_SEED)

t = np.arange(N, dtype=float)
y_true_trend = np.full(N, TRUE_LEVEL, dtype=float)
noise = np.random.uniform(-NOISE_AMPLITUDE, NOISE_AMPLITUDE, size=N)
y_base = y_true_trend + noise

anomaly_indices = np.random.choice(N, size=ANOMALY_COUNT, replace=False)
anomaly_shifts = np.random.uniform(ANOMALY_MIN_SHIFT, ANOMALY_MAX_SHIFT, size=ANOMALY_COUNT)
anomaly_signs = np.random.choice([-1.0, 1.0], size=ANOMALY_COUNT)

y_with_anomalies = y_base.copy()
y_with_anomalies[anomaly_indices] += anomaly_shifts * anomaly_signs
anomaly_indices = np.sort(anomaly_indices)

median_value = np.median(y_with_anomalies)
absolute_deviation = np.abs(y_with_anomalies - median_value)
mad_value = np.median(absolute_deviation)

if np.isclose(mad_value, 0.0):
    std_value = np.std(y_with_anomalies)
    if np.isclose(std_value, 0.0):
        detected_mask = np.zeros_like(y_with_anomalies, dtype=bool)
    else:
        detected_mask = np.abs((y_with_anomalies - np.mean(y_with_anomalies)) / std_value) > MAD_THRESHOLD
else:
    detected_mask = np.abs(0.6745 * (y_with_anomalies - median_value) / mad_value) > MAD_THRESHOLD

y_cleaned = y_with_anomalies.copy()
good_indices = np.where(~detected_mask)[0]
bad_indices = np.where(detected_mask)[0]

if len(good_indices) < 2:
    raise ValueError("Not enough valid points for interpolation.")

y_cleaned[bad_indices] = np.interp(t[bad_indices], t[good_indices], y_with_anomalies[good_indices])

split_index = int(N * TRAIN_RATIO)
x_train = t[:split_index]
y_train = y_cleaned[:split_index]
x_val = t[split_index:]
y_val = y_cleaned[split_index:]

model_results = []

for degree in range(MAX_POLY_DEGREE + 1):
    X_train = np.vander(x_train, N=degree + 1, increasing=False)
    coeffs, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)

    X_val = np.vander(x_val, N=degree + 1, increasing=False)
    y_train_pred = X_train @ coeffs
    y_val_pred = X_val @ coeffs

    train_rmse = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
    val_rmse = np.sqrt(np.mean((y_val - y_val_pred) ** 2))
    train_mae = np.mean(np.abs(y_train - y_train_pred))
    val_mae = np.mean(np.abs(y_val - y_val_pred))

    model_results.append({
        "degree": degree,
        "coeffs": coeffs,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "train_mae": train_mae,
        "val_mae": val_mae
    })

min_val_rmse = min(item["val_rmse"] for item in model_results)
candidate_models = [item for item in model_results if item["val_rmse"] <= min_val_rmse * (1 + MODEL_TOLERANCE)]
best_model = min(candidate_models, key=lambda item: item["degree"])

final_degree = best_model["degree"]
X_full = np.vander(t, N=final_degree + 1, increasing=False)
final_coeffs, _, _, _ = np.linalg.lstsq(X_full, y_cleaned, rcond=None)
y_fit_cleaned = X_full @ final_coeffs

obs_rmse_raw = np.sqrt(np.mean((y_true_trend - y_with_anomalies) ** 2))
obs_rmse_cleaned = np.sqrt(np.mean((y_true_trend - y_cleaned) ** 2))
obs_rmse_model = np.sqrt(np.mean((y_true_trend - y_fit_cleaned) ** 2))
obs_mae_model = np.mean(np.abs(y_true_trend - y_fit_cleaned))

forecast_horizon = N // 2
t_future = np.arange(N, N + forecast_horizon, dtype=float)
X_future = np.vander(t_future, N=final_degree + 1, increasing=False)
y_future_true = np.full(forecast_horizon, TRUE_LEVEL, dtype=float)
y_future_pred = X_future @ final_coeffs
forecast_rmse = np.sqrt(np.mean((y_future_true - y_future_pred) ** 2))

true_mask = np.zeros(N, dtype=bool)
true_mask[anomaly_indices] = True
tp = np.sum(true_mask & detected_mask)
fp = np.sum(~true_mask & detected_mask)
fn = np.sum(true_mask & ~detected_mask)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

print("Selected model:", f"polynomial degree {final_degree}")
print("Validation RMSE:", f"{min_val_rmse:.6f}")
print("Raw RMSE:", f"{obs_rmse_raw:.6f}")
print("Cleaned RMSE:", f"{obs_rmse_cleaned:.6f}")
print("Model RMSE:", f"{obs_rmse_model:.6f}")
print("Model MAE:", f"{obs_mae_model:.6f}")
print("Forecast RMSE:", f"{forecast_rmse:.6f}")
print("Anomaly detection precision:", f"{precision:.4f}")
print("Anomaly detection recall:", f"{recall:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(t, y_true_trend, label="True trend", linewidth=2)
plt.scatter(t, y_with_anomalies, label="Data with anomalies", s=30)
plt.plot(t, y_cleaned, label="Cleaned data", linewidth=2)
plt.plot(t, y_fit_cleaned, label=f"Polynomial model (deg={final_degree})", linewidth=2)
plt.title("Polynomial Regression After Anomaly Cleaning")
plt.xlabel("t")
plt.ylabel("y")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(12, 6))
plt.plot(t_future, y_future_true, label="True future value", linestyle='--', linewidth=2)
plt.plot(t_future, y_future_pred, label="Forecast", linewidth=2)
plt.axvline(x=N - 1, linestyle='--', linewidth=1.5, label="End of observation interval")
plt.title("Forecast Over 0.5 of the Observation Interval")
plt.xlabel("t")
plt.ylabel("y")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()
