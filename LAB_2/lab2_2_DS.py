
import numpy as np
import matplotlib.pyplot as plt

RANDOM_SEED = 42
N = 100
TRUE_LEVEL = 50.0
NOISE_AMPLITUDE = 2.0
ANOMALY_COUNT = 8
ANOMALY_MIN_SHIFT = 12.0
ANOMALY_MAX_SHIFT = 20.0
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

residual_clip = max(3.0 * NOISE_AMPLITUDE, 3.0 * np.std(y_cleaned))
velocity_limit = max(2.0 * NOISE_AMPLITUDE, 4.0 * np.std(np.diff(y_cleaned)))
second_diff = np.diff(np.diff(y_cleaned))
acceleration_limit = max(2.0 * NOISE_AMPLITUDE, 4.0 * np.std(second_diff) if len(second_diff) > 0 else 2.0 * NOISE_AMPLITUDE)

alpha_values = np.linspace(0.01, 0.30, 30)
beta_values = np.linspace(0.000, 0.050, 26)
gamma_values = np.linspace(0.000, 0.020, 21)

search_results = []

for alpha in alpha_values:
    for beta in beta_values:
        x_state = y_cleaned[0]
        v_state = 0.0
        filtered = np.zeros(N, dtype=float)
        filtered[0] = x_state

        for i in range(1, N):
            x_pred = x_state + v_state
            v_pred = v_state
            residual = y_cleaned[i] - x_pred
            residual = np.clip(residual, -residual_clip, residual_clip)

            x_state = x_pred + alpha * residual
            v_state = v_pred + beta * residual
            v_state = np.clip(v_state, -velocity_limit, velocity_limit)

            filtered[i] = x_state

        val_rmse = np.sqrt(np.mean((filtered[split_index:] - y_true_trend[split_index:]) ** 2))
        search_results.append({
            "filter_type": "alpha-beta",
            "complexity": 1,
            "alpha": alpha,
            "beta": beta,
            "gamma": 0.0,
            "val_rmse": val_rmse
        })

for alpha in alpha_values:
    for beta in beta_values:
        for gamma in gamma_values:
            x_state = y_cleaned[0]
            v_state = 0.0
            a_state = 0.0
            filtered = np.zeros(N, dtype=float)
            filtered[0] = x_state

            for i in range(1, N):
                x_pred = x_state + v_state + 0.5 * a_state
                v_pred = v_state + a_state
                a_pred = a_state
                residual = y_cleaned[i] - x_pred
                residual = np.clip(residual, -residual_clip, residual_clip)

                x_state = x_pred + alpha * residual
                v_state = v_pred + beta * residual
                a_state = a_pred + 2.0 * gamma * residual

                v_state = np.clip(v_state, -velocity_limit, velocity_limit)
                a_state = np.clip(a_state, -acceleration_limit, acceleration_limit)

                filtered[i] = x_state

            val_rmse = np.sqrt(np.mean((filtered[split_index:] - y_true_trend[split_index:]) ** 2))
            search_results.append({
                "filter_type": "alpha-beta-gamma",
                "complexity": 2,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "val_rmse": val_rmse
            })

min_val_rmse = min(item["val_rmse"] for item in search_results)
candidate_models = [item for item in search_results if item["val_rmse"] <= min_val_rmse * (1 + MODEL_TOLERANCE)]
best_model = sorted(candidate_models, key=lambda item: (item["complexity"], item["val_rmse"]))[0]

selected_type = best_model["filter_type"]
selected_alpha = best_model["alpha"]
selected_beta = best_model["beta"]
selected_gamma = best_model["gamma"]

filtered_final = np.zeros(N, dtype=float)
filtered_raw = np.zeros(N, dtype=float)

if selected_type == "alpha-beta":
    x_state = y_cleaned[0]
    v_state = 0.0
    filtered_final[0] = x_state

    for i in range(1, N):
        x_pred = x_state + v_state
        v_pred = v_state
        residual = y_cleaned[i] - x_pred
        residual = np.clip(residual, -residual_clip, residual_clip)

        x_state = x_pred + selected_alpha * residual
        v_state = v_pred + selected_beta * residual
        v_state = np.clip(v_state, -velocity_limit, velocity_limit)

        filtered_final[i] = x_state

    x_state = y_with_anomalies[0]
    v_state = 0.0
    filtered_raw[0] = x_state

    for i in range(1, N):
        x_pred = x_state + v_state
        v_pred = v_state
        residual = y_with_anomalies[i] - x_pred
        residual = np.clip(residual, -residual_clip, residual_clip)

        x_state = x_pred + selected_alpha * residual
        v_state = v_pred + selected_beta * residual
        v_state = np.clip(v_state, -velocity_limit, velocity_limit)

        filtered_raw[i] = x_state
else:
    x_state = y_cleaned[0]
    v_state = 0.0
    a_state = 0.0
    filtered_final[0] = x_state

    for i in range(1, N):
        x_pred = x_state + v_state + 0.5 * a_state
        v_pred = v_state + a_state
        a_pred = a_state
        residual = y_cleaned[i] - x_pred
        residual = np.clip(residual, -residual_clip, residual_clip)

        x_state = x_pred + selected_alpha * residual
        v_state = v_pred + selected_beta * residual
        a_state = a_pred + 2.0 * selected_gamma * residual

        v_state = np.clip(v_state, -velocity_limit, velocity_limit)
        a_state = np.clip(a_state, -acceleration_limit, acceleration_limit)

        filtered_final[i] = x_state

    x_state = y_with_anomalies[0]
    v_state = 0.0
    a_state = 0.0
    filtered_raw[0] = x_state

    for i in range(1, N):
        x_pred = x_state + v_state + 0.5 * a_state
        v_pred = v_state + a_state
        a_pred = a_state
        residual = y_with_anomalies[i] - x_pred
        residual = np.clip(residual, -residual_clip, residual_clip)

        x_state = x_pred + selected_alpha * residual
        v_state = v_pred + selected_beta * residual
        a_state = a_pred + 2.0 * selected_gamma * residual

        v_state = np.clip(v_state, -velocity_limit, velocity_limit)
        a_state = np.clip(a_state, -acceleration_limit, acceleration_limit)

        filtered_raw[i] = x_state

raw_rmse = np.sqrt(np.mean((y_with_anomalies - y_true_trend) ** 2))
cleaned_rmse = np.sqrt(np.mean((y_cleaned - y_true_trend) ** 2))
filtered_rmse = np.sqrt(np.mean((filtered_final - y_true_trend) ** 2))
filtered_raw_rmse = np.sqrt(np.mean((filtered_raw - y_true_trend) ** 2))

raw_mae = np.mean(np.abs(y_with_anomalies - y_true_trend))
cleaned_mae = np.mean(np.abs(y_cleaned - y_true_trend))
filtered_mae = np.mean(np.abs(filtered_final - y_true_trend))

true_mask = np.zeros(N, dtype=bool)
true_mask[anomaly_indices] = True
tp = np.sum(true_mask & detected_mask)
fp = np.sum(~true_mask & detected_mask)
fn = np.sum(true_mask & ~detected_mask)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

print("Selected filter:", selected_type)
print("alpha:", f"{selected_alpha:.3f}", "beta:", f"{selected_beta:.3f}", "gamma:", f"{selected_gamma:.3f}")
print("Validation RMSE:", f"{min_val_rmse:.6f}")
print("Raw RMSE:", f"{raw_rmse:.6f}")
print("Cleaned RMSE:", f"{cleaned_rmse:.6f}")
print("Filtered RMSE:", f"{filtered_rmse:.6f}")
print("Filtered RMSE on raw anomalous data:", f"{filtered_raw_rmse:.6f}")
print("Raw MAE:", f"{raw_mae:.6f}")
print("Cleaned MAE:", f"{cleaned_mae:.6f}")
print("Filtered MAE:", f"{filtered_mae:.6f}")
print("Anomaly detection precision:", f"{precision:.4f}")
print("Anomaly detection recall:", f"{recall:.4f}")

if filtered_rmse < cleaned_rmse:
    print("Verification: filtering improved the cleaned signal.")
else:
    print("Verification: filtering did not improve the cleaned signal.")

if filtered_raw_rmse < raw_rmse:
    print("Divergence protection is effective on raw anomalous data.")
else:
    print("Divergence protection should be tuned further.")

plt.figure(figsize=(12, 6))
plt.plot(t, y_true_trend, label="True trend", linewidth=2)
plt.scatter(t, y_with_anomalies, label="Data with anomalies", s=30)
plt.scatter(t[detected_mask], y_with_anomalies[detected_mask], label="Detected anomalies", s=80, marker='x')
plt.title("Input Data With Anomalies")
plt.xlabel("t")
plt.ylabel("y")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(12, 6))
plt.plot(t, y_true_trend, label="True trend", linewidth=2)
plt.plot(t, y_cleaned, label="Cleaned data", linewidth=2)
plt.plot(t, filtered_final, label=f"Filtered cleaned data ({selected_type})", linewidth=2)
plt.title("Cleaning and Recursive Filtering")
plt.xlabel("t")
plt.ylabel("y")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(12, 6))
plt.plot(t, y_true_trend, label="True trend", linewidth=2)
plt.plot(t, filtered_raw, label=f"Filtered raw anomalous data ({selected_type})", linewidth=2)
plt.plot(t, filtered_final, label="Filtered cleaned data", linewidth=2)
plt.title("Verification of Filter Stability")
plt.xlabel("t")
plt.ylabel("y")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()
