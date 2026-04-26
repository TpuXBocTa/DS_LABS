import pandas as pd
import numpy as np

CRITERIA = [
    {"column": "conversion_rate_pct", "label": "Conversion rate, %", "kind": "max", "weight": 0.12},
    {"column": "avg_session_duration_sec", "label": "Average session duration, s", "kind": "max", "weight": 0.06},
    {"column": "pages_per_session", "label": "Pages per session", "kind": "max", "weight": 0.05},
    {"column": "user_satisfaction_10", "label": "User satisfaction, 1-10", "kind": "max", "weight": 0.10},
    {"column": "mobile_performance_100", "label": "Mobile performance, 0-100", "kind": "max", "weight": 0.09},
    {"column": "assortment_completeness_pct", "label": "Assortment completeness, %", "kind": "max", "weight": 0.06},
    {"column": "page_load_time_sec", "label": "Page load time, s", "kind": "min", "weight": 0.10},
    {"column": "bounce_rate_pct", "label": "Bounce rate, %", "kind": "min", "weight": 0.08},
    {"column": "cart_abandonment_pct", "label": "Cart abandonment, %", "kind": "min", "weight": 0.08},
    {"column": "error_rate_pct", "label": "Error rate, %", "kind": "min", "weight": 0.07},
    {"column": "support_first_response_min", "label": "Support first response time, min", "kind": "min", "weight": 0.05},
    {"column": "delivery_cost_uah", "label": "Delivery cost, UAH", "kind": "min", "weight": 0.04},
    {"column": "return_rate_pct", "label": "Return rate, %", "kind": "min", "weight": 0.03},
    {"column": "complaints_per_1000_orders", "label": "Complaints per 1000 orders", "kind": "min", "weight": 0.03},
    {"column": "checkout_steps", "label": "Checkout steps", "kind": "min", "weight": 0.02},
    {"column": "security_incidents_per_year", "label": "Security incidents per year", "kind": "min", "weight": 0.02},
]

COLUMNS = ["site_name"] + [criterion["column"] for criterion in CRITERIA]

ROWS = [
    ["ElectroHub", 2.8, 280, 5.4, 7.8, 84, 88, 3.5, 42, 61, 1.9, 18, 85, 6.2, 14, 6, 1],
    ["TechNova", 3.6, 340, 6.2, 8.5, 91, 92, 2.4, 34, 52, 1.2, 9, 70, 4.1, 8, 4, 0],
    ["SmartCart", 2.4, 250, 4.9, 7.4, 79, 80, 4.1, 49, 66, 2.8, 25, 95, 7.8, 20, 7, 2],
    ["PixelMarket", 3.1, 310, 5.8, 8.1, 88, 86, 2.9, 38, 55, 1.5, 12, 75, 4.8, 10, 5, 1],
    ["ElectroZone", 2.7, 270, 5.2, 7.6, 82, 84, 3.6, 44, 60, 2.1, 17, 82, 5.9, 15, 6, 1],
    ["MegaBuy", 2.9, 300, 5.5, 7.9, 86, 89, 3.1, 40, 58, 1.7, 14, 78, 5.1, 11, 5, 1],
    ["NovaShop", 3.3, 325, 6.0, 8.3, 90, 90, 2.7, 36, 54, 1.4, 11, 74, 4.5, 9, 4, 0],
    ["DeviceStore", 2.5, 260, 5.1, 7.5, 80, 83, 3.8, 46, 63, 2.3, 19, 88, 6.6, 17, 6, 2],
    ["PrimeTech", 3.8, 355, 6.4, 8.8, 94, 95, 2.2, 32, 50, 1.1, 8, 68, 3.9, 7, 4, 0],
    ["HomeDigital", 2.6, 275, 5.0, 7.7, 81, 85, 3.4, 43, 59, 2.0, 16, 80, 5.7, 13, 6, 1],
    ["CyberMall", 3.0, 295, 5.6, 8.0, 87, 87, 3.0, 39, 57, 1.6, 13, 77, 5.0, 12, 5, 1],
    ["ClickMarket", 2.3, 240, 4.7, 7.2, 76, 78, 4.4, 52, 68, 3.0, 28, 98, 8.1, 22, 7, 3],
]

OUTPUT_FILE = "erp_sites_results.xlsx"

df = pd.DataFrame(ROWS, columns=COLUMNS)

for column in COLUMNS[1:]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

normalized_df = pd.DataFrame(index=df.index)
for criterion in CRITERIA:
    column = criterion["column"]
    series = df[column]
    series_min = series.min()
    series_max = series.max()
    if series_max == series_min:
        normalized_df[column] = pd.Series([1.0] * len(series), index=series.index)
    elif criterion["kind"] == "max":
        normalized_df[column] = (series - series_min) / (series_max - series_min)
    else:
        normalized_df[column] = (series_max - series) / (series_max - series_min)

weighted_df = normalized_df.copy()
for criterion in CRITERIA:
    weighted_df[criterion["column"]] = weighted_df[criterion["column"]] * criterion["weight"]
weighted_sum_scores = weighted_df.sum(axis=1)

matrix = df[[criterion["column"] for criterion in CRITERIA]].astype(float).to_numpy()
denominator = np.sqrt((matrix ** 2).sum(axis=0))
denominator[denominator == 0] = 1.0
normalized_matrix = matrix / denominator
weights = np.array([criterion["weight"] for criterion in CRITERIA], dtype=float)
weighted_matrix = normalized_matrix * weights

ideal_best = []
ideal_worst = []
for index, criterion in enumerate(CRITERIA):
    column_values = weighted_matrix[:, index]
    if criterion["kind"] == "max":
        ideal_best.append(column_values.max())
        ideal_worst.append(column_values.min())
    else:
        ideal_best.append(column_values.min())
        ideal_worst.append(column_values.max())

ideal_best = np.array(ideal_best)
ideal_worst = np.array(ideal_worst)
distance_to_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
distance_to_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))
topsis_scores = distance_to_worst / (distance_to_best + distance_to_worst + 1e-12)

integrated_scores = 0.5 * weighted_sum_scores + 0.5 * topsis_scores

ranking_df = df[["site_name"]].copy()
ranking_df["weighted_sum_score"] = weighted_sum_scores.round(4)
ranking_df["topsis_score"] = pd.Series(topsis_scores, index=df.index).round(4)
ranking_df["integrated_score"] = pd.Series(integrated_scores, index=df.index).round(4)
ranking_df["rank"] = ranking_df["integrated_score"].rank(ascending=False, method="dense").astype(int)
ranking_df = ranking_df.sort_values(["rank", "integrated_score"], ascending=[True, False]).reset_index(drop=True)

eligibility_mask = (
    (df["conversion_rate_pct"] >= 2.8)
    & (df["page_load_time_sec"] <= 3.2)
    & (df["error_rate_pct"] <= 2.0)
    & (df["security_incidents_per_year"] <= 1)
    & (df["checkout_steps"] <= 5)
)

selected_site = None
selection_mode = "no-feasible-solution"

try:
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    decision_variables = [model.NewBoolVar(f"x_{index}") for index in range(len(df))]
    model.Add(sum(decision_variables) == 1)

    for index, is_eligible in enumerate(eligibility_mask.tolist()):
        if not is_eligible:
            model.Add(decision_variables[index] == 0)

    scaled_scores = (pd.Series(integrated_scores, index=df.index) * 10000).round().astype(int).tolist()
    model.Maximize(sum(decision_variables[index] * scaled_scores[index] for index in range(len(decision_variables))))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        selected_index = next(index for index in range(len(decision_variables)) if solver.Value(decision_variables[index]) == 1)
        selected_site = df.loc[selected_index, "site_name"]
        selection_mode = "cp-sat"
except Exception:
    pass

if selected_site is None:
    eligible_df = df.loc[eligibility_mask].copy()
    if not eligible_df.empty:
        best_index = pd.Series(integrated_scores, index=df.index).loc[eligible_df.index].idxmax()
        selected_site = df.loc[best_index, "site_name"]
        selection_mode = "fallback"

detailed_df = df.copy()
ranking_indexed = ranking_df.set_index("site_name")
detailed_df["weighted_sum_score"] = ranking_indexed.loc[detailed_df["site_name"], "weighted_sum_score"].values
detailed_df["topsis_score"] = ranking_indexed.loc[detailed_df["site_name"], "topsis_score"].values
detailed_df["integrated_score"] = ranking_indexed.loc[detailed_df["site_name"], "integrated_score"].values

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="input_data", index=False)
    normalized_df.to_excel(writer, sheet_name="normalized_data", index=False)
    ranking_df.to_excel(writer, sheet_name="ranking", index=False)
    detailed_df.to_excel(writer, sheet_name="detailed_scores", index=False)

print("\nFINAL RANKING:")
print(ranking_df.to_string(index=False))
print("\nBest alternative by integrated MCDA score:")
print(f"1st place: {ranking_df.iloc[0]['site_name']} (score = {ranking_df.iloc[0]['integrated_score']:.4f})")

