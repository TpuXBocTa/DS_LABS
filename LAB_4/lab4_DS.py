import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILE = "ecommerce_sites.csv"
OUTPUT_EXCEL = "erp_ecommerce_olap_results.xlsx"

MAX_CRITERIA = [
    "conversion_rate",
    "avg_order_value",
    "revenue_per_visitor",
    "retention_rate",
    "availability",
    "seo_visibility"
]

MIN_CRITERIA = [
    "bounce_rate",
    "page_load_time",
    "cart_abandonment",
    "customer_complaints",
    "return_rate",
    "support_response_time",
    "checkout_steps",
    "cost_per_order",
    "error_rate",
    "ad_cost_per_conversion"
]

ALL_CRITERIA = MAX_CRITERIA + MIN_CRITERIA

WEIGHTS = {
    "conversion_rate": 0.10,
    "avg_order_value": 0.07,
    "revenue_per_visitor": 0.10,
    "retention_rate": 0.08,
    "availability": 0.07,
    "seo_visibility": 0.06,
    "bounce_rate": 0.06,
    "page_load_time": 0.07,
    "cart_abandonment": 0.07,
    "customer_complaints": 0.05,
    "return_rate": 0.05,
    "support_response_time": 0.04,
    "checkout_steps": 0.04,
    "cost_per_order": 0.05,
    "error_rate": 0.05,
    "ad_cost_per_conversion": 0.04
}

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Input file was not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

required_columns = ["site_id", "site_name", "product_type", "platform", "region"] + ALL_CRITERIA

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing column in input file: {column}")

if len(df) != 12:
    raise ValueError("Input file must contain exactly 12 e-commerce websites.")

if abs(sum(WEIGHTS.values()) - 1.0) > 0.000001:
    raise ValueError("The sum of criterion weights must be equal to 1.0.")

normalized = df.copy()

for criterion in MAX_CRITERIA:
    max_value = df[criterion].max()
    normalized[criterion + "_norm"] = df[criterion] / max_value

for criterion in MIN_CRITERIA:
    min_value = df[criterion].min()
    normalized[criterion + "_norm"] = min_value / df[criterion]

ranked = normalized.copy()
ranked["integrated_score"] = 0.0

for criterion in ALL_CRITERIA:
    ranked["integrated_score"] = ranked["integrated_score"] + ranked[criterion + "_norm"] * WEIGHTS[criterion]

ranked["integrated_percent"] = ranked["integrated_score"] * 100
ranked["integrated_rank"] = ranked["integrated_score"].rank(ascending=False, method="dense").astype(int)
ranked = ranked.sort_values("integrated_rank").reset_index(drop=True)

weighted_matrix = ranked[[criterion + "_norm" for criterion in ALL_CRITERIA]].values * np.array(
    [WEIGHTS[criterion] for criterion in ALL_CRITERIA]
)

ideal_best = weighted_matrix.max(axis=0)
ideal_worst = weighted_matrix.min(axis=0)

distance_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
distance_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))

topsis = ranked[["site_id", "site_name"]].copy()
topsis["topsis_score"] = distance_worst / (distance_best + distance_worst)
topsis["topsis_rank"] = topsis["topsis_score"].rank(ascending=False, method="dense").astype(int)
topsis = topsis.sort_values("topsis_rank").reset_index(drop=True)

ranked["efficiency_group"] = pd.cut(
    ranked["integrated_score"],
    bins=[0, 0.55, 0.70, 1.00],
    labels=["low", "medium", "high"]
)

olap_product = pd.pivot_table(
    ranked,
    values="integrated_score",
    index="product_type",
    aggfunc=["mean", "max", "min", "count"]
)

olap_platform = pd.pivot_table(
    ranked,
    values="integrated_score",
    index="platform",
    aggfunc=["mean", "max", "min", "count"]
)

olap_region = pd.pivot_table(
    ranked,
    values="integrated_score",
    index="region",
    aggfunc=["mean", "max", "min", "count"]
)

olap_product_platform = pd.pivot_table(
    ranked,
    values="integrated_score",
    index="product_type",
    columns="platform",
    aggfunc="mean"
)

olap_region_platform = pd.pivot_table(
    ranked,
    values="integrated_score",
    index="region",
    columns="platform",
    aggfunc="mean"
)

olap_efficiency_groups = pd.pivot_table(
    ranked,
    values="site_id",
    index="product_type",
    columns="efficiency_group",
    aggfunc="count",
    fill_value=0,
    observed=False
)

best_by_category = ranked.loc[
    ranked.groupby("product_type")["integrated_score"].idxmax(),
    ["product_type", "site_name", "platform", "region", "integrated_score", "integrated_rank"]
].sort_values("product_type")

np.random.seed(42)

base_weights = np.array([WEIGHTS[criterion] for criterion in ALL_CRITERIA])
norm_matrix = ranked[[criterion + "_norm" for criterion in ALL_CRITERIA]].values
sensitivity_scores = []

for i in range(500):
    random_factors = np.random.uniform(0.85, 1.15, size=len(base_weights))
    changed_weights = base_weights * random_factors
    changed_weights = changed_weights / changed_weights.sum()
    changed_score = norm_matrix @ changed_weights
    sensitivity_scores.append(changed_score)

sensitivity_scores = np.array(sensitivity_scores)

sensitivity = ranked[["site_id", "site_name", "integrated_rank"]].copy()
sensitivity["mean_score"] = sensitivity_scores.mean(axis=0)
sensitivity["std_score"] = sensitivity_scores.std(axis=0)
sensitivity["min_score"] = sensitivity_scores.min(axis=0)
sensitivity["max_score"] = sensitivity_scores.max(axis=0)
sensitivity["stability_percent"] = (1 - sensitivity["std_score"] / sensitivity["mean_score"]) * 100
sensitivity = sensitivity.sort_values("integrated_rank").reset_index(drop=True)

ranking_plot = ranked.sort_values("integrated_score", ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(ranking_plot["site_name"], ranking_plot["integrated_score"])
plt.xlabel("Integrated score")
plt.ylabel("E-commerce website")
plt.title("Ranking of e-commerce websites by integrated efficiency score")
plt.tight_layout()
plt.savefig("ranking_integrated_score.png", dpi=300)
plt.close()

platform_plot = ranked.groupby("platform")["integrated_score"].mean().sort_values()

plt.figure(figsize=(8, 5))
plt.bar(platform_plot.index, platform_plot.values)
plt.xlabel("Platform")
plt.ylabel("Average integrated score")
plt.title("Average integrated score by platform")
plt.tight_layout()
plt.savefig("olap_platform_integrated_score.png", dpi=300)
plt.close()

product_plot = ranked.groupby("product_type")["integrated_score"].mean().sort_values()

plt.figure(figsize=(8, 5))
plt.bar(product_plot.index, product_plot.values)
plt.xlabel("Product type")
plt.ylabel("Average integrated score")
plt.title("Average integrated score by product type")
plt.tight_layout()
plt.savefig("olap_product_type_integrated_score.png", dpi=300)
plt.close()

ranked.to_csv("final_website_ranking.csv", index=False, encoding="utf-8-sig")
topsis.to_csv("topsis_validation.csv", index=False, encoding="utf-8-sig")
sensitivity.to_csv("sensitivity_analysis.csv", index=False, encoding="utf-8-sig")

with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Input data", index=False)
    ranked.to_excel(writer, sheet_name="Integrated ranking", index=False)
    topsis.to_excel(writer, sheet_name="TOPSIS validation", index=False)
    sensitivity.to_excel(writer, sheet_name="Sensitivity", index=False)
    olap_product.to_excel(writer, sheet_name="OLAP product")
    olap_platform.to_excel(writer, sheet_name="OLAP platform")
    olap_region.to_excel(writer, sheet_name="OLAP region")
    olap_product_platform.to_excel(writer, sheet_name="Product platform")
    olap_region_platform.to_excel(writer, sheet_name="Region platform")
    olap_efficiency_groups.to_excel(writer, sheet_name="Efficiency groups")
    best_by_category.to_excel(writer, sheet_name="Best by category", index=False)

merged = ranked.merge(topsis, on=["site_id", "site_name"])
rank_correlation = merged["integrated_rank"].corr(merged["topsis_rank"], method="spearman")

optimal_site = ranked.iloc[0]

print()
print("ERP E-COMMERCE WEBSITE MULTI-CRITERIA OLAP ANALYSIS")
print("=" * 80)

print()
print("Input model:")
print(f"Number of websites: {len(ranked)}")
print(f"Number of criteria: {len(ALL_CRITERIA)}")
print(f"Maximized criteria: {len(MAX_CRITERIA)}")
print(f"Minimized criteria: {len(MIN_CRITERIA)}")
print(f"Sum of weights: {sum(WEIGHTS.values()):.4f}")

print()
print("Final integrated ranking:")
print(
    ranked[
        [
            "integrated_rank",
            "site_name",
            "product_type",
            "platform",
            "region",
            "integrated_score",
            "integrated_percent"
        ]
    ].to_string(index=False)
)

print()
print("Optimal website:")
print(f"Site name: {optimal_site['site_name']}")
print(f"Product type: {optimal_site['product_type']}")
print(f"Platform: {optimal_site['platform']}")
print(f"Region: {optimal_site['region']}")
print(f"Integrated score: {optimal_site['integrated_score']:.4f}")
print(f"Integrated percent: {optimal_site['integrated_percent']:.2f}%")

print()
print("TOP-3 websites:")
print(
    ranked.head(3)[
        [
            "integrated_rank",
            "site_name",
            "integrated_percent",
            "product_type",
            "platform"
        ]
    ].to_string(index=False)
)

print()
print("OLAP analysis by product type:")
print(olap_product)

print()
print("OLAP analysis by platform:")
print(olap_platform)

print()
print("OLAP analysis by region:")
print(olap_region)

print()
print("Best website by product type:")
print(best_by_category.to_string(index=False))

print()
print("Adequacy validation:")
print(f"Spearman correlation between integrated ranking and TOPSIS ranking: {rank_correlation:.4f}")
print(f"Average sensitivity stability: {sensitivity['stability_percent'].mean():.2f}%")

print()
print("Generated files:")
print("1. final_website_ranking.csv")
print("2. topsis_validation.csv")
print("3. sensitivity_analysis.csv")
print("4. erp_ecommerce_olap_results.xlsx")
print("5. ranking_integrated_score.png")
print("6. olap_platform_integrated_score.png")
print("7. olap_product_type_integrated_score.png")