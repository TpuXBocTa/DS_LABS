from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


DATA_FILE = Path("sample_data.xlsx")
DESCRIPTION_FILE = Path("data_description.xlsx")
RESULTS_DIR = Path("results")

RESULTS_XLSX = RESULTS_DIR / "scoring_results.xlsx"
INDICATORS_XLSX = RESULTS_DIR / "selected_indicators.xlsx"
SUMMARY_XLSX = RESULTS_DIR / "cluster_summary.xlsx"
HISTOGRAM_PNG = RESULTS_DIR / "score_distribution.png"
SCATTER_PNG = RESULTS_DIR / "score_clusters.png"

RANDOM_STATE = 42


def read_source_files():
    data = pd.read_excel(DATA_FILE)
    description = pd.read_excel(DESCRIPTION_FILE)

    return data, description


def prepare_columns(data):
    data = data.copy()

    data = data.replace("NULL", np.nan)
    data = data.drop(columns=[column for column in data.columns if str(column).startswith("Unnamed")])

    rename_map = {
        "Application": "id",
        "Marital status": "marital_status_id",
    }
    data = data.rename(columns=rename_map)

    return data


def add_calculated_indicators(data):
    data = data.copy()

    date_columns = ["applied_at", "birth_date", "fact_addr_start_date", "employment_date"]
    for column in date_columns:
        if column in data.columns:
            data[column] = pd.to_datetime(data[column], errors="coerce")

    data["age_years"] = (data["applied_at"] - data["birth_date"]).dt.days / 365.25
    data["residence_years"] = (data["applied_at"] - data["fact_addr_start_date"]).dt.days / 365.25
    data["employment_years"] = (data["applied_at"] - data["employment_date"]).dt.days / 365.25

    data["loan_to_income"] = data["loan_amount"] / data["monthly_income"].replace(0, np.nan)

    return data


def clean_data(data, indicators):
    data = data.copy()

    for column in indicators:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data[indicators] = data[indicators].replace([np.inf, -np.inf], np.nan)

    for column in indicators:
        median_value = data[column].median()
        data[column] = data[column].fillna(median_value)

    data = data.drop_duplicates(subset=["id"])
    data = data.reset_index(drop=True)

    return data


def minmax_score(series, benefit=True):
    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        result = pd.Series(0.5, index=series.index)
    else:
        result = (series - min_value) / (max_value - min_value)

    if not benefit:
        result = 1 - result

    return result.clip(0, 1)


def age_score(age):
    score = 1 - (age - 40).abs() / 30
    return score.clip(0, 1)


def calculate_score(data):
    score_parts = pd.DataFrame(index=data.index)

    score_parts["loan_amount"] = minmax_score(data["loan_amount"], benefit=False)
    score_parts["loan_days"] = minmax_score(data["loan_days"], benefit=False)
    score_parts["age_years"] = age_score(data["age_years"])
    score_parts["children_count_id"] = minmax_score(data["children_count_id"], benefit=False)
    score_parts["education_id"] = minmax_score(data["education_id"], benefit=True)
    score_parts["fact_addr_owner_type_id"] = minmax_score(data["fact_addr_owner_type_id"], benefit=True)
    score_parts["residence_years"] = minmax_score(data["residence_years"], benefit=True)
    score_parts["has_immovables"] = minmax_score(data["has_immovables"], benefit=True)
    score_parts["has_movables"] = minmax_score(data["has_movables"], benefit=True)
    score_parts["employment_type_id"] = minmax_score(data["employment_type_id"], benefit=True)
    score_parts["organization_type_id"] = minmax_score(data["organization_type_id"], benefit=True)
    score_parts["empoyees_count_id"] = minmax_score(data["empoyees_count_id"], benefit=True)
    score_parts["employment_years"] = minmax_score(data["employment_years"], benefit=True)
    score_parts["seniority_years"] = minmax_score(data["seniority_years"], benefit=True)
    score_parts["monthly_income"] = minmax_score(data["monthly_income"], benefit=True)
    score_parts["monthly_expenses"] = minmax_score(data["monthly_expenses"], benefit=False)
    score_parts["other_loans_active"] = minmax_score(data["other_loans_active"], benefit=False)
    score_parts["other_loans_about_monthly"] = minmax_score(data["other_loans_about_monthly"], benefit=False)
    score_parts["loan_to_income"] = minmax_score(data["loan_to_income"], benefit=False)

    weights = {
        "loan_amount": 0.05,
        "loan_days": 0.04,
        "age_years": 0.05,
        "children_count_id": 0.04,
        "education_id": 0.04,
        "fact_addr_owner_type_id": 0.04,
        "residence_years": 0.05,
        "has_immovables": 0.04,
        "has_movables": 0.03,
        "employment_type_id": 0.04,
        "organization_type_id": 0.04,
        "empoyees_count_id": 0.04,
        "employment_years": 0.06,
        "seniority_years": 0.06,
        "monthly_income": 0.10,
        "monthly_expenses": 0.08,
        "other_loans_active": 0.07,
        "other_loans_about_monthly": 0.07,
        "loan_to_income": 0.12,
    }

    score = sum(score_parts[column] * weight for column, weight in weights.items())

    return score.round(4), score_parts


def cluster_applicants(data, indicators):
    cluster_features = data[indicators + ["Scor"]].copy()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_features)

    model = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10)
    data["cluster"] = model.fit_predict(scaled_features)

    cluster_score = data.groupby("cluster")["Scor"].mean()
    approved_cluster = cluster_score.idxmax()

    data["decision_binary"] = np.where(data["cluster"] == approved_cluster, 1, 0)
    data["decision"] = np.where(data["decision_binary"] == 1, "Надати кредит", "Відмова")

    return data


def make_outputs(data, indicators, description):
    RESULTS_DIR.mkdir(exist_ok=True)

    output_columns = [
        "id",
        *indicators,
        "Scor",
        "cluster",
        "decision_binary",
        "decision",
    ]

    result_table = data[output_columns].copy()
    result_table.to_excel(RESULTS_XLSX, index=False)

    available_descriptions = description.rename(columns={"Field_in_data": "indicator"})
    selected_description = available_descriptions[available_descriptions["indicator"].isin(indicators)].copy()

    calculated_rows = pd.DataFrame(
        {
            "indicator": ["age_years", "residence_years", "employment_years", "loan_to_income"],
            "Description_of_information": [
                "Вік позичальника на момент подання заявки",
                "Тривалість проживання за фактичною адресою",
                "Тривалість роботи на поточному місці",
                "Відношення суми кредиту до щомісячного доходу",
            ],
        }
    )
    selected_description = pd.concat([selected_description, calculated_rows], ignore_index=True)
    selected_description.to_excel(INDICATORS_XLSX, index=False)

    summary = data.groupby("decision").agg(
        applicants_count=("id", "count"),
        average_score=("Scor", "mean"),
        min_score=("Scor", "min"),
        max_score=("Scor", "max"),
        average_income=("monthly_income", "mean"),
        average_loan_amount=("loan_amount", "mean"),
    )
    summary = summary.round(3).reset_index()
    summary.to_excel(SUMMARY_XLSX, index=False)

    plt.figure(figsize=(9, 5))
    plt.hist(data["Scor"], bins=20)
    plt.title("Розподіл інтегрованої скорингової оцінки")
    plt.xlabel("Scor")
    plt.ylabel("Кількість заявників")
    plt.tight_layout()
    plt.savefig(HISTOGRAM_PNG, dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    for decision, group in data.groupby("decision"):
        plt.scatter(group["loan_to_income"], group["Scor"], label=decision, alpha=0.7)
    plt.title("Кластеризація заявників за скоринговою оцінкою")
    plt.xlabel("loan_to_income")
    plt.ylabel("Scor")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SCATTER_PNG, dpi=150)
    plt.close()

    return result_table, summary


def main():
    data, description = read_source_files()
    data = prepare_columns(data)
    data = add_calculated_indicators(data)

    indicators = [
        "loan_amount",
        "loan_days",
        "age_years",
        "children_count_id",
        "education_id",
        "fact_addr_owner_type_id",
        "residence_years",
        "has_immovables",
        "has_movables",
        "employment_type_id",
        "organization_type_id",
        "empoyees_count_id",
        "employment_years",
        "seniority_years",
        "monthly_income",
        "monthly_expenses",
        "other_loans_active",
        "other_loans_about_monthly",
        "loan_to_income",
    ]

    data = clean_data(data, indicators)
    data["Scor"], score_parts = calculate_score(data)
    data = cluster_applicants(data, indicators)

    result_table, summary = make_outputs(data, indicators, description)

    print("Selected indicators:", len(indicators))
    print(summary.to_string(index=False))
    print(f"Saved: {RESULTS_XLSX}")
    print(f"Saved: {INDICATORS_XLSX}")
    print(f"Saved: {SUMMARY_XLSX}")
    print(f"Saved: {HISTOGRAM_PNG}")
    print(f"Saved: {SCATTER_PNG}")


if __name__ == "__main__":
    main()
