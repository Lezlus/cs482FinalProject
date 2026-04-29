import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score


def main():
    df = pd.read_csv("./data.csv")
    tier_order = ["Free", "Budget", "Mid", "Premium", "Ultra"]

    # Task 1
    top_10_providers = df["provider"].value_counts().head(10)
    print("Task 1: Top 10 Providers by Model Count")
    print(top_10_providers)
    print("-" * 50)

    top_10_providers.plot(kind="bar")
    plt.title("Top 10 Providers by Model Count")
    plt.xlabel("Provider")
    plt.ylabel("Number of Models")
    plt.tight_layout(pad=2.0)
    plt.savefig("./top_10_providers.png")
    plt.close()

    # Task 2
    df_filtered = df[df["pricing_tier"].isin(tier_order)]

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df_filtered,
        x="pricing_tier",
        y="aa_intelligence_index",
        order=tier_order,
        palette="viridis"
    )
    plt.title("Distribution of Intelligence Index by Pricing Tier",
              fontdict={"size": 16})
    plt.xlabel("Pricing Tier", fontdict={"size": 16})
    plt.ylabel("AA Intelligence Index (0-100)", fontdict={"size": 16})

    sns.swarmplot(
        data=df_filtered,
        x="pricing_tier",
        y="aa_intelligence_index",
        order=tier_order,
        color=".25",
        size=3,
    )
    plt.tight_layout()
    plt.savefig("./intelligence_by_pricing_tier.png")
    plt.close()

    # Task 3
    benchmarks = {
        "GPQA Diamond": "gpqa_diamond",
        "Humanity's Last Exam": "humanitys_last_exam",
        "LiveCode Bench": "livecodebench",
        "SciCode": "scicode",
        "MATH-500": "math_500"
    }

    for display_name, col_name in benchmarks.items():
        if col_name in df.columns:
            top_5 = df.nlargest(5, col_name)[["model_name", col_name]]
            print(f"Top 5 For {display_name}")

            for i, (index, row) in enumerate(top_5.iterrows(), 1):
                score = row[col_name]
                print(f"{i}. {row['model_name']}: {score:.4f}")
            print("-" * 30)
        else:
            print(f"Col {col_name} not found in DF")

    # Task 4: API cost modeling and forecasting.
    # Goal: predict blended API cost using intelligence scores and pricing tier.
    # The target variable is blended_cost_usd_per_1m.
    cost_df = df[
        [
            "model_name",
            "provider",
            "pricing_tier",
            "aa_intelligence_index",
            "composite_benchmark",
            "blended_cost_usd_per_1m",
        ]
    ].copy()

    cost_df = cost_df.dropna(subset=["blended_cost_usd_per_1m"])
    cost_df = cost_df[cost_df["pricing_tier"].isin(tier_order)].copy()

    cost_df["aa_intelligence_index"] = cost_df["aa_intelligence_index"].fillna(
        cost_df["aa_intelligence_index"].median()
    )
    cost_df["composite_benchmark"] = cost_df["composite_benchmark"].fillna(
        cost_df["composite_benchmark"].median()
    )

    X = cost_df[["aa_intelligence_index",
                 "composite_benchmark", "pricing_tier"]]
    y = cost_df["blended_cost_usd_per_1m"]

    # Convert pricing_tier text categories into numeric dummy columns.
    X = pd.get_dummies(X, columns=["pricing_tier"], drop_first=False)

    # 70% training data, 30% testing data.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
    )

    # Use log1p so very expensive Ultra models do not dominate the model too much.
    model = Ridge(alpha=1.0)
    model.fit(X_train, np.log1p(y_train))

    # Convert predictions back from log scale to normal dollar scale.
    y_pred = np.expm1(model.predict(X_test))

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Task 4: API Cost Modeling")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    print("-" * 50)

    # Visualize the fitted model: actual cost vs predicted cost.
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)

    min_value = min(y_test.min(), y_pred.min())
    max_value = max(y_test.max(), y_pred.max())
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")

    plt.title("Fitted Cost Model: Actual vs Predicted Blended Cost")
    plt.xlabel("Actual Blended Cost USD per 1M Tokens")
    plt.ylabel("Predicted Blended Cost USD per 1M Tokens")
    plt.tight_layout()
    plt.savefig("./cost_model_actual_vs_predicted.png", dpi=150)
    plt.close()

    # Cost projections for the top-3 LLMs by intelligence index.
    top_3_llms = cost_df.dropna(subset=["aa_intelligence_index"]).nlargest(
        3,
        "aa_intelligence_index",
    )[
        [
            "model_name",
            "provider",
            "aa_intelligence_index",
            "pricing_tier",
            "blended_cost_usd_per_1m",
        ]
    ]

    token_volumes_millions = [1, 10, 50, 100]
    projection_rows = []

    for index, row in top_3_llms.iterrows():
        for volume in token_volumes_millions:
            projection_rows.append({
                "model_name": row["model_name"],
                "provider": row["provider"],
                "pricing_tier": row["pricing_tier"],
                "aa_intelligence_index": row["aa_intelligence_index"],
                "token_volume_millions": volume,
                "blended_cost_usd_per_1m": row["blended_cost_usd_per_1m"],
                "projected_cost_usd": row["blended_cost_usd_per_1m"] * volume,
            })

    projection_df = pd.DataFrame(projection_rows)
    projection_df.to_csv("./top3_cost_projection.csv", index=False)

    print("Task 4: Top 3 LLM Cost Projections")
    print(projection_df.to_string(index=False))
    print("-" * 50)

    plt.figure(figsize=(10, 6))
    for model_name, group in projection_df.groupby("model_name"):
        short_name = model_name[:35] + \
            "..." if len(model_name) > 35 else model_name
        plt.plot(
            group["token_volume_millions"],
            group["projected_cost_usd"],
            marker="o",
            label=short_name,
        )

    plt.title("Projected API Cost for Top 3 LLMs")
    plt.xlabel("Monthly Token Volume in Millions")
    plt.ylabel("Projected Monthly Cost in USD")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./top3_cost_projection.png", dpi=150)
    plt.close()

    # Task 5: Temporal capability tracking.
    # Use release_year to chart intelligence progression and price compression from 2023 to 2026.
    temporal_df = df[
        ["release_year", "aa_intelligence_index", "blended_cost_usd_per_1m"]
    ].copy()

    temporal_df = temporal_df.dropna(subset=["release_year"])
    temporal_df = temporal_df[
        (temporal_df["release_year"] >= 2023) &
        (temporal_df["release_year"] <= 2026)
    ].copy()

    temporal_df["release_year"] = temporal_df["release_year"].astype(int)

    temporal_summary = temporal_df.groupby("release_year").agg(
        mean_intelligence=("aa_intelligence_index", "mean"),
        median_intelligence=("aa_intelligence_index", "median"),
        median_blended_cost=("blended_cost_usd_per_1m", "median"),
        mean_blended_cost=("blended_cost_usd_per_1m", "mean"),
        model_count=("aa_intelligence_index", "count"),
    ).reset_index()

    print("Task 5: Temporal Summary")
    print(temporal_summary.to_string(index=False))
    print("-" * 50)

    temporal_summary.to_csv("./temporal_summary.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(
        temporal_summary["release_year"],
        temporal_summary["mean_intelligence"],
        marker="o",
        label="Mean Intelligence Index",
    )
    ax1.set_xlabel("Release Year")
    ax1.set_ylabel("Mean AA Intelligence Index")
    ax1.set_xticks(temporal_summary["release_year"])

    ax2 = ax1.twinx()
    ax2.plot(
        temporal_summary["release_year"],
        temporal_summary["median_blended_cost"],
        marker="s",
        linestyle="--",
        label="Median Blended Cost",
    )
    ax2.set_ylabel("Median Blended Cost USD per 1M Tokens")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    plt.title("Intelligence Progression and Price Compression by Release Year")
    plt.tight_layout()
    plt.savefig("./temporal_intelligence_price.png", dpi=150)
    plt.close()

    print("Done. All charts and CSV files were saved in this folder.")


if __name__ == "__main__":
    main()
