import pandas as pd


def main():
    df = pd.read_csv("./data.csv")
    # Task 1
    # top_10_providers = df["provider"].value_counts().head(10)
    # print(top_10_providers)

    # top_10_providers.plot(kind="bar")
    # plt.title("Top 10 Providers by Model Count")
    # plt.xlabel("Provider")
    # plt.ylabel("Number of Models")
    # plt.tight_layout(pad=2.0)
    # plt.savefig("./top_10_providers.png")

    # Task 2
    # tier_order = ["Free", "Budget", "Mid", "Premium", "Ultra"]
    # df_filtered = df[df['pricing_tier'].isin(tier_order)]

    # sns.set_theme(style='whitegrid')
    # plt.figure(figsize=(10, 6))
    # sns.boxplot(
    #     data=df_filtered,
    #     x='pricing_tier',
    #     y='aa_intelligence_index',
    #     order=tier_order,
    #     palette='viridis'
    # )
    # plt.title("Distribution of Intelligence Index by Pricing Tier",fontdict={"size": 16})
    # plt.xlabel("Pricing Tier", fontdict={"size": 16})
    # plt.ylabel("AA Intelligence Index (0-100)", fontdict={"size": 16})

    # sns.swarmplot(
    #     data=df_filtered,
    #     x='pricing_tier',
    #     y='aa_intelligence_index',
    #     order=tier_order,
    #     color='.25',
    #     size=3,
    # )
    # plt.tight_layout()
    # plt.savefig("./intelligence_by_pricing_tier.png")
    
    # Task 3
    benchmarks = {
        "GPQA Diamond": "gpqa_diamond",
        "Humaniy's Last Exam": "humanitys_last_exam",
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



if __name__ == "__main__":
    main()