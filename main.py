import matplotlib.pyplot as plt
import pandas as pd


def main():
    df = pd.read_csv("./data.csv")
    top_10_providers = df["provider"].value_counts().head(10)
    print(top_10_providers)

    top_10_providers.plot(kind="bar")
    plt.title("Top 10 Providers by Model Count")
    plt.xlabel("Provider")
    plt.ylabel("Number of Models")
    plt.tight_layout(pad=2.0)
    plt.savefig("./top_10_providers.png")


if __name__ == "__main__":
    main()