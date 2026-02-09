"""
Datasets:
1) Video game sales data (VGChartz)
   https://gist.githubusercontent.com/designernatan/27da044c6dc823f7ac7fe3a01f4513ed/raw/vgsales.csv

2) Video game review scores
   https://raw.githubusercontent.com/Bakikhan/Video-Game-Sales-Dataset/main/Video_Games.csv
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt


# load CSV safely
def safe_read_csv(url):
    try:
        return pd.read_csv(url)
    except:
        print("Error loading data")
        sys.exit(1)


# clean column names
def normalize_columns(df):
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df


# find matching column name
def find_column(df, options):
    for col in options:
        if col in df.columns:
            return col
    return ""


def main():
    # load datasets
    vgsales_url = "https://gist.githubusercontent.com/designernatan/27da044c6dc823f7ac7fe3a01f4513ed/raw/vgsales.csv"
    scores_url = "https://raw.githubusercontent.com/Bakikhan/Video-Game-Sales-Dataset/main/Video_Games.csv"

    vgsales = safe_read_csv(vgsales_url)
    scores = safe_read_csv(scores_url)

    # clean column names
    vgsales = normalize_columns(vgsales)
    scores = normalize_columns(scores)

    # find needed columns
    v_name = find_column(vgsales, ["name", "game", "title"])
    s_name = find_column(scores, ["name", "game", "title"])

    v_platform = find_column(vgsales, ["platform"])
    s_platform = find_column(scores, ["platform"])

    v_genre = find_column(vgsales, ["genre"])
    s_genre = find_column(scores, ["genre"])

    v_sales = find_column(vgsales, ["global_sales"])
    s_critic = find_column(scores, ["critic_score"])
    s_user = find_column(scores, ["user_score"])

    # check columns exist
    if "" in [v_name, v_platform, v_genre, v_sales]:
        print("Missing columns in sales dataset")
        sys.exit(1)

    if "" in [s_name, s_platform, s_genre, s_critic, s_user]:
        print("Missing columns in scores dataset")
        sys.exit(1)

    # rename columns
    vgsales = vgsales.rename(columns={
        v_name: "game",
        v_platform: "platform",
        v_genre: "genre",
        v_sales: "global_sales"
    })

    scores = scores.rename(columns={
        s_name: "game",
        s_platform: "platform",
        s_genre: "genre",
        s_critic: "critic_score",
        s_user: "user_score"
    })

    # keep needed columns
    vgsales = vgsales[["game", "platform", "genre", "global_sales"]]
    scores = scores[["game", "platform", "genre", "critic_score", "user_score"]]

    # clean text
    for col in ["game", "platform", "genre"]:
        vgsales[col] = vgsales[col].astype(str).str.strip()
        scores[col] = scores[col].astype(str).str.strip()

    # convert numbers
    vgsales["global_sales"] = pd.to_numeric(vgsales["global_sales"], errors="coerce")
    scores["critic_score"] = pd.to_numeric(scores["critic_score"], errors="coerce")
    scores["user_score"] = pd.to_numeric(scores["user_score"], errors="coerce")

    # remove duplicates
    vgsales = vgsales.drop_duplicates()
    scores = scores.drop_duplicates()

    # remove missing rows
    vgsales = vgsales.dropna(subset=["game", "platform", "genre", "global_sales"])
    scores = scores.dropna(subset=["game", "platform", "genre", "critic_score", "user_score"])

    # join datasets
    v_indexed = vgsales.set_index(["game", "platform"])
    s_indexed = scores.set_index(["game", "platform"])

    joined = v_indexed.join(
        s_indexed,
        how="inner",
        lsuffix="_sales",
        rsuffix="_scores"
    )

    # fix duplicate genre column
    if "genre_sales" in joined.columns:
        joined = joined.rename(columns={"genre_sales": "genre"})
    if "genre_scores" in joined.columns:
        joined = joined.drop(columns=["genre_scores"])


    # group and aggregate
    genre_summary = (
        joined
        .reset_index()
        .groupby("genre")
        .agg(
            avg_global_sales=("global_sales", "mean"),
            avg_critic_score=("critic_score", "mean"),
            avg_user_score=("user_score", "mean"),
            num_games=("global_sales", "size")
        )
        .reset_index()
    )

    # create new column
    genre_summary["sales_per_critic_point"] = (
    genre_summary["avg_global_sales"] /
    genre_summary["avg_critic_score"].replace(0, pd.NA)
    )


    genre_summary = genre_summary.sort_values(
        by="avg_global_sales",
        ascending=False
    )

    # print results
    print("\nModified dataset:\n")
    print(genre_summary)

    # create bar chart
    plt.figure()
    plt.bar(genre_summary["genre"], genre_summary["avg_global_sales"])
    plt.title("Average Global Sales by Video Game Genre")
    plt.xlabel("Genre")
    plt.ylabel("Average Global Sales (millions)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig("avg_global_sales_by_genre.png")
    plt.close()

    print("\nSaved visualization as: avg_global_sales_by_genre.png")

    # print findings
    print("\nFindings based on the visualization:")
    print(
        "The bar chart shows the average global sales for each video game genre "
        "after combining sales and review data."
    )

    top_genre = genre_summary.iloc[0]["genre"]
    top_sales = genre_summary.iloc[0]["avg_global_sales"]

    lowest_genre = genre_summary.iloc[-1]["genre"]
    lowest_sales = genre_summary.iloc[-1]["avg_global_sales"]

    print(f"{top_genre} has the highest average sales at {top_sales:.2f} million units.")
    print(f"{lowest_genre} has the lowest average sales at {lowest_sales:.2f} million units.")
    print("This shows that some genres are more popular and sell better than others.")


if __name__ == "__main__":
    main()
