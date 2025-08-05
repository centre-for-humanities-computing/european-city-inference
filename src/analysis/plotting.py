import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_winner_distribution(df: pd.DataFrame):
    """Affiche la distribution des gagnants sur toutes les simulations."""
    plt.figure(figsize=(10, 6))
    sns.countplot(
        data=df,
        x="winner_index",
        order=df["winner_index"].value_counts().index,
        palette="viridis",
    )
    plt.title("Distribution des Gagnants sur Toutes les Simulations")
    plt.xlabel("Index du Candidat")
    plt.ylabel("Nombre de Victoires")
    plt.grid(axis="y", linestyle="--")
    plt.show()


def plot_round_1_proportions(df: pd.DataFrame):
    """Affiche les proportions moyennes obtenues au 1er tour."""
    r1_cols = sorted([col for col in df.columns if "r1_prop_cand" in col])
    r1_df = df[r1_cols].mean().reset_index()
    r1_df.columns = ["Candidate", "Average Proportion"]
    r1_df["Candidate"] = r1_df["Candidate"].str.extract("(\d+)").astype(int)

    plt.figure(figsize=(12, 7))
    sns.barplot(data=r1_df, x="Candidate", y="Average Proportion", palette="mako")
    plt.title("Proportions Moyennes au 1er Tour")
    plt.xlabel("Index du Candidat")
    plt.ylabel("Proportion Moyenne des Votes")
    plt.ylim(0, max(r1_df["Average Proportion"]) * 1.2)
    plt.grid(axis="y", linestyle="--")
    plt.show()
