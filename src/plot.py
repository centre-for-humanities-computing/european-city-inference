import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.rcParams["figure.constrained_layout.use"] = True


def plot_distributions(
    x_range=(-6.0, 6.0, 102),
    figsize=(3, 5),
    save_path=Path().cwd().parent / "figures" / "fig_1_distributions.svg",
):
    """
    Plot distributions of preferences and beliefs.

    Parameters
    ----------
    - x_range: tuple, the range and number of points for the x-axis.
    - figsize: tuple, the size of the figure.
    - save_path: Path, the path to save the figure.
    """
    x = np.linspace(*x_range)
    _, axs = plt.subplots(figsize=figsize, nrows=3, sharex=True, sharey=True)

    preferences_list = [(4.0, 0.5), (0.0, 2.0), (4.0, 0.75)]
    beliefs_list = [(3.0, 1.0), (0.5, 1.0), (-1.5, 1.0)]

    for i, pref, bel in zip(range(3), preferences_list, beliefs_list):
        # Preferences
        preferences = norm.pdf(x, pref[0], pref[1])
        axs[i].fill_between(x, preferences, alpha=0.4, color="#761e58", linewidth=0.0)
        axs[i].plot(x, preferences, color="k", linewidth=0.6)

        # Beliefs
        beliefs = norm.pdf(x, bel[0], bel[1])
        axs[i].fill_between(x, beliefs, alpha=0.4, color="#234c80", linewidth=0.3)
        axs[i].plot(x, beliefs, color="k", linewidth=0.6)

        axs[i].yaxis.set_visible(False)
        axs[i].spines["left"].set_visible(False)

    sns.despine(left=True)
    plt.savefig(save_path)


def kl_divergence_gaussian(mu1, sigma1, mu2, sigma2):
    """Compute KL(P || Q) where P ~ N(mu1, sigma1²), Q ~ N(mu2, sigma2²)."""
    ratio = sigma2 / sigma1
    return np.log(ratio) + (sigma1**2 + (mu1 - mu2) ** 2) / (2 * sigma2**2) - 0.5


def plot_kl_divergences(
    figsize=(1, 5),
    save_path=Path().cwd().parent / "figures" / "fig_1_kl_divergences.svg",
):
    """
    Plot KL divergences between preferences and beliefs.

    Parameters
    ----------
    - figsize: tuple, the size of the figure.
    - save_path: Path, the path to save the figure.
    """
    _, axs = plt.subplots(figsize=figsize, nrows=3, sharex=True, sharey=True)

    preferences_list = [(4.0, 0.5), (0.0, 2.0), (4.0, 0.75)]
    beliefs_list = [(3.0, 1.0), (0.5, 1.0), (-1.5, 1.0)]

    for i, pref, bel in zip(range(3), preferences_list, beliefs_list):
        axs[i].bar(
            0,
            kl_divergence_gaussian(
                mu1=pref[0], sigma1=pref[1], mu2=bel[0], sigma2=bel[1]
            ),
            color="#761e58",
            width=0.5,
            edgecolor="black",
        )
        axs[i].set_xticks([])
        axs[i].set_xticklabels([])
        axs[i].set(ylim=(0, 16), xlim=(-1, 1), ylabel="KL divergence")

    sns.despine()
    plt.savefig(save_path)


def plot_time_series(
    observations,
    figsize=(9, 3),
    save_path=Path().cwd().parent / "figures" / "fig_1_time_series.svg",
):
    """
    Plot time series data.

    Parameters
    ----------
    - observations: numpy.ndarray, array of observations to plot.
    - figsize: tuple, the size of the figure.
    - save_path: Path, the path to save the figure.
    """
    time_steps = np.arange(observations.shape[0])
    n_nodes = observations.shape[1]

    fig, axs = plt.subplots(n_nodes, 1, figsize=figsize, sharex=True, sharey=True)
    colors = ["#c78b8d", "#a86483", "#4d315f"]

    for i, color in zip(range(n_nodes), colors):
        axs[i].scatter(time_steps, observations[:, i], s=30, edgecolor="k", color=color)
        axs[i].plot(
            time_steps, observations[:, i], color="grey", linewidth=0.75, zorder=-1
        )
        axs[i].set(ylim=(0, 1))
        axs[i].minorticks_on()

    plt.tight_layout()
    sns.despine()
    plt.savefig(save_path)


def plot_proportions(
    proportions, title="Scenario 2", figsize=(4, 6), colormap="tab20", alpha=0.7
):
    """
    Plot the proportions of categories across simulations.

    Parameters
    ----------
    - proportions: list of lists, where each sublist contains the categories for a simulation.
    - title: str, the title of the plot.
    - figsize: tuple, the size of the figure.
    - colormap: str, the colormap to use for the plot.
    - alpha: float, the transparency of the plot.
    """
    # Prepare the data for the plot
    df = pd.DataFrame(
        {
            "Simulation": np.repeat(
                np.arange(1, len(proportions) + 1), [len(p) for p in proportions]
            ),
            "Category": np.concatenate([p for p in proportions]),
        }
    )

    # Calculate the proportions by category and simulation
    category_proportions = (
        df.groupby(["Simulation", "Category"]).size().unstack(fill_value=0)
    )
    category_proportions = category_proportions.div(
        category_proportions.sum(axis=1), axis=0
    )  # Normalization

    # Plot the proportions
    category_proportions.plot(
        kind="area",
        stacked=True,
        figsize=figsize,
        colormap=colormap,
        alpha=alpha,
        linewidth=0,
    )

    # Set the labels and title
    plt.xlabel("Simulation", fontsize=12)
    plt.ylabel("Proportion", fontsize=12)
    plt.title(title, fontsize=14)

    # Set the legend
    plt.legend(title="Categories", loc="center left", bbox_to_anchor=(1, 0.5))

    # Adjust the layout
    plt.tight_layout()
    plt.show()


def plot_trajectories(
    nodes_traje, preferences, pref_labels, line_styles=None, figsize=(18, 7)
):
    """
    Plot the trajectories of expected means for different preferences.

    Parameters
    ----------
    - nodes_traje: dict, a dictionary containing the trajectories for each preference.
    - preferences: list, a list of indices for the preferences to plot.
    - pref_labels: list, a list of labels for the preferences.
    - line_styles: list, a list of line styles to use for the plots.
    - figsize: tuple, the size of the figure.
    """
    # Set global font parameters
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]

    # Default line styles if not provided
    if line_styles is None:
        line_styles = ["-"] * 10  # All line styles are solid lines

    # Create a figure with 3 subplots side by side
    fig, axes = plt.subplots(
        1, len(preferences), figsize=figsize, sharey=True, facecolor="#f9f9f9"
    )

    # Generate a palette of pastel colors
    def generate_pastel_colors(n):
        pastel_colors = []
        for i in range(n):
            hue = i / n  # Distribute hues evenly
            saturation = 0.4  # Slightly reduced saturation for more pastel colors
            lightness = 0.85  # Higher lightness for very light colors
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            pastel_colors.append(rgb)
        return pastel_colors

    pastel_colors = generate_pastel_colors(10)

    for idx, pref in enumerate(preferences):
        ax = axes[idx]
        ax.set_facecolor("#f9f9f9")  # Slightly lighter background for the plot

        for n_agent in range(10):
            # Use the corresponding pastel color
            color = pastel_colors[n_agent]
            # Set transparency
            alpha = 0.6 + 0.3 * (n_agent / 10)  # Variants of transparency
            ax.plot(
                nodes_traje[pref]["expected_mean"][n_agent],
                label=f"Agent {n_agent + 1}" if idx == 0 else "",
                color=color,
                linestyle=line_styles[n_agent % len(line_styles)],
                linewidth=1.5,  # Slightly thinner lines
                alpha=alpha,
            )

        ax.set_xlabel("Time Step", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Trajectory of Expected Mean ({pref_labels[idx]})",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(
            True, linestyle="--", alpha=0.4, color="#e0e0e0", linewidth=0.5
        )  # Very subtle grid

        # Add a subtle border around each subplot
        for spine in ax.spines.values():
            spine.set_edgecolor("#e0e0e0")  # Very light border color
            spine.set_linewidth(0.8)

    # Add a common legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(1.1, 1),
        title="Agents",
        facecolor="#f9f9f9",
        edgecolor="#e0e0e0",
    )

    # Add a global title
    fig.suptitle(
        "Trajectories of Expected Means for Different Preferences",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    # Adjust margins and spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjustment for the global title
    plt.show()
