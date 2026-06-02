from eci.plots._context import STYLE, _get_context
from eci.plots.belief import (
    animate_belief_trajectory,
    plot_belief_trajectory,
    plot_belief_vote_evolution,
)
from eci.plots.preference import plot_preference, plot_vote_shares
from eci.plots.voting import (
    compute_vote_shares,
    plot_voting_metrics,
    plot_voting_system_comparison,
)
from eci.plots.winners import (
    _bootstrap_proportion_ci,
    plot_winner_distribution,
    plot_winner_distribution_grouped,
    plurality_results_to_share_df,
)

__all__ = [
    # context
    "STYLE",
    "_get_context",
    # preference
    "plot_preference",
    "plot_vote_shares",
    # belief
    "plot_belief_trajectory",
    "animate_belief_trajectory",
    "plot_belief_vote_evolution",
    # winners
    "plurality_results_to_share_df",
    "plot_winner_distribution",
    "plot_winner_distribution_grouped",
    "_bootstrap_proportion_ci",
    # voting
    "compute_vote_shares",
    "plot_voting_system_comparison",
    "plot_voting_metrics",
]
