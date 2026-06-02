# API Reference

Auto-generated reference for the public API of `eci`. See the
[Algorithms](algorithms.md) page for the mathematical specification and the
[Tutorials](tutorials/tutorial_1_decision_making.ipynb) for end-to-end examples.

## Environment

::: eci.environment.EnvConfig

::: eci.environment.Environment
    options:
      members:
        - __init__
        - run_one_simulation
        - run_n_simulation

## Agents

::: eci.agents.Agent
::: eci.agents.Voter
::: eci.agents.Candidate

## Voting Systems

### Plurality

::: eci.voting.plurality

### Quadratic

::: eci.voting.quadratic

### Decision helpers

KL gap helpers, candidate-utility computation and `response_function`
variants used by both voting rules, living in `eci.decision`.
(`_get_pref_belief_gap` was renamed `_get_belief_preference_gap`.)

::: eci.decision.utilities

::: eci.decision.response

### Scoring

Pluggable scoring strategies that turn KL gaps into candidate utilities.
See [Choosing a scoring function](algorithms.md#choosing-a-scoring-function)
for the math and a side-by-side comparison.

::: eci.decision.scoring

## Metrics

::: eci.metrics

## Utilities

Observation generation, KL divergence and trajectory accessors.

::: eci.utils
    options:
      members:
        - kl_divergence
        - generate_observations
        - get_voter_trajectory_data

## Plots

::: eci.plots
