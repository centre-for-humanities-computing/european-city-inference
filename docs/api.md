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

::: eci.voting_system.plurality

### Quadratic

::: eci.voting_system.quadratic

### Decision helpers

Score functions, KL gap helpers and `response_function` variants used by both
voting rules. The legacy `eci.voting_system.beliefs` module has been merged
into this one — `_get_pref_belief_gap` was renamed `_get_belief_preference_gap`.

::: eci.voting_system.decisions

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
