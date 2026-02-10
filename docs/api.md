# API Reference


# 🌍 Environment

::: eci.environment.Environment
    options:
      members:
        - run_n_simulation
        - create_data_frame
        - initialize_network

# 🤖 Agents

::: eci.agents.Voter
::: eci.agents.Candidate
::: eci.agents.Agent

# 🗳️ Voting Systems

::: eci.voting_system.plurality
::: eci.voting_system.quadratic
::: eci.voting_system.random_voting

# 🤖 Decision Making

::: eci.voting_system.decisions
::: eci.voting_system.beliefs
    options:
      members:
        - _get_pref_belief_gap
        - _get_pref_candidate_gap


# 🛠️ Utilities & Tools

## Metrics
::: eci.metrics