# Core Algorithms

## Agent Decision Logic

Each agent $i$ evaluates candidates $c$ based on how well the candidate's policy minimizes the divergence from the agent's preferred state compared to the agent's current belief.

### Inputs
* **Agent Preferences ($P_i$)**: Target distribution ($\mu_{pref}, \pi_{pref}$).
* **Agent Beliefs ($B_i$)**: Current belief about the world ($\mu_{belief}, \pi_{belief}$).
* **Candidate Policies ($Q_c$)**: Proposal distribution ($\mu_{cand}, \pi_{cand}$).

### Algorithm
1.  **Calculate Disatisfaction toward the world**:
    Measure how far the current belief is from the preference.
    $$R_{belief} = D_{KL}(B_i \parallel P_i)$$

2.  **Calculate Disatisfaction toward the candidate**:
    Measure how far the candidate's policy is from the preference.
    $$R_{cand, c} = D_{KL}(Q_c \parallel P_i)$$

3.  **Compute Candidate utility (normalized)**:
    The *relative* reduction of dissatisfaction offered by the candidate,
    normalized by the agent's current dissatisfaction so that scores are
    comparable across agents with different baseline gaps.
    $$U_{i,c} = \frac{R_{belief} - R_{cand, c}}{R_{belief}} = 1 - \frac{R_{cand, c}}{R_{belief}}$$
    * If $U > 0$: Candidate is better than the status quo (closer to preference than current belief).
    * If $U = 0$: Candidate is exactly as good as the status quo.
    * If $U < 0$: Candidate is worse (further from preference than current belief).

    A small epsilon is added to $R_{belief}$ in the denominator to guard
    against division by zero when beliefs already equal preferences. See
    `_compute_candidate_utilities` in `eci.voting_system.decisions`.

4.  **Generate Selection Probabilities (Softmax)**:
    Convert into a probability distribution for the vote.
    $$P(\text{vote}_i = c) = \text{softmax}(U_{i,c}) = \frac{e^{U_{i,c}}}{\sum_j e^{U_{i,j}}}$$

5.  **Sample Vote**:
    Sample a discrete choice from the probability distribution.
    $$v_i \sim \text{Categorical}(P(\text{vote}_i))$$

---