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

3.  **Compute Candidate preference**:
    The reduction of dissatisfaction offered by the candidate.
    $$U_{i,c} = R_{belief} - R_{cand, c}$$
    * If $U > 0$: Candidate is better than the status quo (reduces divergence).
    * If $U < 0$: Candidate is worse (increases divergence).

4.  **Generate Selection Probabilities (Softmax)**:
    Convert into a probability distribution for the vote.
    $$P(\text{vote}_i = c) = \text{softmax}(U_{i,c}) = \frac{e^{U_{i,c}}}{\sum_j e^{U_{i,j}}}$$

5.  **Sample Vote**:
    Sample a discrete choice from the probability distribution.
    $$v_i \sim \text{Categorical}(P(\text{vote}_i))$$

---