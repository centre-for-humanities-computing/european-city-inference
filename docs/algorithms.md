# Core Algorithms

This page outlines the primary logic governing agent decision-making and the voting process. The system combines **Active Inference** principles (KL Divergence) with **Social Choice Theory**.

## 1. Agent Decision Logic (The "Vote" Generation)

Each agent $i$ evaluates candidates $c$ based on how well the candidate's policy minimizes the divergence from the agent's preferred state compared to the agent's current belief about the status quo.

### Inputs
* **Agent Preferences ($P_i$)**: Target distribution ($\mu_{pref}, \pi_{pref}$).
* **Agent Beliefs ($B_i$)**: Current belief about the world ($\mu_{belief}, \pi_{belief}$).
* **Candidate Policies ($Q_c$)**: Proposal distribution ($\mu_{cand}, \pi_{cand}$).

### Algorithm
1.  **Calculate Disatisfaction toward the world**:
    Measure how far the current belief is from the preference (Risk).
    $$R_{belief} = D_{KL}(B_i \parallel P_i)$$

2.  **Calculate Disatisfaction toward the candidate**:
    Measure how far the candidate's policy is from the preference.
    $$R_{cand, c} = D_{KL}(Q_c \parallel P_i)$$

3.  **Compute Utility (Logit)**:
    The utility is the reduction of dissatisfaction offered by the candidate (Relative reduction in Free Energy).
    $$U_{i,c} = R_{belief} - R_{cand, c}$$
    * If $U > 0$: Candidate is better than the status quo (reduces divergence).
    * If $U < 0$: Candidate is worse (increases divergence).

4.  **Generate Selection Probabilities (Softmax)**:
    Convert utilities into a probability distribution for the vote.
    $$P(\text{vote}_i = c) = \text{softmax}(U_{i,c}) = \frac{e^{U_{i,c}}}{\sum_j e^{U_{i,j}}}$$

5.  **Sample Vote**:
    Sample a discrete choice from the probability distribution.
    $$v_i \sim \text{Categorical}(P(\text{vote}_i))$$

---

## 2. Plurality Voting Process (Two-Round Runoff)

The system aggregates individual decisions into a collective outcome using a two-round system.

### Round 1
1.  **Collect Votes**:
    $$V_1 = \{v_i \mid i \in \text{Agents}\}$$
2.  **Tally**:
    $$\text{Count}_c = \sum_{v \in V_1} \mathbb{I}(v == c)$$
3.  **Filter**:
    Select top 2 candidates: $W_1, W_2 = \text{argmax}_2(\text{Count})$.

### Round 2 (Runoff)
1.  **Mask Preferences**:
    Set utility $U_{i,c} = -\infty$ for all $c \notin \{W_1, W_2\}$.
2.  **Re-Sample Votes**:
    Agents vote again, constrained to the top 2.
    $$v'_i \sim \text{Categorical}(\text{softmax}(U_{i} \text{ masked}))$$
3.  **Final Tally**:
    $$\text{Winner} = \text{argmax}(\text{Count}(V'))$$