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
    `_compute_candidate_utilities` in `eci.decision.utilities`.

4.  **Generate Selection Probabilities (Softmax)**:
    Convert into a probability distribution for the vote.
    $$P(\text{vote}_i = c) = \text{softmax}(U_{i,c}) = \frac{e^{U_{i,c}}}{\sum_j e^{U_{i,j}}}$$

5.  **Sample Vote**:
    Sample a discrete choice from the probability distribution.
    $$v_i \sim \text{Categorical}(P(\text{vote}_i))$$

---

## Choosing a scoring function

Step 3 above shows the **default** scoring formula
($U_{i,c} = (R_{belief} - R_{cand,c}) / R_{belief}$), but ECI ships four
interchangeable strategies in [`eci.decision.scoring`](api.md#scoring).
They all consume the same inputs ($R_{belief}$, $R_{cand,c}$) but trade
off differently between scale-invariance and intensity sensitivity. Pick
one by passing it to `_compute_candidate_utilities(..., scoring_fn=...)`.

| Strategy | Formula | When to use |
|---|---|---|
| `score_normalized` *(default)* | $(R_{b} - R_{c}) / R_{b}$ | Heterogeneous populations where baseline dissatisfaction varies widely. Scale-invariant: every agent's utility is in $(-\infty, 1]$. |
| `score_absolute` | $R_{b} - R_{c}$ | When you *want* high-dissatisfaction voters to dominate. Sensitive to baseline scale — agents with the largest $R_{b}$ get sharper softmaxes. |
| `score_inverted` | $R_{c} - R_{b}$ | Adversarial sanity check. A softmax over these elects the *worst* candidate; useful to confirm the simulation pipeline correctly amplifies score direction. |
| `score_product` | $-\,R_{b} \cdot R_{c}$ | Models *issue-salience*: dissatisfaction multiplies candidate distance, so a fit on a high-stakes dimension is rewarded multiplicatively rather than additively. A perfectly satisfied agent ($R_{b}=0$) is indifferent across candidates. |

### Visual intuition

For an agent with current belief gap $R_{b} = 1.0$ and a candidate at
distance $R_{c}$, the four scores look like:

| $R_{c}$ | normalized | absolute | inverted | product |
|---:|---:|---:|---:|---:|
| 0.0 | $+1.00$ | $+1.00$ | $-1.00$ | $\;\;0.00$ |
| 0.5 | $+0.50$ | $+0.50$ | $-0.50$ | $-0.50$ |
| 1.0 | $\;\;0.00$ | $\;\;0.00$ | $\;\;0.00$ | $-1.00$ |
| 2.0 | $-1.00$ | $-1.00$ | $+1.00$ | $-2.00$ |

Notice that `normalized` and `absolute` produce the **same numbers** when
$R_b = 1$ — the strategies diverge as agents become more or less
dissatisfied than baseline.

### Custom scorers

Any callable matching the [`ScoringFn`](api.md#scoring) protocol works:

```python
import jax.numpy as jnp
from eci.decision import _compute_candidate_utilities

def score_squared(belief_gap, pref_cand_gap):
    """Quadratic in candidate distance — amplifies large mismatches."""
    return -pref_cand_gap ** 2

u, *_ = _compute_candidate_utilities(data, scoring_fn=score_squared)
```

---