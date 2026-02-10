# Glossary

Terminology used in the **European City Inference (ECI)** project.

### Agent

An autonomous entity in the simulation. ECI distinguishes between two types:

* **Voters:** Agents who infer states and cast votes based on their preferences.
* **Candidates:** Agents who propose policy platforms to get elected.

### HGF (Hierarchical Gaussian Filter)

A mathematical model used to simulate how agents (voters) update their beliefs in response to new information (candidate policies) under uncertainty. It allows for tracking how "volatile" or unstable the environment feels to the agent.

* *Library used:* [PyHGF Documentation](https://github.com/ilabcode/pyhgf)
* *Reference:* [Mathys et al. (2011)](https://www.google.com/search?q=https://frontiersin.org/articles/10.3389/fnhum.2011.00028/full)

### Plurality Voting

A standard voting system where each voter is allowed to vote for only one candidate, and the candidate who polls the most among their counterparts (a plurality) is elected.

* *See also:* [Plurality Voting (Wikipedia)](https://en.wikipedia.org/wiki/Plurality_voting)

### Policy Dimensions

The specific thematic axes on which opinions are formed (e.g., Economy, Environment, Security). In the code, these are handled as independent continuous state dimensions (`num_preferences`).

### Precision

A measure of an agent's confidence in a belief. Mathematically, it is the inverse of variance ().

### Prediction Error

The difference between what an agent expected to observe (based on their internal model) and what they actually observed. This error signal drives the learning process in the HGF, forcing the agent to update their beliefs to minimize future errors.

### Quadratic Voting

A collective decision-making procedure where participants are given a budget of "voice credits". They can allocate multiple votes to a candidate, but the cost is quadratic: buying  votes costs  credits. This allows voters to express the **intensity** of their preferences, not just the direction.

* *See also:* [Quadratic Voting (Wikipedia)](https://en.wikipedia.org/wiki/Quadratic_voting)

### Regret

A metric measuring the difference between the maximum utility a voter *could* have received (if their ideal candidate had won) and the utility they *actually* received from the election winner.

* *See also:* [Regret (Decision Theory)](https://en.wikipedia.org/wiki/Regret_(decision_theory))

### Scenario

A specific configuration for generating input data (observations) that defines how the political environment evolves over time (e.g., a stable period, a sudden shock, or a slow drift in ideologies).

### Social Welfare

A global metric used to evaluate the quality of an election outcome. It is typically calculated as the sum of individual utilities of all voters. Maximizing social welfare is often considered the goal of an optimal voting system.

* *See also:* [Social Welfare Function](https://en.wikipedia.org/wiki/Social_welfare_function)

### Tonic Volatility

A parameter in the HGF model () representing the baseline level of environmental volatility as perceived by an agent.

* **High Tonic Volatility:** The agent believes the world is changing rapidly, leading to faster learning rates (they change their mind quickly).
* **Low Tonic Volatility:** The agent believes the world is stable, leading to slower learning rates (they are more dogmatic).

### Trajectory

The time series of an agent's internal states throughout the simulation. It visualizes how an agent's beliefs (means) and uncertainties (precisions) evolve step-by-step in response to candidate actions.