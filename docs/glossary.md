# Glossary

Terminology used in the **European City Inference (ECI)** project.

## Agent

Autonomous entities (e.g., consumers, bacteria, soldiers) with specific,, often heterogeneous,, attributes and behaviors. ECI distinguishes between two types:

* **Voters:** Agents who infer states of the world given observation and cast votes based on their preferences.
* **Candidates:** Agents who propose policy to get elected.

## Environment: 
The space or network where agents exist, interact, and move.

## Interaction Rules: 
Defined behaviors or algorithms (e.g., "if agent A touches agent B, infection spreads") that govern how agents interact with each other and their environment.

## Emergence: 

The core outcome of an ABM, where high-level, system-wide patterns arise from low-level, individual, interactions

## HGF (Hierarchical Gaussian Filter)

A mathematical model used to simulate how people update their beliefs in response to new information under uncertainty.

* *Library used:* [PyHGF Documentation](https://github.com/ilabcode/pyhgf)
* *Reference:* [Mathys et al. (2011)](https://www.google.com/search?q=https://frontiersin.org/articles/10.3389/fnhum.2011.00028/full)

## Tonic Volatility

A parameter in the HGF model () representing the baseline level of environmental volatility as perceived by an agent.

* **High Tonic Volatility:** The agent believes the world is changing rapidly, leading to faster learning rates (they change their mind quickly).
* **Low Tonic Volatility:** The agent believes the world is stable, leading to slower learning rates (they are more dogmatic).

## Precision

A measure of an agent's confidence in a belief. Mathematically, it is the inverse of variance ().

## Prediction Error

The difference between what an agent expected to observe (based on their internal model) and what they actually observed. This error signal drives the learning process in the HGF, the agent to update their beliefs to minimize future errors.

## Plurality Voting

A standard voting system where each voter is allowed to vote for only one candidate, and the candidate who polls the most among their counterparts (a plurality) is elected.

* *See also:* [Plurality Voting (Wikipedia)](https://en.wikipedia.org/wiki/Plurality_voting)

## Quadratic Voting

A collective decision-making procedure where participants are given a budget of "voice credits". They can allocate multiple votes to a candidate, but the cost is quadratic: buying  votes costs  credits. This allows voters to express the **intensity** of their preferences, not just the direction.

* *See also:* [Quadratic Voting (Wikipedia)](https://en.wikipedia.org/wiki/Quadratic_voting)

## Policy Dimensions

The specific thematic axes on which beliefs are formed (e.g., Economy, Environment, Security).

## Scenario

A specific configuration for generating input data (observations) that defines how the political environment evolves over time (e.g., a stable period, a sudden shock, or a slow drift of dimensions).

## Social Welfare

A global metric used to evaluate the quality of an election outcome. It is typically calculated as the sum of individual utilities of all voters. Maximizing social welfare is often considered the goal of an optimal voting system.

* *See also:* [Social Welfare Function](https://en.wikipedia.org/wiki/Social_welfare_function)