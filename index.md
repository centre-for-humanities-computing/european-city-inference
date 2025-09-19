---
title: "Quadratic voting protects collective decision-making against predatory politics"
short_title: Executable science
abstract: |
    Social structures are shaping human existence with both fulfilling and repressive influences. While
    individuals inherit these structures, they also actively participate in reshaping them through their actions,
    collective decisions, or passivity. Democracy is a principled system for shaping social structure by
    aggregating and balancing individual preferences from an estimate of their proportions in the population.
    Presumably, this is the most efficient and peaceful system available. Yet in practice, it appears vulnerable.
    Informed citizens sometimes support systems that would contradict their own preferences, and electoral
    outcomes can diverge sharply from the population’s interests. Such misalignments open the door to
    predatory political strategies, where the common good is no longer maximised by quantity. Critics have
    pointed out the distortion of two-round elections with majority voting, and a collection of alternative
    systems, such as quadratic voting (QV), have been introduced to mitigate these problems. However, these
    models have been approached mostly from a theoretical standpoint, and evidence grounded on principled
    behavioural mechanisms at scale is still lacking. In this study, we simulate collective decision-making
    using agents endowed with Bayesian belief updating and predictive-coding neural architectures, providing
    biologically plausible approximations of human behaviours. Our model reproduces key patterns observed
    in democratic elections and identifies conditions under which citizens could vote against their interests.
    We then introduce a predatory candidate capable of strategically crafting political offers to dominate
    elections, both deceptively and transparently. Our results show that while the predatory agent can
    manipulate standard elections, it fails to do so under QV. Instead, the system favours candidates who
    genuinely reflect voters’ preferences. These findings offer strong computational support for the potential
    of quadratic voting to protect democratic integrity, and we provide tools to detect and analyse predatory
    political
keywords:
  - collective behaviours
  - decision-making
  - democracy
  - predictive coding
abbreviations:
  MyST: Markedly Structured Text
  HPC: High Performance Computing
bibliography: refs.bib
---


# Introduction

Democracy is a principled political organisation serving the
optimal expression of preferences and collective decision-
making. Core principles like freedom of speech and incen-
tives for citizen education and enlightment are the inner
drivers that promote the expression of individual prefer-
ences and desires. Only a society that build its principles
around the expression of individual interest will be able to
promote environment that drive individual happiness and
development.
Yet, this organisation is not widely adopted. As of early
2025, only XX countries out of XX were categorised as demo-
cratic societies. And it is also apparent that even democratic
societies are still vulnerable to fallacious expression and rep-
resentation of individual interests. For example, ...
The origin or such weakness has been extensively studied,
from a theoretical and mathematical perspective by contrast-
ing different voting system. And also from the psychological
approach to collective decision-making. Political behaviours
often appear as irrational and stochastic if we compare them
to optimal Bayesian solutions. Recent directions have high-
lighted the importance of reinforcement learning as unifying
framework to understand political cognition @Schulz2024, suggesting
that these behaviours can emerge from boundedly rational
agents operating under uncertainty, partial observability,
and social influence. The dynamic estimation of uncertainty
is central to human cognition and the driver of several psy-
chiatric conditions @Sandhu2023.
In this paper, we take a cognitive computational mod-
elling approach to collective decision-making @Tump2023.
The main contributions of our work are:

# Materials and Methods

```{admonition} Quadratic Voting Explained
:class: tip

Quadratic voting (QV) is a collective decision-making procedure where individuals express the strength of their preferences rather than just the direction. Each participant is given a budget of voice credits which they can spend across multiple issues. The cost of casting *n* votes on an issue is proportional to *n²*, which means the more strongly you feel, the more it "costs" you to express it.

This method addresses the "tyranny of the majority" by allowing minority preferences to surface if they are felt strongly enough.
```

## Simulation Design


**Overview of the computational framework.** We developed an environment where the population of agents, sharing connections in an implied graph network, are exposed to collective events and information (time series in the upper part) along a set of dimensions. Agents are equipped with Bayesian belief updating to track the states of the environment. At any time step, citizen dissatisfaction is quantified through the Kullback-Leibler divergence between the preferences (purple distributions) and the inferred environment states. During an election, a candidate offers to shape future collective event distributions according to a new set of priors, thereby resolving dissatisfaction.

Here, we propose an agent-based simulation of voting behaviours built on predictive coding and Bayesian decision-making principles. Our approach was developed from the following principles:
	1.	An optimal simulation should represent agents (e.g. citizens and candidates) embedded in a volatile social environment observed through “social events”.
	2.	Updating beliefs about the latent factors of this historical stream drives internal dissatisfaction, denoting the discrepancy between an agent’s preferences and the inferred reality.
	3.	Choosing a candidate is guided by the need to minimise this dissatisfaction.
	4.	Agents are also embedded in social networks through which they have noisy access to others’ preferences, which can influence their decisions.

An agent is defined by a fixed set of preferences $\mathcal{P} = {P_\theta}$ over the dimensions $d \in \mathcal{D}$ of the collective societal or political environment (e.g., economy, ecology, security…). These distributions are defined as sigmoid-normal (logit-normal) distributions:

$$
Z \sim \mathcal{N}(\mu, \sigma^2), \quad P = s(Z) = \frac{1}{1 + e^{-Z}}, \quad P \in (0, 1)
$$

Where $s(\cdot)$ denotes the sigmoid function.

Agents are exposed to events and information streams that reflect the evolution of the actual world state with respect to their preferences. An event $\mathcal{H}_t$ at time $t$ is defined by observations in one or several preference dimensions.

Each agent is equipped with binary filters to track current states, with input nodes encoding expectations over future values in sigmoid-normal form. This modular design allows the preference space to be represented as branches of a predictive coding neural network.

Voting Decisions

After observing a sequence of historical events, agents vote by selecting a candidate $c \in \mathcal{C}$. Like agents, candidates have fixed preferences $\mathcal{P} = {P_\theta}$, but they differ in that they offer to reshape the latent causes of future events.

Dissatisfaction is expressed as the KL divergence between current beliefs and preferences. If $P = \mathcal{N}(\mu_1, \sigma_1^2)$ and $Q = \mathcal{N}(\mu_2, \sigma_2^2)$, then:

$$
\mathrm{KL}(P | Q) = \log\left(\frac{\sigma_2}{\sigma_1}\right) + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}
$$

Let $\delta_d$ be the dissatisfaction over dimension $d$:

$$
\delta_d = \mathrm{KL}(\mathcal{N}_p | \mathcal{N}_b)
$$

Where $\mathcal{N}_p$ is the implied Gaussian of preferences and $\mathcal{N}_b$ is the belief about the latent state.

The agent then evaluates expected dissatisfaction $\hat{\delta}_d$ if a candidate is elected:

$$
\hat{\delta}_d = \mathrm{KL}(\mathcal{N}_p | \mathcal{N}_c)
$$

Where $\mathcal{N}_c$ is the candidate’s distribution.

One-Round Elections with Majority Voting

Agents choose the candidate that maximises dissatisfaction reduction:

$$
\Delta_c = \sum_{d \in \mathcal{D}} \delta_d - \hat{\delta}_d
$$

Then a categorical distribution is derived via softmax:

$$
p_c = \frac{\exp(\Delta_c)}{\sum_{c=1}^{\mathcal{C}} \exp(\Delta_j)}, \quad x \sim \mathrm{Categorical}(p)
$$

Two-Round Elections with Majority Voting

(Section to be completed)

Quadratic Voting

(Section to be completed)

Computational Modelling

The computational models are based on the generalised Hierarchical Gaussian Filter [@weber:2023] implemented in PyHGF [@legrand:2024]. This is an extension of the classic HGF [@mathys:2011; @mathys:2014] to predictive coding networks with exponential family distributions.

Bayesian inference is approximated using Hamiltonian Monte Carlo [@hoffman:2011; @betancourt:2017] via PyMC v5.23.0 [@pymc:2023].

# Results

In order to validate our model, we first conducted a series of simulations reproducing voting behaviours in various contexts. Social decisions are highly multifactorial, and a robust model of democratic behaviour should be amenable to a modularity that leverages the representation of these influences. We were therefore interested in simulating votes along historical, social, and cognitive dimensions that could influence elections. We describe here the generative models of our simulations. The values used for each dimension are reported in Table X.

One-round election with majority voting

We first started with a simple one-round election with a homogeneous population of $n$ citizens and a stationary distribution of historical events across $k$ time steps. The preferences of the agents were defined along $d$ dimensions, and the distributions were sampled from the following distributions:

$$
\begin{aligned}
\mu_d &\sim \mathcal{N}(\mu_0, \sigma_0^2) \
\sigma_d &\sim \text{HalfNormal}(\sigma_1^2)
\end{aligned}
$$

Two candidates ${C_1, C_2}$ were proposed at the election, one of which was offering a high dissatisfaction reduction ($C_1$), and the second a low dissatisfaction reduction ($C_2$). We simulated 1,000 elections where all parameters were sampled independently. The results show that, as expected, $C_1$ was elected a majority of the time (XX victories, XX losses).

⸻

Volatile events history

To be completed.

⸻

Influence of group

To be completed.

⸻

Influence of group appurtenance

To be completed.

⸻

# Discussion

# Simulation Parameters

| Parameter         | Description                                | Value / Distribution          |
|------------------|--------------------------------------------|-------------------------------|
| $\mu_0$          | Mean of preference priors                  | 0                             |
| $\sigma_0^2$     | Variance of group preference priors        | 1                             |
| $\sigma_1^2$     | Variance of individual preference noise    | HalfNormal(1)                 |
| $n$              | Number of citizens                         | 1,000                         |
| $k$              | Number of time steps                       | 10                            |
| $d$              | Number of dimensions (e.g. economy, eco)   | 3–5                           |


# Glossary

```{dropdown} KL divergence
A measure of how one probability distribution diverges from a second, expected distribution.
```

```{dropdown} Predictive coding
A theory of brain function suggesting that perception and action result from the minimisation of prediction errors.
```

```{dropdown} Quadratic voting(QV)
A voting system where individuals allocate votes based on preference intensity, with the cost of votes increasing quadratically.
```

```{dropdown} Hiearchical Gaussian Filter
A Bayesian model for hierarchical belief updating under uncertainty.
```
