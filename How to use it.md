Of course\! Here is a tutorial in English based on the provided code snippets, explaining how to create agents and candidates, and how to run and visualize a voting simulation.

-----

# Tutorial: Simulating a Two-Round Election with Multi-Agent Systems

This tutorial walks you through setting up, running, and visualizing a multi-agent simulation of a two-round election. We'll cover how to:

1.  **Create Candidates**: Define candidates with specific preferences on a range of issues.
2.  **Create Agents**: Build agents with an internal belief network to evaluate candidates.
3.  **Run & Analyze Simulations**: Execute a two-round voting simulation and analyze the results using powerful visualization tools.

Let's get started\! 🗳️

-----

## How to Create a Candidate

First, we need to define the candidates in our election. Each candidate is characterized by a set of preferences, where each preference corresponds to a stance on a specific issue.

We model each preference as a normal distribution defined by a mean ($\\mu$) and a standard deviation ($\\sigma$). The mean represents the candidate's position on the issue, and the standard deviation represents their conviction or flexibility.

### Implementation

The following Python code generates a specified number of candidates (`n_candidates`), each with a set of `n_preferences`. The $\\mu$ and $\\sigma$ for each preference are drawn from random distributions.

```python
# Generate candidates with preferences
# Each candidate has a list of preferences, each preference is a tuple (mu, sigma)
extreme_indices = np.random.choice(n_candidates, replace=False)
candidates = []
for i in range(n_candidates):
    preferences = []
    mu_sigma = 1
    sigma_scale = 1
    for pref in range(n_preferences):
        mu = norm.rvs(2, mu_sigma)
        sigma = halfnorm.rvs(scale=sigma_scale)
        mu_sigma_tuple = (np.float64(mu), np.float64(sigma))
        preferences.append(mu_sigma_tuple)
    candidates.append(tuple(preferences))

print("Content of candidates:")
print(candidates) # candidates is a list of tuples, each tuple contains preferences formalized as (mu, sigma) pairs
```

Here, `candidates` becomes a list of tuples. Each tuple represents one candidate and contains their preference pairs `(mu, sigma)`.

-----

## How to Create an Agent

Next, we create the agents who will vote in the election. An agent's decision-making process is modeled using a belief network. This network structure allows the agent to process information about candidates' preferences and form a judgment.

### Implementation

We'll use a custom `Network` class to build the agent's internal model. The network consists of:

  * **Binary-state nodes**: One for each preference, representing a basic "agree/disagree" state.
  * **Value nodes**: Linked to each binary-state node to help evaluate the information.

<!-- end list -->

```python
# Create agent preferences dynamically based on n_preferences
network = Network()

# Add binary-state nodes to the network
network.add_nodes(kind="binary-state", n_nodes=n_preferences)

# Add value nodes for each binary-state node
for i in range(n_preferences):
    network.add_nodes(value_children=i)

# Plot the network to visualize its structure
network.plot_network()
```

This code initializes a network that is structurally identical for all agents, providing a common framework for how they perceive and evaluate candidates.

-----

## Describe Simulation & Scenarios

With our candidates and agents defined, we can now run the election simulation. The simulation consists of multiple iterations of a two-round voting process. This allows us to observe trends and variability in the election outcomes.

### The Simulation Loop

The main loop runs the election process. In each simulation:

1.  **First Round**: All agents vote for their preferred candidate from the full list.
2.  **Second Round**: Only the top two candidates from the first round advance. The agents vote again, choosing between these two finalists.
3.  **Results**: The vote proportions for both rounds are recorded.

The `get_votes` function (assumed to be defined elsewhere) is vectorized using JAX's `vmap` for efficient computation across all agents.

```python
# Assuming get_votes and other necessary functions are defined elsewhere
results = []
simulation_number = 0

while simulations > 0:
    # Initialize random keys for JAX
    key = random.PRNGKey(int(time.time()))
    keys = random.split(key, n_agents)

    # Setup and vectorize the get_votes function for the first round
    get_votes_fn = Partial(
        get_votes,
        network=copy.deepcopy(network),
        input_data=input_data,
        n_preferences=n_preferences,
        candidates=candidates,
    )
    vmap_get_votes_fn = vmap(get_votes_fn)
    attribute, nodes_traje = vmap_get_votes_fn(tonic_volatilities, keys)
    votes_1st_round = attribute[-1]["votes"]
    counts = np.bincount(votes_1st_round)

    # Calculate proportions for the first round
    proportions_1st = [v / sum(counts) for v in counts] if sum(counts) > 0 else []

    # Check if there are at least two candidates for the second round
    if len(counts) >= 2:
        top2 = np.argsort(counts)[-2:][::-1]
        top_two_candidates = [candidates[i] for i in top2]

        # Setup and vectorize the get_votes function for the second round
        get_votes_fn_2nd = Partial(
            get_votes,
            network=copy.deepcopy(network),
            input_data=input_data,
            n_preferences=n_preferences,
            candidates=top_two_candidates,
        )
        vmap_get_votes_fn_2nd = vmap(get_votes_fn_2nd)
        attribute2, nodes_traje2 = vmap_get_votes_fn_2nd(tonic_volatilities, keys)
        votes_2nd_round = attribute2[-1]["votes"]
        counts_2nd = np.bincount(votes_2nd_round)
        
        # Calculate proportions for the second round
        proportions_2nd = [v / sum(counts_2nd) for v in counts_2nd] if sum(counts_2nd) > 0 else []
    else:
        top_two_candidates = []
        proportions_2nd = []

    # Store results for this simulation
    simulation_number += 1
    results.append({
        "simulation_number": simulation_number,
        "first_round": {"proportions": proportions_1st, "candidates": candidates},
        "second_round": {"proportions": proportions_2nd, "candidates": top_two_candidates},
    })
    simulations -= 1
```

### Data Processing

After running the simulations, we need to process the raw results into a clean format suitable for analysis and plotting. We normalize the data into a Pandas DataFrame where each row represents a single candidate's result in a specific round and simulation.

```python
# Normalize the data for the DataFrame
normalized_data = []
for result in results:
    simulation_number = result["simulation_number"]
    # Add first round data
    for candidate, proportion in zip(result["first_round"]["candidates"], result["first_round"]["proportions"]):
        normalized_data.append({
            "simulation_number": simulation_number,
            "round": "first",
            "candidate": candidate,
            "proportion": proportion
        })
    # Add second round data
    for candidate, proportion in zip(result["second_round"]["candidates"], result["second_round"]["proportions"]):
        normalized_data.append({
            "simulation_number": simulation_number,
            "round": "second",
            "candidate": candidate,
            "proportion": proportion
        })

# Create a DataFrame from the normalized data
final_df = pd.DataFrame(normalized_data)
```

To make plotting easier, we convert the complex `candidate` data into a unique numerical ID.

```python
# Convert 'candidate' column to categorical data type and assign a unique ID
final_df['candidate'] = final_df['candidate'].astype('category')
final_df['candidate_id'] = final_df['candidate'].cat.codes

# Split the DataFrame into first and second rounds
df_first_round = pd.DataFrame(final_df[final_df['round'] == 'first'])
df_second_round = pd.DataFrame(final_df[final_df['round'] == 'second'])
```

### Visualizing Election Results 📈

Visualizing the results helps us understand the dynamics of the election. We'll use Altair to create interactive stacked area charts showing how vote proportions evolve across simulations.

#### First Round Results

This chart shows the proportion of votes each candidate received in the first round across all simulations.

```python
# Make a copy of the original DataFrame for the first round
df = df_first_round.copy()
all_candidates = sorted(df['candidate_id'].unique())
color_scale = alt.Scale(domain=all_candidates, scheme='pastel1')
highlight = alt.selection_interval(bind='scales', encodings=['x'])

# Create the area chart for vote proportions in the first round
chart_first = alt.Chart(df[df['round'] == 'first']).mark_area(opacity=0.85).encode(
    x=alt.X('simulation_number:N', title='Simulation Number', axis=alt.Axis(labelAngle=0)),
    y=alt.Y('proportion:Q', stack='center', title='Vote Proportion', axis=alt.Axis(format='.0%')),
    color=alt.Color('candidate_id:N', scale=color_scale, legend=alt.Legend(title='Candidate')),
    tooltip=[
        alt.Tooltip('candidate_id:N', title='Candidate'),
        alt.Tooltip('proportion:Q', title='Proportion', format='.1%'),
        alt.Tooltip('simulation_number:N', title='Simulation')
    ]
).add_selection(
    highlight
).properties(
    width=1000,
    height=250,
    title="Evolution of Vote Proportions – First Round"
)

chart_first
```

#### Second Round Results

Similarly, this chart visualizes the results of the second-round runoff between the top two candidates.

```python
# Make a copy of the DataFrame for the second round
df = df_second_round.copy()
all_candidates = sorted(df['candidate_id'].unique())
color_scale = alt.Scale(domain=all_candidates, scheme='category20b')
highlight = alt.selection_interval(bind='scales', encodings=['x'])

# Create the area chart for vote proportions in the second round
chart_second = alt.Chart(df).mark_area(opacity=0.85).encode(
    x=alt.X('simulation_number:N', title='Simulation Number', axis=alt.Axis(labelAngle=0)),
    y=alt.Y('proportion:Q', stack='center', title='Vote Proportion', axis=alt.Axis(format='.0%')),
    color=alt.Color('candidate_id:N', scale=color_scale, legend=alt.Legend(title='Candidate')),
    tooltip=[
        alt.Tooltip('candidate_id:N', title='Candidate'),
        alt.Tooltip('proportion:Q', title='Proportion', format='.1%'),
        alt.Tooltip('simulation_number:N', title='Simulation')
    ]
).add_selection(
    highlight
).properties(
    width=700,
    height=350,
    title="Evolution of Vote Proportions – Second Round"
)

chart_second
```

### Visualizing Agent Beliefs 🧠

Finally, we can look inside the agents' "minds" to see how their beliefs evolve over time. The following Matplotlib code plots the trajectory of the agents' expected mean for several key preferences. This shows how agents update their internal beliefs as they process information during a single simulation.

The `nodes_traje` variable, obtained from the simulation output, contains the time-series data of each agent's internal network states.

```python
# Set global parameters for the font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Define line styles
line_styles = ["-"] * 10  # All solid lines

# Create a figure with 3 side-by-side subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True, facecolor='#f9f9f9')

# List of preferences and their corresponding indices
preferences = [0, 1, 2]
pref_labels = ['Preference 1', 'Preference 2', 'Preference 3']

# Generate a pastel color palette
def generate_pastel_colors(n):
    pastel_colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.4
        lightness = 0.85
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        pastel_colors.append(rgb)
    return generate_pastel_colors(n_agents)

pastel_colors = generate_pastel_colors(n_agents)

for idx, pref in enumerate(preferences):
    ax = axes[idx]
    ax.set_facecolor('#f9f9f9')
    for agent_idx in range(n_agents):
        color = pastel_colors[agent_idx]
        alpha = 0.4 + 0.6 * (agent_idx / n_agents)

        ax.plot(
            nodes_traje[pref]["expected_mean"][agent_idx],
            label=f'Agent {agent_idx + 1}' if idx == 0 else "",
            color=color,
            linestyle=line_styles[agent_idx % len(line_styles)],
            linewidth=1.5,
            alpha=alpha
        )
    ax.set_xlabel('Time Step', fontsize=12, fontweight='bold')
    ax.set_title(f'Trajectory of Expected Mean ({pref_labels[idx]})', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4, color='#e0e0e0', linewidth=0.5)

    for spine in ax.spines.values():
        spine.set_edgecolor('#e0e0e0')
        spine.set_linewidth(0.8)

# Add a common legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1, 1), title="Agents")

# Add a global title
fig.suptitle('Trajectories of Expected Means for Different Preferences', fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

```

These plots reveal how individual agents dynamically adjust their internal representations of the issues over the time steps of the simulation, leading to their final vote.