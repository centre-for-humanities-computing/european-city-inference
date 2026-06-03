# Write your own response function

ECI's voting rules (`_vote_plurality`, `_vote_quadratic`) do not hard-code
*how* each agent decides which candidate to favour. That logic lives in a
swappable component called the **response function**. By writing your own,
you can model:

- novel decision rules (e.g. probit instead of softmax),
- bounded rationality (e.g. choice with a temperature parameter),
- strategic voting (e.g. utilities adjusted by viability),
- entirely new utility models (e.g. prospect theory).

This page explains the contract and walks through implementing one.

---

## The contract

A response function is any callable that satisfies the
[`ResponseFunction`][eci.decision.response.ResponseFunction] protocol:

```python
def my_response(data, key, mask=None) -> tuple[Array, Array, Array, Array]:
    ...
```

| Argument | Meaning | Shape |
|---|---|---|
| `data` | Agent state dict | `{"beliefs", "preferences", "candidates"}` × `{"mean", "precision"}` |
| `key` | JAX PRNG key | scalar key |
| `mask` | Optional boolean — `False` excludes a candidate (e.g. runoff round) | `(n_candidates,)` |

| Return | Meaning | Shape |
|---|---|---|
| `vote` | Sampled candidate index per agent | `(n_agents,)` |
| `softmax_probs` | Vote distribution per agent (rows sum to 1) | `(n_agents, n_candidates)` |
| `candidate_utilities` | Raw scores before softmax — QV reads this | `(n_agents, n_candidates)` |
| `next_key` | Fresh PRNG key | scalar key |

Quadratic voting consumes `candidate_utilities` directly to allocate its
budget, so any function you write must return meaningful utilities — not
just the sampled vote.

---

## Step-by-step example: temperature-controlled softmax

A common extension is to add a **temperature** to the softmax: low
temperature → sharp / decisive vote, high temperature → diffuse / hesitant.

### 1. Write the function

```python
import jax
import jax.numpy as jnp

from eci.decision import _compute_candidate_utilities


def response_function_temperature(data, key, mask=None, temperature: float = 1.0):
    """Sample a vote using softmax with a tunable temperature.

    A temperature < 1 sharpens the choice (more decisive).
    A temperature > 1 flattens it (more random).
    """
    utilities, _, _ = _compute_candidate_utilities(data)
    utilities = utilities / temperature
    if mask is not None:
        utilities = jnp.where(mask, utilities, -jnp.inf)

    sample_key, next_key = jax.random.split(key)
    softmax_probs = jax.nn.softmax(utilities, axis=1)
    vote = jax.random.categorical(sample_key, utilities, axis=1)
    return vote, softmax_probs, utilities, next_key
```

### 2. Use it with any voting rule

```python
from functools import partial
from eci.voting import _vote_plurality

# Decisive voters
decisive = partial(response_function_temperature, temperature=0.3)
result = _vote_plurality(data, decisive, key)

# Hesitant voters
hesitant = partial(response_function_temperature, temperature=3.0)
result = _vote_plurality(data, hesitant, key)
```

That's it. No registration, no subclassing. The voting rule treats your
function exactly like the built-in `response_function`.

### 3. Verify it conforms (recommended)

```python
from eci.decision import ResponseFunction

assert isinstance(response_function_temperature, ResponseFunction)
```

This is a runtime check on the signature. It catches typos and missing
return values before they hit the simulation loop.

---

## Things to watch out for

### Always return all four values
Even if your function does not use `candidate_utilities` internally,
quadratic voting will read it and silently produce nonsense if you return
zeros. Compute meaningful utilities.

### Stay JAX-pure
The voting pipeline `vmap`s over keys. Your function must be a pure
function — no Python `if` on traced arrays (use `jnp.where`), no
`np.random`, no global state.

### Respect the mask
`mask=None` means "all candidates are eligible". When `mask` is provided,
masked-out candidates must end up with `softmax_probs == 0` (set their
utility to `-jnp.inf` before softmax). Multi-round voting (runoff)
depends on this.

### Use the key, then split it
The convention is `sample_key, next_key = jax.random.split(key)`. Use
`sample_key` for `categorical`, return `next_key` so the caller can chain
further random operations without reusing the same draw.

---

## Reference

::: eci.decision.response.ResponseFunction
    options:
      show_root_heading: false

::: eci.decision.response.response_function

::: eci.decision.response.response_function_logpdf

::: eci.decision.response.response_function_pref
