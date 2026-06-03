# Changelog

All notable changes to **ECI (European City Inference)** will be documented
in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Skeleton sub-packages `eci.data` (schemas, loaders, transformers) and
  `eci.fit` (priors, models, diagnostics) for the upcoming v0.2
  calibration pipeline.
- `eci.voting_system.ResponseFunction` — a runtime-checkable
  `typing.Protocol` formalising the extension contract for vote-sampling
  response functions. Anyone can now write a custom response function and
  plug it into `_vote_plurality` / `_vote_quadratic` without registration.
- `eci.voting_system.scoring` — pluggable candidate-utility scoring with
  four built-in strategies (`score_normalized`, `score_absolute`,
  `score_inverted`, and the new dissatisfaction-weighted `score_product`),
  injectable into `_compute_candidate_utilities` via the `scoring_fn`
  parameter. Replaces the previous block of commented-out alternative
  formulas with real, testable functions.
- `eci.voting_system.VoteResult` — uniform `TypedDict` return type for
  all voting rules. Adds `votes_matrix` (n_agents, n_candidates) and
  `votes_per_candidate` (n_candidates,) keys to both plurality and QV
  results so downstream code (e.g. `batch_compute_metrics`) no longer
  needs to branch on which voting rule produced the result. Legacy keys
  (`votes`, `qv_votes_matrix`) are preserved for backward compatibility
  and will be removed in v0.2.
- `eci.voting_system._sample_from_utilities` — DRY helper that holds the
  common tail (mask + key split + sample + return tuple) of every
  response function. Each built-in response function is now a 2-3-line
  body.
- `eci.population` — :class:`Population` (registered as a JAX pytree via
  `jax.tree_util.register_dataclass`) holding voter and candidate
  arrays, plus :class:`PopulationConfig`.
- `eci.world` — :class:`World` (JAX pytree) wrapping the observation
  stream, plus :class:`WorldConfig`.
- `eci.perceptual` — :class:`PerceptualModel` stateful wrapper around
  the HGF network, plus :class:`HGFConfig`. Exposes a single
  `.run(population, world)` entry point returning trajectories as a
  pytree.

### Changed
- `Environment` is now a thin orchestrator that composes
  :class:`Population`, :class:`World` and :class:`PerceptualModel`. All
  legacy attributes (`env.voters`, `env.candidates`, `env.input_data`,
  `env.network`, `env.last_attributes`, `env.node_trajectories`,
  `env.preferences_idx`) are preserved as proxies — notebooks and tests
  continue to work without modification.
- `EnvConfig` remains the user-facing flat config; sub-configs
  (`PopulationConfig`, `WorldConfig`, `HGFConfig`) are derived
  internally via `.from_env_config()` classmethods.
- **Repository layout refactor**: the codebase is now organised by
  *concern* rather than by accident. New top-level structure under
  ``src/eci/``:
    - ``decision/`` (new) — agent-level decision making
      (``scoring``, ``utilities``, ``sampling``, ``response``).
    - ``voting/`` (renamed from ``voting_system/``) — vote aggregation
      only (``plurality``, ``quadratic``, ``types``, ``utils``).
    - ``plots/`` (split from monolithic ``plots.py``) — per-topic plot
      modules (``belief``, ``preference``, ``winners``, ``voting``).
    - ``observations.py`` (new, extracted from ``utils.py``) — synthetic
      observation generation.
    - ``utils.py`` is now thin: only ``kl_divergence``,
      ``get_voter_trajectory_data``, ``_extract_env_data_vectorized``.

### Removed (breaking)
- ``eci.voting_system`` package removed. Imports must move to
  ``eci.decision`` (for response functions, scoring, utilities,
  sampling) or ``eci.voting`` (for voting rules, ``VoteResult``,
  ``_find_winner``).
- ``eci.plots`` is now a package; the monolithic ``eci.plots`` module
  is gone but all public names re-export from the new package — no
  import change needed at the top-level ``eci.plots`` symbol.
- ``eci.utils`` no longer exports ``generate_observations``,
  ``_get_parameter_trajectory``, ``PhaseParams``,
  ``_validate_observation_args``, ``_resolve_shock_times``,
  ``_sample_beta_signal``, ``_rescale_and_add_noise``,
  ``_find_winner``, ``_find_top_k_winners``. See the **Removed**
  table above for new locations.
- Documentation page `docs/extending_response_functions.md` walking
  through a custom temperature-controlled softmax implementation.
- Tests in `tests/test_eci_response_protocol.py` verifying that all
  three built-in response functions satisfy the Protocol and that a
  user-defined function plugs in seamlessly.
- `CONTRIBUTING.md` describing the development workflow.
- Issue templates (`bug_report.md`, `feature_request.md`,
  `config.yml`) and `PULL_REQUEST_TEMPLATE.md` under `.github/`.
- `EnvConfig` exposes three new fields: `dispersion`,
  `shock_pattern`, `obs_seed` — making world-noise calibration possible
  from the public API.
- New tutorial `tutorial_5_belief_to_vote.ipynb` — visualises the
  belief→vote distribution mapping for a single voter under both
  plurality and quadratic voting.
- `poster_figures.ipynb` — regenerates the Neuromonster poster figures
  at print quality.
- Math + algorithm explainers in `tutorial_2_voting_system.ipynb` for
  both plurality and quadratic voting (per-step softmax and Gumbel-top-k
  + square-root rule).

### Changed
- Tutorials renamed and renumbered to short, consistent names:
  - `tutorial_1_how_the_decision_making_work` → `tutorial_1_decision_making`
  - `tutorial_2_how_the_voting_system_work`   → `tutorial_2_voting_system`
  - `tutorial_3_how_the_environement_work`    → `tutorial_3_environment`
  - `tutorial_5_how_the_precision_work`       → `tutorial_4_precision`
  - `tutorial_6_belief_to_vote`               → `tutorial_5_belief_to_vote`
- `docs/tutorials/` now contains symlinks to `notebooks/` instead of
  stale copies — single source of truth.

### Changed
- `ci.yml` no longer runs `mkdocs gh-deploy` (kept exclusively in
  `documentation.yml`); adds an explicit `mypy` step; bumps notebook
  execution timeout to 600 s/cell.
- `.pre-commit-config.yaml` adds `nbstripout` and the standard
  `pre-commit-hooks` (trailing-whitespace, end-of-file-fixer,
  check-added-large-files, check-yaml, check-toml,
  check-merge-conflict).
- `.gitignore` restructured by category and now excludes `figures/`,
  `plots/`, `results/`.

### Fixed
- `scenario=1` vs `scenario=2` of `generate_observations` produced the
  same trajectory because `shock_pattern` was not wired through from
  `EnvConfig`. Both are now exposed and threaded end-to-end.
- The world-volatility tutorial grid actually varies observation
  dispersion across cells now that `dispersion` reaches
  `generate_observations` (previously the dict of dispersion values was
  built but never threaded through to `EnvConfig`).

---

## [0.1.0] — 2026-05-27

First public release, prepared for the Neuromonster conference poster.

### Added
- Agent-based simulation backbone (`eci.environment.Environment`,
  `eci.environment.EnvConfig`).
- HGF-based perceptual inference for voter agents via `pyhgf`.
- Two voting rules: plurality (`_vote_plurality`) and quadratic
  (`_vote_quadratic`) with a sequential allocation.
- Three built-in response functions:
  `response_function`, `response_function_logpdf`, `response_function_pref`.
- Five didactic notebooks (`tutorial_1_decision_making`,
  `tutorial_2_voting_system`, `tutorial_3_environment`,
  `tutorial_4_precision`, `tutorial_5_belief_to_vote`).
- Test suite covering `agents`, `environment`, `utils`, `metrics`,
  `plots`, and the three voting modules.
- MkDocs Material documentation site with `mkdocstrings`-generated
  API reference and `mkdocs-jupyter`-rendered tutorials.
- CI on GitHub Actions (`uv` + ruff + pytest + notebook execution).
- MIT licence.

---

[Unreleased]: https://github.com/sylvainestebe/european-city-inference/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/sylvainestebe/european-city-inference/releases/tag/v0.1.0
