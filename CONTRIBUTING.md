# Contributing to ECI

Thanks for your interest in contributing to **European City Inference (ECI)**!
This document walks you through everything you need to know to get a development
environment running, the conventions we follow, and what we expect in a pull
request.

If something here is unclear or out of date, please open an issue — that
counts as a contribution too.

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/sylvainestebe/european-city-inference.git
cd european-city-inference
make install
```

`make install` will:

- install `uv` if you do not have it,
- create a `.venv/`,
- install all runtime + dev dependencies,
- install the package in editable mode.

### 2. Install the pre-commit hooks

```bash
uv run pre-commit install
```

From now on, every commit will run `ruff format`, `ruff check`, `mypy`, and
`nbstripout` (strips notebook outputs so diffs stay reviewable). If a hook
modifies files, `git add` them and commit again.

### 3. Verify your setup

```bash
make tests
```

If the test suite passes you are ready to go.

---

## Codebase layout

```
src/eci/
├── agents.py                # Voter, Candidate dataclasses
├── environment.py           # EnvConfig + Environment (HGF wiring)
├── perceptual.py            # HGF perceptual model wrapper
├── population.py            # voter / candidate parameter sampling
├── world.py                 # observation-stream generation
├── observations.py          # synthetic observation generators
├── metrics.py               # collective-outcome metrics
├── utils.py                 # KL divergence + env-extraction helpers
├── decision/
│   ├── response.py          # response functions (+ ResponseFunction protocol)
│   ├── scoring.py           # pluggable candidate-utility scoring
│   ├── utilities.py         # KL gaps + candidate-utility computation
│   └── sampling.py          # softmax / categorical vote sampling
├── voting/
│   ├── plurality.py         # plurality voting rule
│   └── quadratic.py         # quadratic voting rule + allocation
└── plots/                   # belief / preference / voting plot helpers
tests/                       # mirrors src/eci/ — one test file per module
notebooks/                   # tutorials + experimental analyses
docs/                        # mkdocs site source
scripts/                     # one-off CLI scripts
```

When you add a new module under `src/eci/`, please add a matching
`tests/test_eci_<module>.py` file.

---

## Running tests

```bash
make tests                                        # full suite + coverage report
uv run pytest tests/test_eci_environment.py -v    # one file
uv run pytest -k "quadratic and tie" -v           # by keyword
```

Coverage is configured in `pyproject.toml`. After `make tests`, open
`htmlcov/index.html` for the line-by-line view.

### Notebooks are tests too

The CI pipeline executes every notebook in `notebooks/` (`make
run-all-notebooks` locally). If you add a notebook, make sure it runs
end-to-end on a clean kernel. Heavy notebooks should keep their compute
under ~10 min per cell — there is a 600 s timeout configured in CI.

---

## Code style

We use [`ruff`](https://github.com/astral-sh/ruff) for formatting and linting,
and [`mypy`](https://mypy.readthedocs.io/) for type checking. Both run in
pre-commit and CI; PRs that fail will be marked red.

```bash
make lint                  # format + lint
uv run mypy src/           # type check
```

### Docstrings

We use the **NumPy docstring style** (enforced via `ruff`'s `pydocstyle`
rules). Every public function should document:

- a one-line summary,
- a `Parameters` section,
- a `Returns` section with array shapes when relevant.

### JAX conventions

- Prefer `jax.numpy` (`jnp`) over `numpy` inside functions consumed by JAX
  pipelines (`vmap`, `jit`).
- Functions touched by `vmap` must be pure (no side effects, no Python `if`
  on traced arrays — use `jnp.where` or `lax.cond`).
- PRNG keys are explicit — split, never reuse.

---

## Pull request workflow

1. **Branch** from `main`:

   ```bash
   git checkout -b feature/short-descriptive-name
   ```

2. **One logical change per PR.** A PR that fixes a bug, refactors a module
   *and* adds a new feature is hard to review and easy to revert badly.

3. **Add or update tests** for any behavioural change.

4. **Update `CHANGELOG.md`** under the `## Unreleased` section.

5. **Run locally before pushing**:

   ```bash
   make lint
   make tests
   uv run mypy src/
   ```

6. **Open the PR** — fill in the template completely. A reviewer will get
   back to you within a few days.

7. **Address review** — push fixups, don't force-push during review unless
   asked. Once approved, the maintainer will squash-merge.

---

## Reporting bugs

Use the **bug report** issue template. The most useful bug reports include:

- the smallest snippet that reproduces the problem,
- the output (full traceback if it crashes),
- your Python version (`python --version`) and OS,
- the ECI version or commit hash (`git rev-parse HEAD`).

---

## Proposing new features

Open a **feature request** issue *before* writing code. We will discuss the
scope, the API, and whether it fits the project's goals. This saves you from
writing a PR that gets rejected for design reasons.

---

## Code of conduct

Be kind. Assume good faith. Disagree with ideas, not people. We follow the
[Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
Maintainers reserve the right to remove comments, commits, or contributors
who do not respect this baseline.

---

## Questions?

- Open a [GitHub Discussion](https://github.com/sylvainestebe/european-city-inference/discussions) for general questions.
- Email the maintainers: `sylvainestebe@cas.au.dk`, `nicolas.legrand@cas.au.dk`.

Thanks again for contributing!
