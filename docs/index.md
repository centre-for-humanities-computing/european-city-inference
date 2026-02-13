---
hide:
  - navigation
---

# European City Inference

<p align="center">
  <img src="assets/logo.svg" width="180" alt="ECI Logo">
</p>

<p align="center">
  <strong>Agent-Based Political Election Simulator using Active Inference and JAX.</strong>
</p>

<p align="center">
  <a class="md-button md-button--primary" href="tutorials/tutorial_1_decision_making.ipynb">
    Start the Tutorials
  </a>
  <a class="md-button" href="api.md">
    Explore API
  </a>

</p>

---

## Welcome to the Documentation

**European City Inference (ECI)** provides a framework to simulate how thousands of individual voters update their beliefs and cast votes under different systems (Plurality, Quadratic).

This documentation is designed to help you navigate the project components:

<div class="grid cards" markdown>

-   :material-school: **Interactive Tutorials**
    
    * [**Decision Making**](tutorials/tutorial_1_decision_making.ipynb): Understand how the decision making work.
    * [**Voting Systems**](tutorials/tutorial_2_voting_system.ipynb): Compare Plurality vs Quadratic voting.
    * [**Polarization**](tutorials/tutorial_3_polarization.ipynb): Compare polarize position under different voting system.

-   :material-code-json: **API Reference**

    Detailed technical [**documentation**](api.md#core-simulation) for developers.
    
-   :material-book-open-page-variant: **Glossary**

    Unsure about a term? Check the [**definition**](glossary.md).

</div>

## Quick Install

Get up and running in seconds using `make` and `uv`:

```bash
git clone [https://github.com/sylvainestebe/european-city-inference.git](https://github.com/sylvainestebe/european-city-inference.git)
cd european-city-inference
make install
```