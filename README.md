# Boosting — AdaBoost from Scratch

A Jupyter notebook that implements the **AdaBoost** algorithm from scratch using decision stumps as weak learners, then benchmarks the custom implementation against scikit-learn's `AdaBoostClassifier`, `GradientBoostingClassifier`, and `RandomForestClassifier` on multiple synthetic datasets.

---

## Table of contents

- [Overview](#overview)
- [What's in the notebook](#whats-in-the-notebook)
- [Algorithm design](#algorithm-design)
- [Datasets](#datasets)
- [Models compared](#models-compared)
- [Evaluation metrics](#evaluation-metrics)
- [Key results](#key-results)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [How to run](#how-to-run)
- [Notes](#notes)

---

## Overview

[AdaBoost](https://en.wikipedia.org/wiki/AdaBoost) (Adaptive Boosting) is an ensemble method that combines many *weak* learners (here, depth-1 decision trees a.k.a. **decision stumps**) into a single strong classifier by iteratively re-weighting the training samples so the next weak learner focuses on examples the previous ones got wrong.

This notebook:

1. Builds a clean, vectorized AdaBoost from scratch.
2. Demonstrates it on a non-linear dataset (concentric circles).
3. Stress-tests it on hand-crafted datasets that highlight its strengths and weaknesses (label noise, small separable data).
4. Compares it against three reference models from scikit-learn.
5. Performs hyperparameter selection using `GridSearchCV` and learning-curve inspection.

## What's in the notebook

| Section | Description |
| --- | --- |
| **Algorithm design** | Explanation of the AdaBoost loop, how weights are updated, and how each stump's contribution $\alpha$ is computed. |
| **Implementation** | A `DecisionStump` class plus an `AdaBoostCustom` class with `fit` / `predict` methods. |
| **Demonstration** | Trained on a 2-D concentric circles dataset; decision boundary is plotted and inspected. |
| **Hyperparameter selection** | Train/test error curves vs. `n_estimators` to pick a sweet spot before overfitting. |
| **Sklearn comparison** | A discussion of the differences between the custom implementation and `sklearn.ensemble.AdaBoostClassifier` (aggregation rule, error computation, base estimator). |
| **Additional datasets** | Two extra datasets designed to show specific behaviours: (1) noisy labels, (2) small linearly separable data prone to overfitting. |
| **Multi-model benchmark** | Custom AdaBoost vs. sklearn AdaBoost vs. Gradient Boosting vs. Random Forest, on each dataset. |
| **Use of generative AI** | Disclosure of how generative AI was used while writing the notebook. |

## Algorithm design

The custom AdaBoost works as follows for a binary problem with labels $y_i \in \\{-1, +1\\}$:

1. Initialize sample weights uniformly: $w_i = 1/N$.
2. For $t = 1, \dots, T$:
   - Fit a `DecisionStump` that minimizes the weighted classification error

$$ \varepsilon_t = \sum_i w_i \cdot \mathbb{1}[h_t(x_i) \neq y_i] $$

   - Compute the stump's weight

$$ \alpha_t = \tfrac{1}{2} \ln \left( \tfrac{1 - \varepsilon_t}{\varepsilon_t} \right) $$

   - Update sample weights $w_i \leftarrow w_i \cdot e^{-\alpha_t y_i h_t(x_i)}$ and renormalize.

3. Final prediction:

$$ \hat{y}(x) = \mathrm{sign} \left( \sum_t \alpha_t \, h_t(x) \right) $$

### Differences vs. `sklearn.ensemble.AdaBoostClassifier`

- **Aggregation**: the custom implementation uses a weighted sum of $\alpha$ values + `np.sign`. Sklearn uses a weighted majority vote and natively supports multiclass (SAMME / SAMME.R).
- **Error computation**: the custom version computes the weighted error directly; sklearn uses a numerically optimized routine.
- **Base estimator**: the custom version is hard-wired to a `DecisionStump`. Sklearn accepts any base estimator (defaults to `DecisionTreeClassifier(max_depth=1)`).

## Datasets

| # | Dataset | Why it's interesting |
| --- | --- | --- |
| 1 | Concentric circles (`make_circles`, 2-D) | Classic non-linear toy dataset; great for visualising decision boundaries. |
| 2 | Noisy classification | A subset of points have flipped labels, exposing AdaBoost's known sensitivity to label noise. |
| 3 | Small, perfectly separable | Few samples with a clean margin — encourages overfitting. |

## Models compared

- `AdaBoostCustom` — our from-scratch implementation.
- `sklearn.ensemble.AdaBoostClassifier`.
- `sklearn.ensemble.GradientBoostingClassifier` — another boosting algorithm.
- `sklearn.ensemble.RandomForestClassifier` — non-boosting baseline (bagging).

For each model, key hyperparameters (`n_estimators`, `learning_rate`, `max_depth`) are tuned with `GridSearchCV` or with manual sweeps.

## Evaluation metrics

- **Accuracy** — overall correctness.
- **F1-score** — balances precision and recall (useful for the noisy dataset).
- **ROC-AUC** — separability between the two classes.
- **Precision** — sensitivity to false positives.

Decision boundaries are also plotted to make the qualitative behaviour of each model visible.

## Key results

- On the **noisy** dataset, the custom AdaBoost reaches ~85% accuracy / 92% precision but produces a jagged decision boundary, illustrating its tendency to overweight mislabeled points. Random Forest is slightly more accurate (~86%) and clearly smoother thanks to bagging.
- On the **small separable** dataset, the custom AdaBoost overfits (~82%), while sklearn's AdaBoost (~94%) and Gradient Boosting (~96%) handle it much better.
- A hyperparameter sweep on `n_estimators` shows training error continues to drop while test error plateaus around ~200–300 estimators, justifying the chosen configuration.

See the notebook for full plots, tables, and discussion.

## Project structure

```
ml-boosting-notebook/
├── Boosting.ipynb       # The main notebook
├── README.md            # This file
├── requirements.txt     # Python dependencies
└── .gitignore
```

## Requirements

- Python 3.9 or newer
- `numpy >= 1.15.4`
- `scikit-learn`
- `matplotlib`
- `jupyter`

Install everything with:

```bash
pip install -r requirements.txt
```

## How to run

```bash
git clone https://github.com/galsnir/ml-boosting-notebook.git
cd ml-boosting-notebook
pip install -r requirements.txt
jupyter notebook Boosting.ipynb
```

Then run the cells top-to-bottom. The notebook is fully self-contained — no external data files are required (datasets are generated with scikit-learn helpers such as `make_circles`, `make_classification`, `make_moons`, and `make_gaussian_quantiles`).

## Notes

- A fixed `np.random.seed(42)` is used for reproducibility.
- The grid searches can take a few minutes on a laptop CPU.
- The notebook discloses where generative AI was used (mainly for rephrasing explanations and helping with plotting boilerplate).
