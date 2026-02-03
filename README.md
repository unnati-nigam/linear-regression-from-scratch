# Linear Regression — From Statistical Theory to Practice

This repository contains a Jupyter notebook that develops linear regression from a rigorous statistical perspective while connecting it to modern machine learning workflows.

The notebook is structured to build intuition, mathematical clarity, and implementation skills simultaneously.

---

## Model Setup

We consider the classical linear model:

$$
\mathbf{y} = X\boldsymbol{\beta} + \boldsymbol{\varepsilon}
$$

where

- $X \in \mathbb{R}^{n \times p}$ is the design matrix  
- $\boldsymbol{\beta} \in \mathbb{R}^p$ is the parameter vector  
- $\boldsymbol{\varepsilon} \sim \mathcal{N}(0, \sigma^2 I)$ is the noise  

The objective is to estimate $\boldsymbol{\beta}$ so that predictions generalize well.

---

## Ordinary Least Squares (OLS)

The OLS estimator is obtained by minimizing the squared loss:

$$
\hat{\boldsymbol{\beta}}
= \arg\min_{\boldsymbol{\beta}}
\|\mathbf{y} - X\boldsymbol{\beta}\|_2^2
$$

When $X^TX$ is invertible,

$$
\hat{\boldsymbol{\beta}}
= (X^TX)^{-1}X^T\mathbf{y}
$$

This solution corresponds to projecting $\mathbf{y}$ onto the column space of $X$.

---

## Gradient Descent View

Linear regression can also be solved iteratively.

Parameter update:

$$
\boldsymbol{\beta}^{(t+1)}
=
\boldsymbol{\beta}^{(t)}
-
\eta
\nabla_{\boldsymbol{\beta}}
\| \mathbf{y} - X\boldsymbol{\beta}^{(t)} \|_2^2
$$

which simplifies to

$$
\boldsymbol{\beta}^{(t+1)}
=
\boldsymbol{\beta}^{(t)}
+
2\eta X^T(\mathbf{y} - X\boldsymbol{\beta}^{(t)})
$$

where $\eta$ is the learning rate.

---

## Numerical Stability

The conditioning of the problem depends on the eigenvalues of $X^TX$.

A large condition number

$$
\kappa(X^TX)
=
\frac{\lambda_{\max}}{\lambda_{\min}}
$$

implies instability and high estimator variance.

---

## Ridge Regression

To control variance, we introduce $\ell_2$ regularization:

$$
\hat{\boldsymbol{\beta}}_{\text{ridge}}
=
(X^TX + \lambda I)^{-1}X^T\mathbf{y}
$$

This guarantees invertibility and shrinks coefficients toward zero.

---

## Lasso

The Lasso estimator solves:

$$
\hat{\boldsymbol{\beta}}
=
\arg\min_{\boldsymbol{\beta}}
\left(
\|\mathbf{y} - X\boldsymbol{\beta}\|_2^2
+
\lambda \|\boldsymbol{\beta}\|_1
\right)
$$

Unlike Ridge, Lasso promotes **sparsity**, making it useful for feature selection.

---

## Bias–Variance Tradeoff

Prediction risk decomposes as

$$
\mathbb{E}[(y - \hat{y})^2]
=
\text{Bias}^2 + \text{Variance} + \sigma^2
$$

Regularization increases bias but reduces variance, often lowering total risk.

---

## What This Notebook Covers

- Synthetic data generation  
- OLS from first principles  
- Conditioning and multicollinearity  
- Gradient-based optimization  
- Ridge and Lasso  
- Residual diagnostics  
- Cross-validation  
- Production-ready pipelines  

---

## Repository Structure

