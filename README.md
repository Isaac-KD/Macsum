# Macsum — Interval Aggregation Learning

> A PyTorch implementation of the **Macsum operator** for interval-valued prediction via the asymmetric Choquet integral.

Based on the research paper: **"Macsum aggregation learning"** — Y. Hmidy, A. Rico, O. Strauss (*Fuzzy Sets and Systems, 2023*).

---

## What is Macsum?

Classical regression models output a single point estimate. Macsum instead outputs an **interval** $[\underline{y},\, \overline{y}]$, representing the full range of plausible outputs for a given input — without requiring probabilistic assumptions.

This is particularly useful when a system's shift-invariance property is violated: the same input can produce different outputs due to hidden variables, measurement noise, or sensor fluctuations. Rather than collapsing this ambiguity into a single value, Macsum makes it explicit.

The model is governed by a **single learnable kernel** $\varphi \in \mathbb{R}^N$, and the interval bounds are computed via:

$$\underline{y} = \mathbb{C}_{\nu_\varphi^c}(x) \qquad \overline{y} = \mathbb{C}_{\nu_\varphi}(x)$$

where $\mathbb{C}$ denotes the asymmetric discrete Choquet integral, $\nu_\varphi$ is the (concave) Macsum set function, and $\nu_\varphi^c$ is its complementary (convex) function. By construction, the interval captures the output set of a convex family of linear functions that share the same gain.

---

## Installation

Requires Python 3.9+ and PyTorch.

```bash
pip install torch numpy tqdm
```

Clone and import directly:

```python
from Macsum import Macsum, IMSELoss, SPILoss
```

---

## Quickstart

### Training with IMSE

```python
import numpy as np
from Macsum import Macsum, IMSELoss

X_train = np.random.rand(500, 10)
y_train = np.sum(X_train, axis=1) + np.random.normal(0, 0.1, 500)

model = Macsum()
model.train(X_train, y_train, loss_fn=IMSELoss, epochs=100)

preds = model.predict(X_train[:5])
print(f"Lower  : {preds.lower}")
print(f"Center : {preds.center}")
print(f"Upper  : {preds.upper}")
print(f"Width  : {preds.mpiw}")
```

### Validation monitoring

```python
model = Macsum()
model.train(
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    loss_fn=SPILoss,
    epochs=100
)

h = model.history()
# h contains: train_loss, val_loss, train_picp, val_picp, train_mpiw, val_mpiw, lr
```

### Custom loss configuration

```python
from Macsum import SPILoss
from losses import PinballLoss

# SPILoss targeting 90% coverage (beta = 0.1)
custom_loss = lambda yt, yl, yu: SPILoss(yt, yl, yu, beta=0.1, k_sigmoid=30)
model.train(X_train, y_train, loss_fn=custom_loss, epochs=150)

# Or Pinball Loss with a specific quantile
model.train(X_train, y_train, loss_fn=PinballLoss, epochs=50)
```

---

## API Reference

### `Macsum`

The main high-level interface.

```python
model = Macsum()
```

#### `model.train(...)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `np.ndarray` | — | Input features, shape `(n_samples, n_features)` |
| `y` | `np.ndarray` | — | Target values, shape `(n_samples,)` |
| `loss_fn` | `Callable` | — | Loss function `(y_true, y_low, y_up) → scalar` |
| `X_val` | `np.ndarray` | `None` | Optional validation inputs |
| `y_val` | `np.ndarray` | `None` | Optional validation targets |
| `epochs` | `int` | `100` | Number of training epochs |
| `lr` | `float` | `0.001` | Learning rate |
| `batch_size` | `int` | `32` | Mini-batch size |
| `optimizer_cls` | `Optimizer` | `Adam` | PyTorch optimizer class |

#### `model.predict(X) → Prediction`

Returns a `Prediction` object with attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `.lower` | `np.ndarray` | Lower bound $\underline{y}$ |
| `.upper` | `np.ndarray` | Upper bound $\overline{y}$ |
| `.center` | `np.ndarray` | Midpoint `(lower + upper) / 2` |
| `.mpiw` | `np.ndarray` | Width `upper - lower` |

#### `model.history() → dict`

Returns training metrics logged across epochs:

| Key | Description |
|-----|-------------|
| `train_loss` / `val_loss` | Objective function value |
| `train_picp` / `val_picp` | Coverage probability (fraction of targets inside the interval) |
| `train_mpiw` / `val_mpiw` | Mean interval width |
| `lr` | Learning rate at each epoch |

---

## Loss Functions

All loss functions have the signature `(y_true, y_low, y_up) → torch.Tensor`.

### `IMSELoss` — Interval Mean Squared Error

Minimizes the sum of squared distances from $y$ to both bounds. Use this as a general-purpose baseline.

$$\mathcal{L} = \mathbb{E}\left[(y - \underline{y})^2 + (y - \overline{y})^2\right]$$

### `PinballLoss(beta=0.05)`

Quantile regression loss targeting symmetric coverage at level $1 - \beta$. `beta=0.05` targets a 95% prediction interval.

### `SPILoss(beta=0.05, k_sigmoid=50.0)`

Smooth Penalization Interval Loss. Jointly minimizes interval width (MPIW) while penalizing coverage violations:

$$\mathcal{L} = \mathbb{E}[\overline{y} - \underline{y}] + \frac{2}{\beta}\left(\text{pen}_{\underline{y}} + \text{pen}_{\overline{y}}\right)$$

The `k_sigmoid` parameter controls penalty sharpness. Higher values approach a hard step function.

---

## Project Structure

```
.
├── model.py      # MacsumCore — PyTorch nn.Module 
├── macsum.py     # Macsum — high-level training/inference API
├── losses.py     # IMSELoss, PinballLoss, SPILoss
├── __init__.py   # Public exports
└── test.py       # Integration tests and sanity checks
```

---

## References

Hmidy, Y., Rico, A., & Strauss, O. (2023). **Macsum aggregation learning**. *Fuzzy Sets and Systems*. 