
# Macsum Aggregation Learning

This repository provides a Python implementation of the **Macsum operator**, a new interval-valued linear operator for multi-input/single-output (MISO) system identification. The model is designed to represent system incoherence and imprecision by replacing a single linear function with a convex set of linear functions.

This work is based on the research paper:  
**"Macsum aggregation learning"** by Yassine Hmidy, Agnès Rico, and Olivier Strauss (*Fuzzy Sets and Systems, 2023*).

## Overview

The Macsum model aims to handle systems where the shift invariance property is not fulfilled, meaning the same input can lead to different outputs due to hidden variables or sensor fluctuations. 

The aggregation function yields an interval-valued output $[y_{low}, y_{up}]$, representing the lack of accuracy in predicting the system output. Key characteristics include:
* **Single Parametric Vector**: The model is ruled by a single precise kernel $\varphi$, despite the imprecise output.
* **Choquet Integral**: The aggregation is based on the asymmetric discrete Choquet integral with respect to the Macsum operator.
* **Linearity**: The model acts as an interval-valued linear aggregation, preserving homogeneity and a constant gain.

## Key Features

* **Differentiable Implementation**: Optimized using PyTorch's Autograd for efficient gradient descent.
* **Flexible Loss Functions**: Includes implementations for IMSE (Interval Mean Squared Error), Pinball Loss, and SPILoss (Smooth Penalization Interval Loss).
* **Performance Metrics**: Built-in tracking for PICP (Prediction Interval Coverage Probability) and MPIW (Mean Prediction Interval Width).

## Project Structure

* `model.py`: Contains the `MacsumCore` PyTorch module implementing the Choquet integral logic.
* `macsum.py`: The high-level API wrapper for training and prediction.
* `losses.py`: Definitions for various interval-based cost functions.
* `test_simple.py`: A professional integration script to validate the installation and data flow.

## Installation

Ensure you have a Python 3.9+ environment with the following dependencies:

```bash
pip install torch numpy tqdm matplotlib
```

## Usage

### Basic Training with IMSE

```python
import numpy as np
from Macsum import Macsum, IMSELoss

# 1. Prepare Data (NumPy arrays)
X_train = np.random.rand(500, 10)
y_train = np.sum(X_train, axis=1) + np.random.normal(0, 0.1, 500)

# 2. Initialize Model
model = Macsum()

# 3. Train using Interval Mean Squared Error
model.train(X_train, y_train, loss_fn=IMSELoss, epochs=100)

# 4. Predict
preds = model.predict(X_train[:5])
print(f"Prediction Center: {preds.center}")
```

### Advanced Calibration (SPILoss)

Use the Smooth Penalization Interval Loss to target a specific coverage rate (e.g., 90%):

```python
from Macsum import SPILoss

# Configure loss with beta=0.1 for 90% target coverage
target_loss = lambda yt, yl, yu: SPILoss(yt, yl, yu, beta=0.1)
model.train(X_train, y_train, loss_fn=target_loss, epochs=150)
```

## Metrics and Monitoring

The `model.history()` method returns a dictionary containing:
* `train_loss` / `val_loss`: Evolution of the objective function.
* `train_picp`: Proportion of ground truth values captured within the predicted intervals.
* `train_mpiw`: Average width of the predicted intervals.

## Mathematical Context

The Macsum operator $\nu_\varphi$ is a concave set function, and its complementary $\nu_\varphi^c$ is convex. The output interval is defined by:
* Lower bound: $\underline{y} = \mathbb{C}_{\nu_\varphi^c}(x)$
* Upper bound: $\overline{y} = \mathbb{C}_{\nu_\varphi}(x)$

By construction, the model ensures that the interval represents the set of all outputs obtained by a convex set of linear functions sharing the same gain.

## References

[1] Y. Hmidy, A. Rico, and O. Strauss. "Macsum aggregation learning." *Fuzzy Sets and Systems*, 2023.
```