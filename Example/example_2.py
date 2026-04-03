import numpy as np
from Macsum import Macsum, IMSELoss, SPILoss
from losses import PinballLoss


# 1. Dummy data preparation
X_train = np.random.rand(200, 10)
y_train = np.random.rand(200)

model = Macsum()

# Example A: Training with IMSE (no additional parameters)
model.train(X_train, y_train, loss_fn=IMSELoss, epochs=50)

# Example B: Training with default Pinball Loss (beta=0.05)
model.train(X_train, y_train, loss_fn=PinballLoss, epochs=50)

# Example C: Training with SPILoss using beta=0.1
# Using a lambda allows you to fix the beta parameter!
custom_spi = lambda y_t, y_l, y_u: SPILoss(y_t, y_l, y_u, beta=0.1, k_sigmoid=30)
model.train(X_train, y_train, loss_fn=custom_spi, epochs=50)


# 2. Inference and analysis
preds = model.predict(X_train[:5])

print("\n--- RÉSULTATS ---")
print(f"Bornes Inf : {preds.lower}")
print(f"Bornes Sup : {preds.upper}")
print(f"Largeur    : {preds.mpiw}")