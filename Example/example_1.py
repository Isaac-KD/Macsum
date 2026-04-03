import numpy as np 

import matplotlib.pyplot as plt
from Macsum import Macsum, SPILoss

X_train = np.random.rand(200, 10)
y_train = np.random.rand(200)
X_val = np.random.rand(200, 10)
y_val = np.random.rand(200)

# 1. Train with validation
model = Macsum()
model.train(
    X_train, y_train, 
    X_val=X_val, y_val=y_val, 
    loss_fn=SPILoss, 
    epochs=100
)

# 2. On récupère tout le dico d'un coup
h = model.history()

# 3. Bim, on plot la Loss et le PICP
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(h['train_loss'], label='Train Loss')
plt.plot(h['val_loss'], label='Val Loss')
plt.title('Convergence de la Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(h['train_picp'], label='Train PICP')
plt.plot(h['val_picp'], label='Val PICP')
plt.axhline(y=0.95, color='r', linestyle='--', label='Cible 95%')
plt.title('Évolution de la Couverture (PICP)')
plt.legend()

plt.show()