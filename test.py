import numpy as np
import time

from macsum import Macsum
from losses import IMSELoss

GREEN = "\033[1;32m"
RED = "\033[1;31m"
RESET = "\033[0m"
INFO = "\033[1;34m"


def run_sanity_check():
    print(f"{INFO}[INFO] Starting Macsum library integration tests...{RESET}")

    # 1. Data Generation
    print(
        f"{INFO}[INFO] Generating synthetic data (100 samples, 5 features)...{RESET}")
    X_train = np.random.rand(100, 5)
    y_train = np.random.rand(100)

    # 2. Model Instantiation
    print(f"{INFO}[INFO] Instantiating the model...{RESET}")
    try:
        model = Macsum()
        print(f"{GREEN}[SUCCESS] Model instantiated successfully.{RESET}")
    except Exception as e:
        print(f"{RED}[ERROR] Instantiation failed: {e}{RESET}")
        return

    # 3. Training Run
    print(f"{INFO}[INFO] Launching validation training (3 epochs)...{RESET}")
    try:
        start_time = time.time()
        model.train(
            X_train,
            y_train,
            loss_fn=IMSELoss,
            epochs=3,
            batch_size=32)
        print(
            f"{GREEN}[SUCCESS] Training completed in "
            f"{time.time() - start_time:.2f} seconds.{RESET}"
        )
    except Exception as e:
        print(f"{RED}[ERROR] Training failed: {e}{RESET}")
        return

    # 4. History Verification
    print(f"{INFO}[INFO] Verifying training history structure...{RESET}")
    history = model.history()
    if len(history.get('train_loss', [])) == 3:
        print(
            f"{GREEN}[SUCCESS] History correctly recorded "
            f"metrics for all 3 epochs.{RESET}"
        )
    else:
        print(
            f"{RED}[ERROR] History does not contain the "
            f"expected number of epochs.{RESET}"
        )

    # 5. Inference and Dimension Validation
    print(f"{INFO}[INFO] Testing inference...{RESET}")
    try:
        preds = model.predict(X_train[:5])

        assert preds.lower.shape == (5,), "Incorrect lower bound dimension."
        assert preds.upper.shape == (5,), "Incorrect upper bound dimension."

        print(
            f"{GREEN}[SUCCESS] Predictions generated and "
            f"validated successfully.{RESET}"
        )
        print("\nResults Detail for the first sample:")
        print(f"  - Lower Bound : {preds.lower[0]:.4f}")
        print(f"  - Center      : {preds.center[0]:.4f}")
        print(f"  - Upper Bound : {preds.upper[0]:.4f}")
        print(f"  - Width       : {preds.mpiw[0]:.4f}")

    except AssertionError as ae:
        print(f"{RED}[ERROR] Assertion error during inference: {ae}{RESET}")
        return
    except Exception as e:
        print(f"{RED}[ERROR] Unexpected failure during inference: {e}{RESET}")
        return

    print(
        f"{GREEN}\n[SUCCESS] All modules are communicating correctly. "
        f"Codebase validated.{RESET}"
    )


if __name__ == "__main__":
    run_sanity_check()
