import torch
import torch.nn as nn
import numpy as np


class MacsumCore(nn.Module):
    """
    Core implementation of the Macsum aggregation operator.
    Outputs the lower and upper bounds of a convex set of linear functions.
    """

    def __init__(self, N: int, phi_init=None, device="cpu"):
        super().__init__()
        self.N = N

        # Initialize the kernel phi (the unique parameter vector of the model)
        if phi_init is None:
            phi_init_tensor = torch.randn(
                N, device=device, dtype=torch.float32)
        elif isinstance(phi_init, np.ndarray):
            phi_init_tensor = torch.tensor(
                phi_init, device=device, dtype=torch.float32)
        else:
            phi_init_tensor = phi_init.clone().detach().to(
                device=device, dtype=torch.float32)

        self._phi = nn.Parameter(phi_init_tensor)

    def forward(self, x: torch.Tensor):
        original_ndim = x.ndim
        if original_ndim == 1:
            x = x.unsqueeze(0)

        batch_size, n_features = x.shape
        if n_features != self.N:
            raise ValueError(
                f"Input feature dimension must be {self.N}, got {n_features}")

        # Calculate permutations (sorting does not require gradients)
        with torch.no_grad():
            perm_decreasing = torch.argsort(self._phi, descending=True).long()

        # Split phi into positive and negative parts while keeping autograd
        # connection
        phi_plus = torch.clamp_min(self._phi, 0)
        phi_minus = torch.clamp_max(self._phi, 0)

        # Apply sorting to inputs and parameters
        x_permuted = x[:, perm_decreasing]
        phi_plus_sorted = phi_plus[perm_decreasing]
        phi_minus_sorted = phi_minus[perm_decreasing]

        # Cumulative max and min for the Choquet integral computation
        acc_max = torch.cummax(x_permuted, dim=1).values
        acc_min = torch.cummin(x_permuted, dim=1).values

        padding = torch.zeros(batch_size, 1, dtype=x.dtype, device=x.device)
        padded_max = torch.cat((padding, acc_max), dim=1)
        padded_min = torch.cat((padding, acc_min), dim=1)

        diff_max = padded_max[:, 1:] - padded_max[:, :-1]
        diff_min = padded_min[:, 1:] - padded_min[:, :-1]

        # Upper and lower bound calculations via matrix multiplication
        y_upper = torch.matmul(diff_max, phi_plus_sorted) + \
            torch.matmul(diff_min, phi_minus_sorted)
        y_lower = torch.matmul(diff_min, phi_plus_sorted) + \
            torch.matmul(diff_max, phi_minus_sorted)

        if original_ndim == 1:
            return y_lower.squeeze(0), y_upper.squeeze(0)
        return y_lower, y_upper
