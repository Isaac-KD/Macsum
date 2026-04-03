import torch

def IMSELoss(y_true: torch.Tensor, y_low: torch.Tensor, y_up: torch.Tensor) -> torch.Tensor:
    """
    Interval Mean Squared Error (IMSE).
    Extended quadratic distance as defined in Eq. 7 of the research paper (Hmidy, Rico, Strauss).
    """
    return torch.mean((y_true - y_low)**2 + (y_true - y_up)**2)

def PinballLoss(y_true: torch.Tensor, y_low: torch.Tensor, y_up: torch.Tensor, beta: float = 0.05) -> torch.Tensor:
    """
    Quantile-based loss for interval estimation.
    beta: Target error rate (e.g., 0.05 for a 95% prediction interval).
    """
    # Define target quantiles (e.g., 0.025 and 0.975 for beta=0.05)
    tau_low = beta / 2.0
    tau_up = 1.0 - (beta / 2.0)
    
    # Error calculation
    err_low = y_true - y_low
    err_up = y_true - y_up
    
    # Pinball formula: max(tau * err, (tau - 1) * err)
    loss_low = torch.max(tau_low * err_low, (tau_low - 1.0) * err_low).mean()
    loss_up = torch.max(tau_up * err_up, (tau_up - 1.0) * err_up).mean()
    
    return loss_low + loss_up

def _sigmoide(x: torch.Tensor, k: float = 1.0) -> torch.Tensor: 
    """Internal utility function for smooth penalization (SPILoss)."""
    return torch.sigmoid(-k * x)

def SPILoss(y_true: torch.Tensor, y_low: torch.Tensor, y_up: torch.Tensor, beta: float = 0.05, k_sigmoid: float = 50.0) -> torch.Tensor:
    """
    Smooth Penalization Interval Loss (SPIL).
    A smoothed version of the Winkler score used for interval calibration.
    """
    # Penalize the width of the interval (MPIW)
    loss_spread = (y_up - y_low).mean()
    
    # Smooth penalization for points outside the bounds
    diff_low = y_low - y_true
    diff_up = y_true - y_up

    pen_low = (_sigmoide(diff_low, k_sigmoid) * diff_low).mean()
    pen_up = (_sigmoide(diff_up, k_sigmoid) * diff_up).mean()
    
    # Balance between width and coverage coverage (PICP)
    return loss_spread + (2.0 / beta) * (pen_low + pen_up)