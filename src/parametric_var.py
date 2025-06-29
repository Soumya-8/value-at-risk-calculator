import numpy as np
import pandas as pd
from scipy.stats import norm

def calculate_parametric_var(returns: pd.Series, confidence_level: float = 0.95, holding_period: int = 1):
    """
    Calculates the Parametric VaR using the Variance-Covariance method.

    Parameters:
    - returns: pd.Series of daily returns.
    - confidence_level: Confidence level (default is 0.95 for 95%).
    - holding_period: Holding period in days.

    Returns:
    - var_value: Value at Risk (VaR)
    """
    mean_return = returns.mean()
    std_dev = returns.std()

    # Z-score for the given confidence level
    z_score = norm.ppf(1 - confidence_level)

    # Parametric VaR formula
    var_value = -(mean_return + z_score * std_dev) * np.sqrt(holding_period)
    
    return var_value
