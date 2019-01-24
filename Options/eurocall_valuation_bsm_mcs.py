import numpy as np
import math

from statistics import mean

# Valuation of a European call option by Monte Carlo simulation
# Considering a Black-Scholes-Merton setup in which the option's underlying risk factor follows a geometric Brownian motion

# Parameters values
S0 = 100.   # initial index level
K = 105.    # strike price
T = 1.0     # time-to-maturity
r = 0.5     # riskless short rate
sigma = 0.2 # volatility

I = 100000 # nb of simulations

# Valuation algorithm
z = np.random.standard_normal(I)  # pseudo-random nbs
# Index values at maturity
ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * z)
hT = np.maximum(ST - K, 0)  # payoff at maturity
C0 = math.exp(-r * T) * np.mean(hT) # Monte Carlo estimator

# Result output
print("Value of the European call option %5.3f." % C0)
