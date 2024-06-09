#### MERTON #####

import numpy as np
import matplotlib.pyplot as plt

##### Model parameters #####


T = 1.0  # Time horizon (use np.inf for infinite T)
dt = 1/250  # Time step
N = int(T / dt) if T < np.inf else int(10 / dt)  # Number of time steps, use 10 for long simulations
r = 0.03  # Risk-free rate
sigma = 0.2  # Volatility of the risky asset
mu = 0.1  # Expected return of the risky asset
gamma = 2  # CRRA risk aversion coefficient
rho = 0.04  # Subjective discount rate
epsilon = 0.5  # Preference between wealth utility and consumption
n_paths = 100  # Number of simulated paths
W0 = 1.0  # Initial wealth









# Calculation of ν
nu = (rho - (1 - gamma) * ((mu - r) ** 2 / (2 * sigma ** 2 * gamma) + r)) / gamma

def optimal_consumption(W, t, T, nu, epsilon):
    if T == np.inf:
        return nu * W
    elif nu != 0:
        return nu * W / (1 + (nu * epsilon - 1) * np.exp(-nu * (T - t)))
    else:
        return W / (T - t + epsilon)

# Simulation of risky asset paths (Black-Scholes model)
np.random.seed(42)  # For reproducibility
S = np.zeros((n_paths, N + 1))
S[:, 0] = 1.0  # Initial price of the risky asset

for t in range(1, N + 1):
    Z = np.random.standard_normal(n_paths)
    S[:, t] = S[:, t-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

# Calculation of optimal risky asset fraction (constant over time)
pi_constant = (mu - r) / (sigma ** 2 * gamma)

# Calculation of optimal wealth and optimal consumption
W = np.zeros((n_paths, N + 1))
C = np.zeros((n_paths, N))
pi = np.full((n_paths, N), pi_constant)
W[:, 0] = W0

for t in range(N):
    C[:, t] = optimal_consumption(W[:, t], t * dt, T, nu, epsilon)
    dW = (r * W[:, t] + pi_constant * (mu - r) - C[:, t]) * dt + pi_constant * sigma * np.sqrt(dt) * np.random.standard_normal(n_paths)
    W[:, t+1] = W[:, t] + dW

# Plot of optimal wealth
plt.figure(figsize=(12, 6))
for i in range(n_paths):
    plt.plot(np.linspace(0, T if T < np.inf else 10, N + 1), W[i, :], color='blue', alpha=0.1)
plt.plot(np.linspace(0, T if T < np.inf else 10, N + 1), np.mean(W, axis=0), color='red', label='Average wealth')
plt.xlabel('Time')
plt.ylabel('Wealth')
plt.title('Optimal Wealth (100 paths)')
plt.legend()
plt.show()

# Plot of optimal consumption as % of wealth and share of risky asset
plt.figure(figsize=(12, 6))
plt.plot(np.linspace(0, T if T < np.inf else 10, N), np.mean(C / W[:, :-1], axis=0), label='Optimal Consumption (%)')
plt.plot(np.linspace(0, T if T < np.inf else 10, N), np.mean(pi, axis=0), label='Share of Risky Asset')
plt.xlabel('Time')
plt.ylabel('Percentage')
plt.title('Optimal Consumption and Share of Risky Asset')
plt.legend()
plt.show()
