"""
Probability Distributions - Week 2
====================================
Key distributions used in RL:
- Gaussian: continuous action policies
- Categorical: discrete action selection
"""

import numpy as np


def sample_gaussian(mu, sigma, n=1):
    """Sample from N(mu, sigma^2)."""
    return np.random.normal(mu, sigma, n)


def sample_categorical(probs, n=1):
    """Sample discrete action from probability distribution."""
    return np.random.choice(len(probs), size=n, p=probs)


if __name__ == "__main__":
    # Gaussian policy (continuous actions)
    actions = sample_gaussian(mu=0.0, sigma=1.0, n=5)
    print("Gaussian samples:", actions.round(3))

    # Categorical policy (discrete actions)
    probs = [0.1, 0.3, 0.4, 0.2]
    actions = sample_categorical(probs, n=5)
    print("Categorical samples:", actions)
