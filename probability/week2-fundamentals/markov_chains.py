"""
Markov Chains - Week 2: Probability Foundations
================================================
Simulates a simple Markov chain weather model.
Key concept: future state depends ONLY on current state (Markov property).
This is the foundation of MDPs in reinforcement learning.
"""

import numpy as np


def simulate_markov_chain(transition_matrix, initial_state, n_steps):
    """
    Simulate a Markov chain trajectory.

    Args:
        transition_matrix: Row-stochastic matrix T[i,j] = P(j|i)
        initial_state: Starting state index
        n_steps: Number of steps to simulate

    Returns:
        list: Sequence of visited states
    """
    n_states = len(transition_matrix)
    trajectory = [initial_state]
    current = initial_state

    for _ in range(n_steps):
        probs = transition_matrix[current]
        current = np.random.choice(n_states, p=probs)
        trajectory.append(current)

    return trajectory


if __name__ == "__main__":
    # Weather model: 0=Sunny, 1=Cloudy, 2=Rainy
    T = np.array([
        [0.7, 0.2, 0.1],  # From Sunny
        [0.3, 0.5, 0.2],  # From Cloudy
        [0.2, 0.3, 0.5],  # From Rainy
    ])

    state_names = ["Sunny", "Cloudy", "Rainy"]
    traj = simulate_markov_chain(T, initial_state=0, n_steps=20)

    print("Weather trajectory:")
    print(" -> ".join(state_names[s] for s in traj))
