"""
Value Iteration - Week 4: RL Foundations
=========================================
Solves a simple GridWorld MDP using the Bellman optimality equation.
This is the theoretical core of ALL RL algorithms including DreamerV3.

Bellman Optimality: V*(s) = max_a [R(s,a) + gamma * sum_s' P(s'|s,a) * V*(s')]
"""

import numpy as np


def value_iteration(n_states, n_actions, transitions, rewards, gamma=0.99, theta=1e-6):
    """
    Find optimal value function V* using value iteration.

    Args:
        n_states: Number of states
        n_actions: Number of actions
        transitions: T[s,a,s'] = P(s'|s,a)
        rewards: R[s,a] = expected reward
        gamma: Discount factor
        theta: Convergence threshold

    Returns:
        V: Optimal value function
        policy: Optimal greedy policy
    """
    V = np.zeros(n_states)

    for iteration in range(10000):
        delta = 0
        for s in range(n_states):
            q_values = np.array([
                rewards[s, a] + gamma * np.sum(transitions[s, a] * V)
                for a in range(n_actions)
            ])
            v_new = np.max(q_values)
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new

        if delta < theta:
            print(f"Converged in {iteration + 1} iterations")
            break

    # Extract greedy policy
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        q_values = np.array([
            rewards[s, a] + gamma * np.sum(transitions[s, a] * V)
            for a in range(n_actions)
        ])
        policy[s] = np.argmax(q_values)

    return V, policy


if __name__ == "__main__":
    print("Simple 4-state MDP")
    print("States: S0 -> S1 -> S2 -> S3(goal)")

    n_states, n_actions = 4, 2  # actions: 0=stay, 1=forward
    T = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions))

    # Action 0: stay in place
    for s in range(n_states):
        T[s, 0, s] = 1.0
        R[s, 0] = -0.1  # small penalty for staying

    # Action 1: move forward
    for s in range(n_states - 1):
        T[s, 1, s + 1] = 1.0
        R[s, 1] = 1.0 if s + 1 == n_states - 1 else -0.1
    T[n_states - 1, 1, n_states - 1] = 1.0  # goal absorbs

    V, policy = value_iteration(n_states, n_actions, T, R)
    action_names = ["stay", "forward"]

    print("\nOptimal Values:", V.round(3))
    print("Optimal Policy:", [action_names[a] for a in policy])
