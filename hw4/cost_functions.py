import numpy as np


# ========================================================
# 
# Environment-specific cost functions:
#

def cheetah_cost_fn(state, action, next_state):
    if len(state.shape) > 1:
        heading_penalty_factor = 10
        scores = np.zeros((state.shape[0],))

        # dont move front shin back so far that you tilt forward
        front_leg = state[:, 5]
        my_range = 0.2
        scores[front_leg >= my_range] += heading_penalty_factor

        front_shin = state[:, 6]
        my_range = 0
        scores[front_shin >= my_range] += heading_penalty_factor

        front_foot = state[:, 7]
        my_range = 0
        scores[front_foot >= my_range] += heading_penalty_factor

        scores -= (next_state[:, 17] - state[:, 17]) / 0.01  # + 0.1 * (np.sum(action**2, axis=1))
        return scores

    heading_penalty_factor = 10
    score = 0

    # dont move front shin back so far that you tilt forward
    front_leg = state[5]
    my_range = 0.2
    if front_leg >= my_range:
        score += heading_penalty_factor

    front_shin = state[6]
    my_range = 0
    if front_shin >= my_range:
        score += heading_penalty_factor

    front_foot = state[7]
    my_range = 0
    if front_foot >= my_range:
        score += heading_penalty_factor

    score -= (next_state[17] - state[17]) / 0.01  # + 0.1 * (np.sum(action**2))
    return score


def pendulum_cost_fn(state, action, next_state):
    if len(state.shape) > 1:
        assert state.shape[0] == action.shape[0]

        cos_th = state[:, 0:1]
        thdot = state[:, 2:3]
        th = np.arccos(cos_th)

        max_torque = 2.

        u = np.clip(action, -max_torque, max_torque)
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
        costs = np.squeeze(np.nan_to_num(costs))
        # costs = np.sum(costs)

        return costs

    cos_th = state[0]
    thdot = state[2]
    th = np.arccos(cos_th)

    max_torque = 2.

    u = np.clip(action, -max_torque, max_torque)[0]
    costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

    return np.nan_to_num(costs)


# ========================================================
# 
# Cost function for a whole trajectory:
#

def trajectory_cost_fn(cost_fn, states, actions, next_states):
    trajectory_cost = 0
    for i in range(len(actions)):
        trajectory_cost += cost_fn(states[i], actions[i], next_states[i])
    return trajectory_cost


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
