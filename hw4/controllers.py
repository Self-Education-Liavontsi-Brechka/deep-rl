import numpy as np
from cost_functions import trajectory_cost_fn
import time


class Controller():
    def __init__(self):
        pass

    # Get the appropriate action(s) for this state(s)
    def get_action(self, state):
        pass


class RandomController(Controller):
    def __init__(self, env):
        """ YOUR CODE HERE """
        self.env = env

    def get_action(self, state):
        """ YOUR CODE HERE """
        """ Your code should randomly sample an action uniformly from the action space """
        return self.env.action_space.sample()


class MPCcontroller(Controller):
    """ Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """

    def __init__(self,
                 env,
                 dyn_model,
                 horizon=5,
                 cost_fn=None,
                 num_simulated_paths=10,
                 ):
        self.env = env
        self.dyn_model = dyn_model
        self.horizon = horizon
        self.cost_fn = cost_fn
        self.num_simulated_paths = num_simulated_paths

    def get_action(self, state):
        """ YOUR CODE HERE """
        """ Note: be careful to batch your simulations through the model for speed """
        obs_dim = self.env.observation_space.shape[0]
        a_dim = self.env.action_space.shape[0]

        actions = np.zeros((self.horizon, self.num_simulated_paths, a_dim))
        states = np.zeros((self.horizon, self.num_simulated_paths, obs_dim))
        prime_states = np.zeros((self.horizon, self.num_simulated_paths, obs_dim))

        states[0, :, :] = np.full((self.num_simulated_paths, obs_dim), state)

        for step in range(self.horizon):
            states_t = states[step, :, :]
            a_t = np.array([self.env.action_space.sample() for _ in range(self.num_simulated_paths)])
            ps_t = self.dyn_model.predict(states_t, a_t)

            actions[step, :, :] = a_t
            prime_states[step, :, :] = ps_t
            if step < self.horizon - 1:
                states[step + 1, :, :] = ps_t

        costs = trajectory_cost_fn(self.cost_fn, states, actions, prime_states)

        best_traj_index = np.argmin(costs, axis=0)

        return actions[0][best_traj_index]
