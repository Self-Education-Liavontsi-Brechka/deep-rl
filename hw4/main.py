import numpy as np
import tensorflow as tf
import gym
from dynamics import NNDynamicsModel
from controllers import MPCcontroller, RandomController
from cost_functions import cheetah_cost_fn, trajectory_cost_fn, pendulum_cost_fn
import time
import logz
import os
import inspect
import types


def sample(env,
           controller,
           num_paths=10,
           horizon=1000,
           render=False,
           verbose=False):
    """
        Write a sampler function which takes in an environment, a controller (either random or the MPC controller), 
        and returns rollouts by running on the env. 
        Each path can have elements for observations, next_observations, rewards, returns, actions, etc.
    """
    paths = []
    """ YOUR CODE HERE """
    obs_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]

    if verbose:
        print('Sampling data...')
    for i_path in range(num_paths):
        if verbose:
            print('Trajectory {}/{}'.format(i_path + 1, num_paths))

        path = {
            'observations': np.zeros((horizon, obs_dim)),
            'actions': np.zeros((horizon, a_dim)),
            'next_observations': np.zeros((horizon, obs_dim)),
            'deltas': np.zeros((horizon, obs_dim)),
            'rewards': np.zeros(horizon)
        }

        s = env.reset()

        for step in range(horizon):
            a = controller.get_action(s)
            s_prime, reward, done, _ = env.step(a)
            path['observations'][step] = s
            path['actions'][step] = a
            path['next_observations'][step] = s_prime
            path['deltas'][step] = s_prime - s
            path['rewards'][step] = reward

            if render:
                env.render()
            if done:
                break
            if verbose and (step + 1) % 100 == 0:
                print('\rStep {}/{}'.format(step + 1, horizon), end='')

            s = s_prime

        if verbose:
            print()

        paths.append(path)

    return np.array(paths)


# Utility to compute cost a path for a given cost function
def path_cost(cost_fn, path):
    return trajectory_cost_fn(cost_fn, path['observations'], path['actions'], path['next_observations'])


def compute_normalization(data):
    """
    Write a function to take in a dataset and compute the means, and stds.
    Return 6 elements: mean of s_t, std of s_t, mean of (s_t+1 - s_t), std of (s_t+1 - s_t), mean of actions, std of actions
    """

    """ YOUR CODE HERE """
    all_obs = []
    all_actions = []
    all_deltas = []

    for i_path in data:
        all_obs.append(i_path['observations'])
        all_actions.append(i_path['actions'])
        all_deltas.append(i_path['deltas'])

    all_obs = np.concatenate(all_obs)
    all_actions = np.concatenate(all_actions)
    all_deltas = np.concatenate(all_deltas)

    return np.mean(all_obs), np.std(all_obs), np.mean(all_deltas), np.std(all_deltas), \
           np.mean(all_actions), np.std(all_actions)


def plot_comparison(env, dyn_model):
    """
    Write a function to generate plots comparing the behavior of the model predictions for each element of the state to the actual ground truth, using randomly sampled actions. 
    """
    """ YOUR CODE HERE """
    pass


def get_dynamics_test_loss(env, dyn_model):
    return dyn_model.get_loss_on_data(sample(env, RandomController(env)))


def train(env,
          cost_fn,
          logdir=None,
          render=False,
          learning_rate=1e-3,
          onpol_iters=10,
          dynamics_iters=60,
          batch_size=512,
          num_paths_random=10,
          num_paths_onpol=10,
          num_simulated_paths=10,
          env_horizon=1000,
          mpc_horizon=15,
          n_layers=2,
          size=500,
          activation=tf.nn.relu,
          output_activation=None
          ):
    """

    Arguments:

    onpol_iters                 Number of iterations of onpolicy aggregation for the loop to run. 

    dynamics_iters              Number of iterations of training for the dynamics model
    |_                          which happen per iteration of the aggregation loop.

    batch_size                  Batch size for dynamics training.

    num_paths_random            Number of paths/trajectories/rollouts generated 
    |                           by a random agent. We use these to train our 
    |_                          initial dynamics model.
    
    num_paths_onpol             Number of paths to collect at each iteration of
    |_                          aggregation, using the Model Predictive Control policy.

    num_simulated_paths         How many fictitious rollouts the MPC policy
    |                           should generate each time it is asked for an
    |_                          action.

    env_horizon                 Number of timesteps in each path.

    mpc_horizon                 The MPC policy generates actions by imagining 
    |                           fictitious rollouts, and picking the first action
    |                           of the best fictitious rollout. This argument is
    |                           how many timesteps should be in each fictitious
    |_                          rollout.

    n_layers/size/activations   Neural network architecture arguments. 

    """

    logz.configure_output_dir(logdir)

    # Log experimental parameters
    args = inspect.getargspec(train)[0]
    locals_ = locals()
    params = {k: locals_[k] if (k in locals_ and
                                not isinstance(locals_[k], types.FunctionType) and
                                not isinstance(locals_[k], gym.Env)) else None for k in args}
    logz.save_params(params)

    # ========================================================
    # 
    # First, we need a lot of data generated by a random
    # agent, with which we'll begin to train our dynamics
    # model.

    random_controller = RandomController(env)

    """ YOUR CODE HERE """
    data = sample(env, random_controller, num_paths_random, env_horizon)

    # ========================================================
    # 
    # The random data will be used to get statistics (mean
    # and std) for the observations, actions, and deltas
    # (where deltas are o_{t+1} - o_t). These will be used
    # for normalizing inputs and denormalizing outputs
    # from the dynamics network. 
    #
    """ YOUR CODE HERE """
    normalization = compute_normalization(data)

    # ========================================================
    # 
    # Build dynamics model and MPC controllers.
    # 
    sess = tf.Session()

    dyn_model = NNDynamicsModel(env=env,
                                n_layers=n_layers,
                                size=size,
                                activation=activation,
                                output_activation=output_activation,
                                normalization=normalization,
                                batch_size=batch_size,
                                iterations=dynamics_iters,
                                learning_rate=learning_rate,
                                sess=sess)

    mpc_controller = MPCcontroller(env=env,
                                   dyn_model=dyn_model,
                                   horizon=mpc_horizon,
                                   cost_fn=cost_fn,
                                   num_simulated_paths=num_simulated_paths)

    # ========================================================
    # 
    # Tensorflow session building.
    # 
    sess.__enter__()
    tf.global_variables_initializer().run()

    # ========================================================
    # 
    # Take multiple iterations of onpolicy aggregation at each iteration refitting the dynamics model to current dataset and then taking onpolicy samples and aggregating to the dataset. 
    # Note: You don't need to use a mixing ratio in this assignment for new and old data as described in https://arxiv.org/abs/1708.02596
    # 
    for itr in range(onpol_iters):
        print('Iteration {}/{}'.format(itr + 1, onpol_iters))
        """ YOUR CODE HERE """

        dyn_train_loss = dyn_model.fit(data)
        on_policy_data = sample(env, mpc_controller, num_paths_onpol, env_horizon, render, verbose=True)
        data = np.concatenate((data, on_policy_data))

        costs = [path_cost(cost_fn, i_path) for i_path in on_policy_data]
        returns = [np.sum(i_path['rewards']) for i_path in on_policy_data]
        dyn_test_loss = get_dynamics_test_loss(env, dyn_model)

        # LOGGING
        # Statistics for performance of MPC policy using
        # our learned dynamics model
        logz.log_tabular('Iteration', itr)
        # In terms of cost function which your MPC controller uses to plan
        logz.log_tabular('AverageCost', np.mean(costs))
        logz.log_tabular('StdCost', np.std(costs))
        logz.log_tabular('MinimumCost', np.min(costs))
        logz.log_tabular('MaximumCost', np.max(costs))
        # In terms of true environment reward of your rolled out trajectory using the MPC controller
        logz.log_tabular('AverageReturn', np.mean(returns))
        logz.log_tabular('StdReturn', np.std(returns))
        logz.log_tabular('MinimumReturn', np.min(returns))
        logz.log_tabular('MaximumReturn', np.max(returns))
        # Dynamics network
        logz.log_tabular('AverageTrainLoss', np.mean(dyn_train_loss))
        logz.log_tabular('StdTrainLoss', np.std(dyn_train_loss))
        logz.log_tabular('AverageTestLoss', np.mean(dyn_test_loss))
        logz.log_tabular('StdTestLoss', np.std(dyn_test_loss))

        logz.dump_tabular()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v1')
    # Experiment meta-params
    parser.add_argument('--exp_name', type=str, default='mb_mpc')
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--render', action='store_true')
    # Training args
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--onpol_iters', '-n', type=int, default=1)
    parser.add_argument('--dyn_iters', '-nd', type=int, default=60)
    parser.add_argument('--batch_size', '-b', type=int, default=512)
    # Data collection
    parser.add_argument('--random_paths', '-r', type=int, default=10)
    parser.add_argument('--onpol_paths', '-d', type=int, default=10)
    parser.add_argument('--simulated_paths', '-sp', type=int, default=10)
    parser.add_argument('--ep_len', '-ep', type=int, default=1000)
    # Neural network architecture args
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=500)
    # MPC Controller
    parser.add_argument('--mpc_horizon', '-m', type=int, default=15)
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # Make data directory if it does not already exist
    if not (os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    # Make env
    if args.env_name == "HalfCheetah-v1":
        # to solve expiration error issue when it's not available
        from cheetah_env import HalfCheetahEnvNew
        env = HalfCheetahEnvNew()
        cost_fn = cheetah_cost_fn
    elif args.env_name == 'Pendulum-v0':
        env = gym.make(args.env_name)
        env._max_episode_steps = args.ep_len
        cost_fn = pendulum_cost_fn

    train(env=env,
          cost_fn=cost_fn,
          logdir=logdir,
          render=args.render,
          learning_rate=args.learning_rate,
          onpol_iters=args.onpol_iters,
          dynamics_iters=args.dyn_iters,
          batch_size=args.batch_size,
          num_paths_random=args.random_paths,
          num_paths_onpol=args.onpol_paths,
          num_simulated_paths=args.simulated_paths,
          env_horizon=args.ep_len,
          mpc_horizon=args.mpc_horizon,
          n_layers=args.n_layers,
          size=args.size,
          activation=tf.nn.relu,
          output_activation=None,
          )


if __name__ == "__main__":
    main()
