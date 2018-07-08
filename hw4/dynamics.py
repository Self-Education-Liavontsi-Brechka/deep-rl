import tensorflow as tf
import numpy as np
from tqdm import tqdm

eps = np.finfo(np.float32).eps


# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder,
              output_size,
              scope,
              n_layers=2,
              size=500,
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out


class NNDynamicsModel(object):
    def __init__(self,
                 env,
                 n_layers,
                 size,
                 activation,
                 output_activation,
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """
        self.scope = "dynamics"
        self.batch_size = batch_size
        self.iterations = iterations
        self.sess = sess

        mean_obs, std_obs, mean_deltas, std_deltas, mean_action, std_action = normalization
        obs_dim = env.observation_space.shape[0]
        a_dim = env.action_space.shape[0]

        with tf.variable_scope(self.scope):
            self.obs_ph = tf.placeholder(tf.float32, [None, obs_dim], 'state_input')
            self.obs_ph = (self.obs_ph - np.expand_dims(mean_obs, 0)) / (np.expand_dims(std_obs, 0) + eps)
            self.action_ph = tf.placeholder(tf.float32, [None, a_dim], 'action_input')
            self.action_ph = (self.action_ph - np.expand_dims(mean_action, 0)) / (np.expand_dims(std_action, 0) + eps)
            self.input = tf.concat([self.obs_ph, self.action_ph], axis=-1, name='input')

            self.delta_ph = tf.placeholder(tf.float32, [None, obs_dim], 'delta_label')
            self.delta_ph = (self.delta_ph - np.expand_dims(mean_deltas, 0)) / (np.expand_dims(std_deltas, 0) + eps)

        self.output = build_mlp(self.input, obs_dim, self.scope, n_layers, size, activation, output_activation)
        self.output_raw = np.expand_dims(mean_deltas, 0) + np.expand_dims(std_deltas, 0) * self.output

        with tf.variable_scope(self.scope):
            self.loss = tf.losses.mean_squared_error(self.delta_ph, self.output)
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
            self.optimization_result = self.optimizer.minimize(self.loss)

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """

        """YOUR CODE HERE """
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

        n = len(all_obs)
        losses = []

        print('Fitting dynamics...')
        for _ in tqdm(range(self.iterations)):
            permutation = np.random.permutation(n)
            all_obs, all_actions, all_deltas = all_obs[permutation], all_actions[permutation], all_deltas[permutation]

            batch_n = n // self.batch_size
            for i_batch in range(batch_n + 1):
                s_batch = all_obs[i_batch * self.batch_size: min(n, (i_batch + 1) * self.batch_size), :]
                a_batch = all_actions[i_batch * self.batch_size: min(n, (i_batch + 1) * self.batch_size), :]
                deltas_batch = all_deltas[i_batch * self.batch_size: min(n, (i_batch + 1) * self.batch_size), :]

                _, loss = self.sess.run([self.optimization_result, self.loss], {
                    self.obs_ph: s_batch,
                    self.action_ph: a_batch,
                    self.delta_ph: deltas_batch
                })

                losses.append(loss)

        return losses

    def predict(self, states, actions):
        """
        Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model
        """

        """ YOUR CODE HERE """
        assert states.shape[0] == actions.shape[0]

        n = states.shape[0]
        batch_n = n // self.batch_size
        predictions = []

        for i_batch in range(batch_n + 1):
            s_batch = states[i_batch * self.batch_size: min(n, (i_batch + 1) * self.batch_size), :]
            a_batch = actions[i_batch * self.batch_size: min(n, (i_batch + 1) * self.batch_size), :]

            predictions.append(s_batch +
                               self.sess.run(self.output_raw, {self.obs_ph: s_batch, self.action_ph: a_batch}))

        return np.concatenate(predictions)

    def get_loss_on_data(self, data):
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

        n = len(all_obs)
        losses = []
        for _ in tqdm(range(self.iterations)):
            permutation = np.random.permutation(n)
            all_obs, all_actions, all_deltas = all_obs[permutation], all_actions[permutation], all_deltas[permutation]

            batch_n = n // self.batch_size
            for i_batch in range(batch_n + 1):
                s_batch = all_obs[i_batch * self.batch_size: min(n, (i_batch + 1) * self.batch_size), :]
                a_batch = all_actions[i_batch * self.batch_size: min(n, (i_batch + 1) * self.batch_size), :]
                deltas_batch = all_deltas[i_batch * self.batch_size: min(n, (i_batch + 1) * self.batch_size), :]

                losses.append(self.sess.run(self.loss, {
                    self.obs_ph: s_batch,
                    self.action_ph: a_batch,
                    self.delta_ph: deltas_batch
                }))

        return losses
