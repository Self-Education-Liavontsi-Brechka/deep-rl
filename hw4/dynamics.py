import tensorflow as tf
import numpy as np

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

            self.obs_prime_ph = tf.placeholder(tf.float32, [None, obs_dim], 'obs_prime_label')
            self.delta_ph = self.obs_prime_ph - self.obs_ph
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
        s = np.array(data[:, :, 0])
        a = np.array(data[:, :, 1])
        s_prime = np.array(data[:, :, 2])

        n = s.shape[0]

        from tqdm import tqdm

        print('Fitting dynamics...')
        for _ in tqdm(range(self.iterations)):
            permutation = np.random.permutation(n)
            s, a, s_prime = s[permutation], a[permutation], s_prime[permutation]

            batch_n = self.batch_size // n
            for i_batch in range(batch_n + 1):
                s_batch = s[i_batch * self.batch_size: min(n, (i_batch + 1) * self.batch_size), :]
                a_batch = a[i_batch * self.batch_size: min(n, (i_batch + 1) * self.batch_size), :]
                s_prime_batch = s_prime[i_batch * self.batch_size: min(n, (i_batch + 1) * self.batch_size), :]

                self.sess.run(self.optimization_result, {
                    self.obs_ph: s_batch,
                    self.action_ph: a_batch,
                    self.obs_prime_ph: s_prime_batch
                })

    def predict(self, states, actions):
        """
        Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model
        """

        """ YOUR CODE HERE """
        assert states.shape[0] == actions.shape[0]

        n = states.shape[0]
        batch_n = self.batch_size // n
        predictions = []

        for i_batch in range(batch_n + 1):
            s_batch = states[i_batch * self.batch_size: min(n, (i_batch + 1) * self.batch_size), :]
            a_batch = actions[i_batch * self.batch_size: min(n, (i_batch + 1) * self.batch_size), :]

            predictions.append(s_batch +
                               self.sess.run(self.output_raw, {self.obs_ph: s_batch, self.action_ph: a_batch}))

        return np.concatenate(predictions)
