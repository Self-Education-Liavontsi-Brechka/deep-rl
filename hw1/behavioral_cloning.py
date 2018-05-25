import tensorflow as tf
import numpy as np
import itertools
import gym
import os
import time
import pickle


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--expert_data_file', type=str)
    parser.add_argument('--num_expert_episodes', type=int, default=20)
    parser.add_argument('--num_test_episodes', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--summary_dir', type=str)
    parser.add_argument('--render', action='store_true')
    return parser.parse_args()


def load_expert_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)

    return data


def load_expert_policy(filename):
    import load_policy

    assert filename and filename != ''
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(filename)
    print('loaded and built')

    return policy_fn


def generate_expert_data(env, expert_policy, num_episodes, max_timesteps=None):
    print('Generating expert data:')

    max_timesteps = max_timesteps or env.spec.timestep_limit
    transitions = []
    rewards = []

    for i_episode in range(num_episodes):
        total_reward = 0.0

        o = env.reset()
        for t in itertools.count():
            expert_policy_action = np.squeeze(expert_policy(np.expand_dims(o, 0)))
            o_prime, reward, done, _ = env.step(expert_policy_action)

            transitions.append([o, expert_policy_action])

            total_reward += reward
            o = o_prime

            if t % max(1, max_timesteps // 10) == 0:
                print('\rEpisode {}/{}: step {}'.format(i_episode + 1, num_episodes, t), end='')

            if done or t >= max_timesteps:
                break

        rewards.append(total_reward)

    rewards_mean = np.mean(rewards)
    rewards_deviation = np.std(rewards)
    print('\nExpert Rewards Mean: {}, Expert Rewards deviation: {}'.format(rewards_mean, rewards_deviation))

    expert_data_file_path = './experts/data/{}'.format(env.spec.id)
    expert_data_filename = '{}.pkl.data'.format(time.strftime("%Y-%m-%d-%H-%M-%S"))
    if not os.path.exists(expert_data_file_path):
        os.makedirs(expert_data_file_path)

    with open(expert_data_file_path + '/' + expert_data_filename, 'xb') as file:
        pickle.dump(transitions, file, pickle.HIGHEST_PROTOCOL)
        print('Expert data is written to {}'.format(expert_data_file_path + '/' + expert_data_filename))

    return transitions


def build_target_policy(env, starting_learning_rate=0.001):
    global_step = tf.Variable(0, name='global_step', trainable=False)

    learning_rate_decay_steps = 10000
    learning_rate_decay_rate = 0.5

    layer1_units = 512
    layer2_units = 256
    layer3_units = 128
    layer4_units = 128

    # ----------------------- PREDICT PART ------------------------
    with tf.name_scope('prediction'):
        observation_input = tf.placeholder(tf.float32, [None, env.observation_space.shape[0]], 'observation_input')

        # o_mean = np.mean(obs, axis=0)
        # o_std = np.std(obs, axis=0) + 10e-16
        # obs = (obs - np.expand_dims(o_mean, 0)) / np.expand_dims(o_std, 0)

        current_activations = tf.layers.dense(observation_input, layer1_units, tf.nn.tanh, name='hidden_layer_1')
        current_activations = tf.layers.dense(current_activations, layer2_units, tf.nn.tanh, name='hidden_layer_2')
        current_activations = tf.layers.dense(current_activations, layer3_units, tf.nn.tanh, name='hidden_layer_3')
        current_activations = tf.layers.dense(current_activations, layer4_units, tf.nn.tanh, name='hidden_layer_4')
        action_output = tf.layers.dense(current_activations, env.action_space.shape[0], name='action_output')

    # ----------------------- TRAIN PART --------------------------
    with tf.name_scope('training'):
        label_input = tf.placeholder(tf.float32, [None, env.action_space.shape[0]], 'label_input')

        learning_rate = tf.train.exponential_decay(starting_learning_rate, global_step, learning_rate_decay_steps,
                                                   learning_rate_decay_rate, False, 'learning_rate_decay')

        loss = tf.losses.mean_squared_error(label_input, action_output)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_optimization = optimizer.minimize(loss, global_step)

        summaries = tf.summary.merge([
            tf.summary.scalar('learning_rate', learning_rate),
        ])

    with tf.name_scope('test'):
        test_label_input = tf.placeholder(tf.float32, [None, env.action_space.shape[0]], 'test_label_input')

        test_loss = tf.losses.mean_squared_error(test_label_input, action_output)

    def predict_fn(session, observation, test_label=None):
        if test_label is None:
            return session.run(action_output, {observation_input: observation})
        else:
            return session.run([action_output, test_loss],
                               {observation_input: observation, test_label_input: test_label})

    def update_fn(session, observation, label):
        summary, _, loss_result, global_step = session.run(
            [summaries, train_optimization, loss, tf.train.get_global_step()],
            {
                observation_input: observation,
                label_input: label
            }
        )

        return loss_result, summary, global_step

    return predict_fn, update_fn


def create_summary_writer(session, env, summary_dir=None):
    if summary_dir:
        summary_dir = os.path.abspath("./summaries/{}/{}".format(env.spec.id, time.strftime("%Y-%m-%d-%H-%M-%S") + '-' +
                                                                 summary_dir))
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)

    return tf.summary.FileWriter(summary_dir, session.graph) if summary_dir else None


def clone_behaviour(session, target_policy, target_policy_train_fn, expert_data, batch_size=128, num_epochs=20,
                    summary_writer=None):
    print('Cloning behaviour:')

    obs, actions = map(np.array, zip(*expert_data))
    indicies = np.arange(obs.shape[0])

    for i_epoch in range(num_epochs):
        losses = []

        np.random.shuffle(indicies)

        t = 0
        T = obs.shape[0]
        while t < T:
            o_batch = np.take(obs, indicies[t: min(t + batch_size, T)], axis=0)
            a_batch = np.take(actions, indicies[t: min(t + batch_size, T)], axis=0)

            loss, summary, global_step = target_policy_train_fn(session, o_batch, a_batch)

            losses.append(loss)

            print('\rTimestep {}/{}'.format(t + 1, T), end='')
            if summary_writer:
                summary_writer.add_summary(summary, global_step)

            t += batch_size

        loss_mean = np.mean(losses)

        if summary_writer:
            summary = tf.Summary()
            summary.value.add(simple_value=loss_mean, tag='epoch/loss_mean')
            summary_writer.add_summary(summary, i_epoch)

        print('\rAverage loss for the epoch: {}'.format(loss_mean))

    return target_policy


def test_target_policy(session, env, target_policy, expert_policy, num_episodes=5, max_timesteps=None, render=None,
                       summary_writer=None):
    print('Testing target policy:')

    max_timesteps = max_timesteps or env.spec.timestep_limit
    rewards = []
    losses = []

    for i_episode in range(num_episodes):
        total_reward = 0.0

        o = np.expand_dims(env.reset(), 0)
        for t in itertools.count():
            a, loss = target_policy(session, o, expert_policy(o))
            o_prime, reward, done, _ = env.step(a)

            total_reward += reward
            losses.append(loss)
            o = np.expand_dims(o_prime, 0)

            if t % max(1, max_timesteps // 10) == 0:
                print('\rEpisode {}/{}: step {}'.format(i_episode + 1, num_episodes, t), end='')

            if render:
                env.render()
            if done or t >= max_timesteps:
                break

        rewards.append(total_reward)
        loss_mean = np.mean(losses)

        if summary_writer:
            summary = tf.Summary()
            summary.value.add(simple_value=loss_mean, tag='episode/test_loss_mean')
            summary_writer.add_summary(summary, i_episode)

    rewards_mean = np.mean(rewards)
    rewards_deviation = np.sqrt(np.mean((rewards - rewards_mean) ** 2))
    print('\nRewards Mean: {}, Rewards deviation: {}'.format(rewards_mean, rewards_deviation))


def main():
    args = parse_args()
    env = gym.make(args.envname)
    expert_policy = load_expert_policy(args.expert_policy_file)

    with tf.Session() as session:
        if args.expert_data_file:
            expert_data = load_expert_data(args.expert_data_file)
        else:
            expert_data = generate_expert_data(env, expert_policy, args.num_expert_episodes, args.max_timesteps)

        target_policy, target_policy_train_fn = build_target_policy(env, args.learning_rate)
        summary_writer = create_summary_writer(session, env, args.summary_dir)

        session.run(tf.global_variables_initializer())

        target_policy = clone_behaviour(session, target_policy, target_policy_train_fn, expert_data, args.batch_size,
                                        args.num_epochs, summary_writer)
        test_target_policy(session, env, target_policy, expert_policy, args.num_test_episodes, args.max_timesteps,
                           args.render, summary_writer)


if __name__ == '__main__':
    main()
