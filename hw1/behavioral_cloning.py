import tensorflow as tf
import numpy as np
import itertools
import gym
import os
import time


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_episodes', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--num_test_episodes', type=int, default=5,
                        help='Number of test roll outs for target policy')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--summary_dir', type=str, help='Direction to write summaries')
    return parser.parse_args()


def load_expert_policy(args):
    import load_policy

    assert args.expert_policy_file and args.expert_policy_file != ''
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    return policy_fn


def build_target_policy(env):
    global_step = tf.Variable(0, name='global_step', trainable=False)

    layer1_units = 512
    layer2_units = 256
    layer3_units = 128
    layer4_units = 128

    # ----------------------- PREDICT PART ------------------------
    with tf.name_scope('prediction'):
        observation_input = tf.placeholder(tf.float32, [None, env.observation_space.shape[0]], 'observation_input')

        current_activations = tf.layers.dense(observation_input, layer1_units, tf.nn.tanh, name='hidden_layer_1')
        current_activations = tf.layers.dense(current_activations, layer2_units, tf.nn.tanh, name='hidden_layer_2')
        current_activations = tf.layers.dense(current_activations, layer3_units, tf.nn.tanh, name='hidden_layer_3')
        current_activations = tf.layers.dense(current_activations, layer4_units, tf.nn.tanh, name='hidden_layer_4')
        action_output = tf.layers.dense(current_activations, env.action_space.shape[0], name='action_output')

    # ----------------------- TRAIN PART --------------------------
    with tf.name_scope('training'):
        label_input = tf.placeholder(tf.float32, [None, env.action_space.shape[0]], 'label_input')

        loss = tf.losses.mean_squared_error(label_input, action_output)
        optimizer = tf.train.AdamOptimizer()
        train_optimization = optimizer.minimize(loss, tf.train.get_global_step())

    def predict_fn(session, observation):
        return session.run(action_output, {observation_input: observation})

    def update_fn(session, observation, label):
        _, loss_result, global_step = session.run(
            [train_optimization, loss, tf.train.get_global_step()],
            {
                observation_input: observation,
                label_input: label
            }
        )

        return loss_result

    return predict_fn, update_fn


def create_summary_writer(session, env, summary_dir=None):
    if summary_dir:
        summary_dir = os.path.abspath("./{}/{}/{}".format(summary_dir, env.spec.id, time.strftime("%Y-%m-%d-%H-%M-%S")))
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)

    return tf.summary.FileWriter(summary_dir, session.graph) if summary_dir else None


def clone_behaviour(session, env, expert_policy, batch_size, num_epochs, num_episodes, max_timesteps=None,
                    summary_dir=None):
    max_timesteps = max_timesteps or env.spec.timestep_limit
    transitions = []
    rewards = []

    for i_episode in range(num_episodes):
        total_reward = 0.0

        o = env.reset()
        for t in itertools.count():
            expert_policy_action = np.squeeze(expert_policy(np.expand_dims(o, 0)))
            o_prime, reward, done, _ = env.step(expert_policy_action)

            transitions.append((o, expert_policy_action))

            total_reward += reward
            o = o_prime

            if t % max(1, max_timesteps // 10) == 0:
                print('\rEpisode {}/{}: step {}'.format(i_episode + 1, num_episodes, t), end='')

            if done or t >= max_timesteps:
                break

        rewards.append(total_reward)

    rewards_mean = np.mean(rewards)
    rewards_deviation = np.sqrt(np.mean((rewards - rewards_mean) ** 2))
    print('\nExpert Rewards Mean: {}, Expert Rewards deviation: {}'.format(rewards_mean, rewards_deviation))

    target_policy, update_target_policy = build_target_policy(env)
    summary_writer = create_summary_writer(session, env, summary_dir)

    session.run(tf.global_variables_initializer())

    transitions = np.array(transitions)

    for i_epoch in range(num_epochs):
        losses = []

        np.random.shuffle(transitions)

        t = 0
        T = len(transitions)
        while t < T:
            o_batch, a_batch = map(np.array, zip(*transitions[t: min(t + batch_size, T)]))

            losses.append(update_target_policy(session, o_batch, a_batch))

            print('\rTimestep {}/{}'.format(t + 1, T), end='')

            t += batch_size

        loss_mean = np.mean(losses)

        if summary_writer:
            summary = tf.Summary()
            summary.value.add(simple_value=loss_mean, tag='epoch/loss_mean')
            summary_writer.add_summary(summary, i_epoch)

        print('\nAverage loss for the epoch: {}'.format(loss_mean))

    return target_policy


def test_target_policy(session, env, target_policy, num_episodes=5, max_timesteps=None, render=None):
    print('Testing target policy: ')

    max_timesteps = max_timesteps or env.spec.timestep_limit
    rewards = []

    for i_episode in range(num_episodes):
        total_reward = 0.0

        o = np.expand_dims(env.reset(), 0)
        for t in itertools.count():
            a = target_policy(session, o)
            o_prime, reward, done, _ = env.step(a)

            total_reward += reward
            o = np.expand_dims(o_prime, 0)

            if t % max(1, max_timesteps // 10) == 0:
                print('\rEpisode {}/{}: step {}'.format(i_episode + 1, num_episodes, t), end='')

            if render:
                env.render()
            if done or t >= max_timesteps:
                break

        rewards.append(total_reward)

        print('\nTotal reward for the episode: {}'.format(total_reward))

    rewards_mean = np.mean(rewards)
    rewards_deviation = np.sqrt(np.mean((rewards - rewards_mean) ** 2))
    print('Rewards Mean: {}, Rewards deviation: {}'.format(rewards_mean, rewards_deviation))


def main():
    args = parse_args()
    expert_policy = load_expert_policy(args)

    env = gym.make(args.envname)

    with tf.Session() as session:
        target_policy = clone_behaviour(session, env, expert_policy, batch_size=args.batch_size,
                                        num_epochs=args.num_epochs, num_episodes=args.num_episodes,
                                        max_timesteps=args.max_timesteps, summary_dir=args.summary_dir)

        test_target_policy(session, env, target_policy, args.num_test_episodes, args.max_timesteps, args.render)


if __name__ == '__main__':
    main()
