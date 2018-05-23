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
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
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

    layer1_units = 64
    layer2_units = 64

    # ----------------------- PREDICT PART ------------------------
    with tf.name_scope('prediction'):
        observation_input = tf.placeholder(tf.float32, [None, env.observation_space.shape[0]], 'observation-input')

        current_activations = tf.layers.dense(observation_input, layer1_units, tf.nn.tanh, name='hidden-layer-1')
        current_activations = tf.layers.dense(current_activations, layer2_units, tf.nn.tanh, name='hidden-layer-2')
        action_output = tf.layers.dense(current_activations, env.action_space.shape[0], name='action-output')

    # ----------------------- TRAIN PART --------------------------
    with tf.name_scope('training'):
        label_input = tf.placeholder(tf.float32, [None, env.action_space.shape[0]], 'label-input')

        loss = tf.losses.mean_squared_error(label_input, action_output)
        optimizer = tf.train.AdamOptimizer()
        train_optimization = optimizer.minimize(loss, tf.train.get_global_step())

    def predict_fn(session, observation):
        return session.run(action_output, {observation_input: observation})

    def update_fn(session, observation, label):
        loss_result, _, global_step = session.run(
            [loss, train_optimization, tf.train.get_global_step()],
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


def clone_behaviour(session, env, expert_policy, num_episodes=10, max_timesteps=None, summary_dir=None, render=None):
    target_policy, update_target_policy = build_target_policy(env)
    summary_writer = create_summary_writer(session, env, summary_dir)

    session.run(tf.global_variables_initializer())

    max_timesteps = max_timesteps or env.spec.timestep_limit

    for i_episode in range(num_episodes):
        losses = []

        o = np.expand_dims(env.reset(), 0)
        for t in itertools.count():
            expert_policy_action = expert_policy(o)
            o_prime, reward, done, _ = env.step(expert_policy_action)

            losses.append(update_target_policy(session, o, np.array(expert_policy_action)))

            o = np.expand_dims(o_prime, 0)

            if t % max(1, max_timesteps // 10) == 0:
                print('\rEpisode {}/{}: step {}'.format(i_episode + 1, num_episodes, t), end='')

            if render:
                env.render()
            if done or t >= max_timesteps:
                break

        average_loss = np.mean(losses)

        if summary_writer:
            summary = tf.Summary()
            summary.value.add(simple_value=average_loss, tag='episode/loss')
            summary_writer.add_summary(summary, i_episode)

        print('\nAverage loss for the episode: {}'.format(average_loss))

    return target_policy


def main():
    args = parse_args()
    expert_policy = load_expert_policy(args)

    env = gym.make(args.envname)

    with tf.Session() as session:
        result_policy = clone_behaviour(session, env, expert_policy, num_episodes=args.num_rollouts,
                                        max_timesteps=args.max_timesteps,
                                        summary_dir=args.summary_dir, render=args.render)


if __name__ == '__main__':
    main()
