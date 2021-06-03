import datetime,gym,os,pybullet_envs,time,psutil,ray,imageio
from atari_wrappers import make_atari, wrap_deepmind
from ReplayBuffer import ReplayBuffer
from DQN import DQNNetwork
import tensorflow as tf
import numpy as np
from config import Config
from collections import deque

class Agent(object):
    def __init__(self, ):
        # Config
        self.config = Config()

        # Environment
        self.env, self.eval_env = get_envs()
        self.odim = self.env.observation_space.shape
        self.adim = self.env.action_space.n

        # Network
        self.main_network = DQNNetwork(self.odim, self.adim)
        self.target_network = DQNNetwork(self.odim, self.adim)
        self.main_network.build(input_shape=(None, 84, 84, 4))
        self.target_network.build(input_shape=(None, 84, 84, 4))
        self.main_network.summary()
        self.gamma = self.config.gamma
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-6)
        self.loss = tf.keras.losses.Huber()
        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        self.q_metric = tf.keras.metrics.Mean(name="Q_value")
        self.init_explr = 1.0
        self.final_explr = 0.1
        self.final_explr_frame = 1000000
        self.replay_start_size = 10000
        self.log_path = "./log/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_BreakoutNoFrameskip-v4"
        self.summary_writer = tf.summary.create_file_writer(self.log_path + "/summary/")

        # Buffer (Memory)
        self.buffer = ReplayBuffer(buffer_size=self.config.buffer_size, odim=self.odim, adim=self.adim, batch_size=self.config.mini_batch_size)

    @tf.function
    def get_eps(self, current_step, terminal_eps=0.01, terminal_frame_factor=25):
        """Use annealing schedule similar like: https://openai.com/blog/openai-baselines-dqn/ .
        Args:
            current_step (int): Number of entire steps agent experienced.
            terminal_eps (float): Final exploration rate arrived at terminal_frame_factor * self.final_explr_frame.
            terminal_frame_factor (int): Final exploration frame, which is terminal_frame_factor * self.final_explr_frame.
        Returns:
            eps (float): Calculated epsilon for ε-greedy at current_step.
        """
        terminal_eps_frame = self.final_explr_frame * terminal_frame_factor

        if current_step < self.replay_start_size:
            eps = self.init_explr
        elif self.replay_start_size <= current_step and current_step < self.final_explr_frame:
            eps = (self.final_explr - self.init_explr) / (self.final_explr_frame - self.replay_start_size) * (current_step - self.replay_start_size) + self.init_explr
        elif self.final_explr_frame <= current_step and current_step < terminal_eps_frame:
            eps = (terminal_eps - self.final_explr) / (terminal_eps_frame - self.final_explr_frame) * (current_step - self.final_explr_frame) + self.final_explr
        else:
            eps = terminal_eps
        return eps


    @tf.function
    def getQ(self, obs):
        Q=self.main_network(obs)
        return Q

    @tf.function
    def update_main_network(self, o_batch, a_batch, r_batch, o1_batch, d_batch):
        # print('update main network', o_batch, a_batch, r_batch, o1_batch, d_batch)

        with tf.GradientTape() as tape:
            o1_q = self.target_network(o1_batch)
            max_o1_q = tf.reduce_max(o1_q, axis=1)
            expected_q = r_batch + self.gamma * max_o1_q * (1.0-d_batch)
            main_q = tf.reduce_sum(self.main_network(o_batch) * tf.one_hot(a_batch, self.env.action_space.n, 1.0, 0.0), axis=1)
            loss = self.loss(tf.stop_gradient(expected_q), main_q)

        gradients = tape.gradient(loss, self.main_network.trainable_weights)
        clipped_gradients = [tf.clip_by_norm(grad, 10) for grad in gradients]
        self.optimizer.apply_gradients(zip(clipped_gradients, self.main_network.trainable_variables))
        self.loss_metric.update_state(loss)
        self.q_metric.update_state(main_q)

        return loss

    # @tf.function
    def update_target_network(self):
        print('update target network', self.target_network)
        self.target_network.set_weights(self.main_network.get_weights())


    def train(self):
        start_time = time.time()
        latests_100_score = deque(maxlen=100)
        o, r, d, ep_ret, ep_len, n_env_step = self.env.reset(), 0, False, 0, 0, 0
        for epoch in range(self.config.epochs):
            o, d, ep_ret, ep_len = self.env.reset(), False, 0, 0
            o = np.array(o)

            while not (d):
                eps = self.get_eps(tf.constant(n_env_step, tf.float32))
                if tf.random.uniform((), minval=0, maxval=1) < eps:
                    a = self.env.action_space.sample()
                else:
                    Q = self.getQ(tf.cast(tf.constant(value=tf.expand_dims(o, axis=0)), dtype=tf.float32))
                    a = tf.cast(tf.argmax(tf.squeeze(Q)), tf.int32).numpy()
                o1, r, d, _ = self.env.step(a)
                o1 = np.array(o1)
                ep_len += 1
                # r = r + ep_len * 0.1
                ep_ret += r
                n_env_step += 1
                # Save the Experience to our buffer
                self.buffer.append(o, a, r, o1, d)
                o = o1

                if len(self.buffer.buffer) > 5000:
                    o_batch, a_batch, r_batch, o1_batch, d_batch = self.buffer.sample()
                    self.update_main_network(tf.constant(value=o_batch, dtype='float32'), tf.constant(value=a_batch, dtype='int32'), tf.constant(value=r_batch, dtype='float32'), tf.constant(value=o1_batch, dtype='float32'), tf.constant(value=d_batch, dtype='float32'))

            # Evaluate
            if (epoch == 0) or (((epoch + 1) % self.config.evaluate_every) == 0):
                ram_percent = psutil.virtual_memory().percent  # memory usage
                print("[Eval. start] step:[%d/%d][%.1f%%] #step:[%.1e] time:[%s] ram:[%.1f%%]." %
                      (epoch + 1, self.config.epochs, epoch / self.config.epochs * 100,
                       n_env_step,
                       time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)),
                       ram_percent)
                      )
                # Update target network
                self.update_target_network()
                o, d, ep_ret, ep_len = self.eval_env.reset(), False, 0, 0
                _ = self.eval_env.render(mode='human')
                while not (d or (ep_len == self.config.steps_per_epoch)):
                    # a = sess.run(model['mu'], feed_dict={model['o_ph']: o.reshape(1, -1)})
                    Q = self.getQ(tf.cast(tf.constant(value=tf.expand_dims(o, axis=0)), dtype=tf.float32))
                    a = tf.cast(tf.argmax(tf.squeeze(Q)), tf.int32).numpy()
                    o, r, d, _ = self.eval_env.step(a)
                    _ = self.eval_env.render(mode='human')
                    ep_ret += r  # compute return
                    ep_len += 1
                print("[Evaluate] ep_ret:[%.4f] ep_len:[%d]" % (ep_ret, ep_len))
                latests_100_score.append((ep_ret))
                self.write_summary(epoch, latests_100_score, ep_ret, n_env_step, eps)
                print("Saving weights...")
                self.main_network.save_weights(self.log_path + "/weights/episode_{}".format(epoch))
                # self.play(self.log_path + "/weights/", episode=ep_len)

    def write_summary(self, episode, latest_100_score, episode_score, total_step, eps):

        with self.summary_writer.as_default():
            tf.summary.scalar("Reward (clipped)", episode_score, step=episode)
            tf.summary.scalar("Latest 100 avg reward (clipped)", np.mean(latest_100_score), step=episode)
            tf.summary.scalar("Loss", self.loss_metric.result(), step=episode)
            tf.summary.scalar("Average Q", self.q_metric.result(), step=episode)
            tf.summary.scalar("Total Frames", total_step, step=episode)
            tf.summary.scalar("Epsilon", eps, step=episode)

        self.loss_metric.reset_states()
        self.q_metric.reset_states()
    # @tf.function
    # def learn(self):
    #     q_target = rewards + (1- dones) * self.gamma * tf.reduce_max(self.target(next_o))
    def play(self, load_dir=None, episode=None, trial=5, max_playing_time=10):

        if load_dir:
            loaded_ckpt = tf.train.latest_checkpoint(load_dir)
            self.main_network.load_weights(loaded_ckpt)

        frame_set = []
        reward_set = []
        test_env = self.eval_env
        for _ in range(trial):

            o = test_env.reset()
            frames = []
            test_step = 0
            test_reward = 0
            done = False

            while not done:

                frames.append(test_env.render())

                Q = self.getQ(tf.cast(tf.constant(value=tf.expand_dims(o, axis=0)), dtype=tf.float32))
                a = tf.cast(tf.argmax(tf.squeeze(Q)), tf.int32).numpy()

                o, reward, done, info = test_env.step(a)
                test_reward += reward


                test_step += 1

                if done and (info["ale.lives"] != 0):
                    test_env.reset()
                    test_step = 0
                    done = False

                if len(frames) > 15 * 60 * max_playing_time:  # To prevent falling infinite repeating sequences.
                    print("Playing takes {} minutes. Force termination.".format(max_playing_time))
                    break

            reward_set.append(test_reward)
            frame_set.append(frames)

        best_score = np.max(reward_set)
        print("Best score of current network ({} trials): {}".format(trial, best_score))
        best_score_ind = np.argmax(reward_set)
        imageio.mimsave("test.gif", frame_set[best_score_ind], fps=15)

        if episode is not None:
            with self.summary_writer.as_default():
                tf.summary.scalar("Test score", best_score, step=episode)

def get_envs():
    env_name = 'BreakoutNoFrameskip-v4'
    env,eval_env = wrap_deepmind(make_atari(env_name), clip_rewards=True, frame_stack=True),\
                   wrap_deepmind(make_atari(env_name), clip_rewards=False, frame_stack=True)
    _ = eval_env.reset()
    _ = eval_env.render(mode='human') # enable rendering on test_env
    for _ in range(3): # dummy run for proper rendering
        a = eval_env.action_space.sample()
        o,r,d,_ = eval_env.step(a)
        time.sleep(0.01)
    return env,eval_env

a = Agent()
a.play('./log/20210603_101243_BreakoutNoFrameskip-v4/weights/')