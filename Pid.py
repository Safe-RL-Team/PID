import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


class PID:
    def __init__(
            self,
            num_actions,
            num_features,
            P=0.01,
            I=0.95,
            D=0.99
    ):
        self.num_actions = num_actions
        self.num_features = num_features
        self.lr = P
        self.gamma = I
        self.Derivative = D

        self.ep_observation, self.ep_action, self.ep_reward = [], [], []  # 存储 回合信息的 list

        self._build_net()  # 建立 policy 神经网络

        self.sess = tf.Session()  # 使用空值，tf的session就只会使用本地的设备

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_ob = tf.placeholder(tf.float32, [None, self.num_features], name="observations")
            self.tf_ac = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_v = tf.placeholder(tf.float32, [None, ], name="actions_value")

        # 全连接层1
        layer = tf.layers.dense(
            inputs=self.tf_ob,
            units=10,  # 输出个数
            activation=tf.nn.tanh,  # tanh 激励函数
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )

        # 全连接层2
        all_ac = tf.layers.dense(
            inputs=layer,
            units=self.num_actions,  # 输出个数为action_space内action数量
            activation=None,  # 之后再加 Softmax
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        self.all_act_prob = tf.nn.softmax(all_ac, name='act_prob')  # 使用softmax将输出转化为概率

        with tf.name_scope('loss'):
            # 最大化 总体 reward (log_p * R) 就是在最小化 -(log_p * R), 而 tf 的功能里只有minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_ac, labels=self.tf_ac)
            # 所选 action 的概率-log值

            loss = tf.reduce_mean(self.Derivative * neg_log_prob * self.tf_v)
            # (tf_v = 本reward + 衰减的未来reward) 引导参数的导数D项

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)  # 比例项 P* loss

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_ob: observation[np.newaxis, :]})
        # 所有 action 的概率
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        # 根据概率来选 action
        return action

    def store_transition(self, s, a, r):
        # 将这一步的 observation, action, reward 加到列表中去.
        # 本回合完毕之后要清空列表, 然后存储下一回合的数据,
        # 所以会在learn()当中进行清空列表的动作.
        self.ep_observation.append(s)
        self.ep_action.append(a)
        self.ep_reward.append(r)

    def learn(self):
        # 衰减, 并标准化这回合的 reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        self.sess.run(self.train_op, feed_dict={
             self.tf_ob: np.vstack(self.ep_observation),  # shape=[None, n_obs]
             self.tf_ac: np.array(self.ep_action),  # shape=[None, ]
             self.tf_v: discounted_ep_rs_norm,  # shape=[None, ]
        })

        self.ep_observation, self.ep_action, self.ep_reward = [], [], []    # 清空回合 data
        return discounted_ep_rs_norm  # 返回这一回合的 state-action value

    def _discount_and_norm_rewards(self):
        # 衰减, 并标准化这回合的 reward
        discounted_ep_rs = np.zeros_like(self.ep_reward)
        running_add = 0

        for t in reversed(range(0, len(self.ep_reward))):
            running_add = running_add * self.gamma + self.ep_reward[t]  # 积分项
            discounted_ep_rs[t] = running_add

        # 标准化回合奖励
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs


