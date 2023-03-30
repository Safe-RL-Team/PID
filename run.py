import gym
from Pid import PID
import matplotlib.pyplot as plt

RENDER = False  # 在屏幕上显示模拟窗口会拖慢运行速度, 等计算机学得差不多了再显示模拟
DISPLAY_REWARD_THRESHOLD = 400  # 当 回合总 reward 大于设定值时显示模拟窗口

env = gym.make('CartPole-v0').unwrapped  # CartPole 这个模拟
env.seed(0)     # reproducible


RL = PID(
    num_actions=env.action_space.n,
    num_features=env.observation_space.shape[0],
    P=0.02,
    I=0.99,
    D=1
)

for episode in range(150):

    observation = env.reset()

    while True:
        if RENDER:
            env.render()

        action = RL.choose_action(observation)

        observation_next, reward, done, info = env.step(action)

        RL.store_transition(observation, action, reward)   # 存储这一回合的 transition

        if done:
            ep_reward_sum = sum(RL.ep_reward)

            if 'running_reward' not in globals():
                running_reward = ep_reward_sum
            else:
                running_reward = running_reward * 0.99 + ep_reward_sum * 0.01  # 学习率 0.99 衰减率0.01

            if running_reward > DISPLAY_REWARD_THRESHOLD:
                RENDER = True     # rendering

            print("Episode:", episode, "  Reward:", int(running_reward))

            SAV = RL.learn()  # 学习, 输出 vt

            if episode == 0:
                plt.plot(SAV)    # plot 这个回合的 vt
                plt.xlabel('this episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break

        observation = observation_next