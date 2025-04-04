"""
姓名:魏德旭
学号:241404010230
微信:x15279153415
邮箱：wdx20060524@qq.com
"""
#导入所需要的库
import gymnasium as gym
from stable_baselines3 import DQN
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

#定义一个回调函数
class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)  #创建一个空列表 episode_rewards 用于存储每个回合的总奖励
        self.episode_rewards = []                      
        self.current_episode_reward = 0             # 初始化当前回合的累积奖励为 0

    def _on_step(self) -> bool:                     #将当前时间步的奖励累加到 current_episode_reward中
        self.current_episode_reward += self.locals['rewards'][0]
        if self.locals['dones'][0]:                  # 判断当前回合是否结束
            self.episode_rewards.append(self.current_episode_reward)    # 如果回合结束，将当前回合的总奖励添加到 episode_rewards 列表中
            self.current_episode_reward = 0          # 重置当前回合的累积奖励为 0
        return True


# 创建环境并指定渲染模式为 human
env = gym.make('CartPole-v1', render_mode='human')

# 创建奖励记录回调函数
reward_logger = RewardLoggerCallback()

# 手动实现学习率调度
start_lr = 0.001        # 学习率的初始值
end_lr = 0.0001         # 学习率的最终值
decay_fraction = 0.8    # 学习率开始衰减的时间点占总训练步数的比例
total_timesteps = 15000 # 训练步数


def lr_schedule(progress_remaining):        #在训练进度超过 decay_fraction 之前，学习率保持为 start_lr；之后，学习率逐渐从 start_lr 衰减到 end_lr
    if progress_remaining > decay_fraction:  # 在训练进度超过 decay_fraction 之前，学习率保持为 start_lr
        return start_lr
    else:
        return end_lr + (start_lr - end_lr) * (progress_remaining / decay_fraction)    # 之后，学习率逐渐从 start_lr 衰减到 end_lr


# 调整 DQN 算法的超参数,调整不同参数测试实验效果
model = DQN(
    'MlpPolicy',        # 使用多层感知机（MLP）作为策略网络
    env,                 # 指定要训练的环境
    verbose=1,          # 设置日志输出级别为 1
    learning_rate=lr_schedule,      # 使用之前定义的学习率调度函数来动态调整学习率
    gamma=0.99,               # 设置折扣因子为0.99，用于权衡当前奖励和未来奖励的重要性
    buffer_size=200000,     # 经验回放缓冲区的大小，用于存储智能体的经验
    learning_starts=1000,   # 在开始训练之前，先收集 1000 个时间步的经验
    batch_size=128,         # 每次训练时从经验回放缓冲区中采样的样本数量
    train_freq=1,           # 每执行 1 个时间步就进行一次训练
    target_update_interval=1000,   # 每 1000 个时间步更新一次目标网络
    exploration_fraction=0.1,      # 探索率从初始值衰减到最终值的时间占总训练步数的比例
    exploration_final_eps=0.01     # 探索率的最终值
)

# 训练模型，训练总步数为 total_timesteps，并传入 reward_logger 回调函数，用于记录每个回合的总奖励
model.learn(total_timesteps=total_timesteps, callback=reward_logger)

# 测试模型
obs, _ = env.reset()    # 重置环境，获取初始观测值
for _ in range(50):     # 进行 50 次测试
    done = False
    while not done:
        action, _ = model.predict(obs)      #预测智能体的动作
        obs, reward, terminated, truncated, _ = env.step(action)        # 执行预测的动作，获取新的观测值、奖励、回合是否结束的信息
        env.render()             # 渲染环境，显示可视化界面
        done = terminated or truncated      # 判断回合是否结束

# 关闭环境
env.close()

# 绘制训练奖励曲线
plt.plot(reward_logger.episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Reward Curve')
plt.savefig('learning_curve.png')
plt.show()