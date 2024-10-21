import gym
from envs.gridworld_env import CliffWalkingWapper  # 导入自定义装饰器
env = gym.make('CliffWalking-v0')  # 定义环境
env = CliffWalkingWapper(env)  # 装饰环境

