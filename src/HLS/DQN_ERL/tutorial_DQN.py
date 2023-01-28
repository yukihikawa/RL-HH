import os
import gym
from config import Config, get_gym_env_args
from agent import AgentDQN
from run import train_agent, render_agent
from env import hh_env
gym.logger.set_level(40)  # Block warning


def train_dqn_for_cartpole(gpu_id=0):
    agent_class = AgentDQN  # DRL algorithm
    env_class = gym.make
    env_args = {
        'env_name': 'CartPole-v0',  # A pole is attached by an un-actuated joint to a cart.
        # Reward: keep the pole upright, a reward of `+1` for every step taken

        'state_dim': 4,  # (CartPosition, CartVelocity, PoleAngle, PoleAngleVelocity)
        'action_dim': 2,  # (Push cart to the left, Push cart to the right)
        'if_discrete': True,  # discrete action space
    }
    get_gym_env_args(env=gym.make('CartPole-v0'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `config.py Arguments()` for hyperparameter explanation
    args.break_step = int(1e5)  # break training if 'total_step > break_step'
    args.net_dims = (64, 32)  # the middle layer dimension of MultiLayer Perceptron
    args.gpu_id = gpu_id  # the ID of single GPU, -1 means CPU
    args.gamma = 0.95  # discount factor of future rewards

    train_agent(args)

def train_dqn_for_hyper_heuristic(gpu_id=0):
    agent_class = AgentDQN  # DRL algorithm
    env_class = gym.make
    env_args = {
        'env_name': 'hh_env-v0',  # A pole is attached by an un-actuated joint to a cart.
        # Reward: keep the pole upright, a reward of `+1` for every step taken

        'state_dim': 3,  # (CartPosition, CartVelocity, PoleAngle, PoleAngleVelocity)
        'action_dim': 10,  # (Push cart to the left, Push cart to the right)
        'if_discrete': True,  # discrete action space
    }
    get_gym_env_args(env=gym.make('hh_env-v0'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `config.py Arguments()` for hyperparameter explanation
    args.break_step = int(1e5)  # break training if 'total_step > break_step'
    args.net_dims = (64, 32)  # the middle layer dimension of MultiLayer Perceptron
    args.gpu_id = gpu_id  # the ID of single GPU, -1 means CPU
    args.gamma = 0.95  # discount factor of future rewards

    train_agent(args)

def train_dqn_for_lunar_lander(gpu_id=0):
    agent_class = AgentDQN  # DRL algorithm
    env_class = gym.make
    env_args = {
        'env_name': 'LunarLander-v2',  # A lander learns to land on a landing pad and using as little fuel as possible
        # Reward: Lander moves to the landing pad and come rest +100; lander crashes -100.
        # Reward: Lander moves to landing pad get positive reward, move away gets negative reward.
        # Reward: Firing the main engine -0.3,  side engine -0.03 each frame.

        'state_dim': 8,  # coordinates xy, linear velocities xy, angle, angular velocity, two booleans
        'action_dim': 4,  # do nothing, fire left engine, fire main engine, fire right engine.
        'if_discrete': True  # discrete action space
    }
    get_gym_env_args(env=gym.make('LunarLander-v2'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `config.py Arguments()` for hyperparameter explanation
    args.break_step = int(4e5)  # break training if 'total_step > break_step'
    args.gpu_id = gpu_id  # the ID of single GPU, -1 means CPU
    args.explore_rate = 0.1  # the probability of choosing action randomly in epsilon-greedy
    args.net_dims = (128, 64)  # the middle layer dimension of Fully Connected Network

    train_agent(args)
    if input("| Press 'y' to load actor.pth and render:"):
        actor_name = sorted([s for s in os.listdir(args.cwd) if s[-4:] == '.pth'])[-1]
        actor_path = f"{args.cwd}/{actor_name}"
        render_agent(env_class, env_args, args.net_dims, agent_class, actor_path)


if __name__ == "__main__":
    GPU_ID = 0
    #train_dqn_for_cartpole(GPU_ID)
    #train_dqn_for_lunar_lander(GPU_ID)
    train_dqn_for_hyper_heuristic()
