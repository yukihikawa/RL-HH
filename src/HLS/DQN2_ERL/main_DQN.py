import os
import gym
from train.config import Config, get_gym_env_args
from agents.AgentDQN import AgentDQN
from train.run import train_agent, render_agent
from env import hh_env
gym.logger.set_level(40)  # Block warning




def train_dqn_for_hyper_heuristic(gpu_id=0):
    agent_class = AgentDQN  # DRL algorithm
    env_class = gym.make
    env_args = {
        'env_name': 'hh_env-v0',  # A pole is attached by an un-actuated joint to a cart.
        # Reward: keep the pole upright, a reward of `+1` for every step taken

        'state_dim': 3,  # (CartPosition, CartVelocity, PoleAngle, PoleAngleVelocity)
        'action_dim': 8,  # (Push cart to the left, Push cart to the right)
        'if_discrete': True,  # discrete action space
    }
    #get_gym_env_args(env=gym.make('hh_env-v0'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `config.py Arguments()` for hyperparameter explanation
    args.break_step = int(1e6)  # break training if 'total_step > break_step'
    args.net_dims = (64, 64, 32)  # the middle layer dimension of MultiLayer Perceptron
    args.gpu_id = gpu_id  # the ID of single GPU, -1 means CPU
    args.gamma = 0.95  # discount factor of future rewards

    train_agent(args)


if __name__ == "__main__":
    GPU_ID = 0
    train_dqn_for_hyper_heuristic()
