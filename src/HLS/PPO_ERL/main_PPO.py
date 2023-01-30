import os
import gym
from src.HLS.PPO_ERL.train.run import train_agent, render_agent
from src.HLS.PPO_ERL.train.config import Config, get_gym_env_args
from src.HLS.PPO_ERL.agents.AgentPPO import AgentPPO
from src.HLS.PPO_ERL.env import hh_env


def train_ppo_for_lunar_lander():
    agent_class = AgentPPO  # DRL algorithm name
    env_class = gym.make
    env_args = {
        'env_name': 'LunarLanderContinuous-v2',  # A lander learns to land on a landing pad
        'state_dim': 8,  # coordinates xy, linear velocities xy, angle, angular velocity, two booleans
        'action_dim': 2,  # fire main engine or side engine.
        'if_discrete': False  # continuous action space, symbols → direction, value → force
    }
    get_gym_env_args(env=gym.make('LunarLanderContinuous-v2'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `config.py Arguments()` for hyperparameter explanation
    args.break_step = int(4e5)  # break training if 'total_step > break_step'
    args.net_dims = (64, 32)  # the middle layer dimension of MultiLayer Perceptron
    args.repeat_times = 32  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.lambda_entropy = 0.04  # the lambda of the policy entropy term in PPO

    train_agent(args)
    if input("| Press 'y' to load actor.pth and render:"):
        actor_name = sorted([s for s in os.listdir(args.cwd) if s[-4:] == '.pth'])[-1]
        actor_path = f"{args.cwd}/{actor_name}"
        render_agent(env_class, env_args, args.net_dims, agent_class, actor_path)
def train_ppo_for_fjsp(gpu_id=0):
    agent_class = AgentPPO  # DRL algorithm name
    env_class = gym.make  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'hh_env-v0',  # A pole is attached by an un-actuated joint to a cart.
        # Reward: keep the pole upright, a reward of `+1` for every step taken

        'state_dim': 3,  # (CartPosition, CartVelocity, PoleAngle, PoleAngleVelocity)
        'action_dim': 10,  # (Push cart to the left, Push cart to the right)
        'if_discrete': True,  # discrete action space
    }
    get_gym_env_args(env=gym.make('hh_env-v0'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `config.py Arguments()` for hyperparameter explanation
    args.break_step = int(4e5)  # break training if 'total_step > break_step'
    args.net_dims = (64, 32)  # the middle layer dimension of MultiLayer Perceptron
    args.repeat_times = 32  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.lambda_entropy = 0.04  # the lambda of the policy entropy term in PPO

    train_agent(args)
    if input("| Press 'y' to load actor.pth and render:"):
        actor_name = sorted([s for s in os.listdir(args.cwd) if s[-4:] == '.pth'])[-1]
        actor_path = f"{args.cwd}/{actor_name}"
        render_agent(env_class, env_args, args.net_dims, agent_class, actor_path)


if __name__ == "__main__":
    GPU_ID = 0
    train_ppo_for_lunar_lander()
    #train_ppo_for_fjsp(gpu_id=GPU_ID)