import os
import time

import gym
import torch

from src.HLS.DQN2_ERL.train.evaluator import get_rewards_and_steps, get_rewards_and_steps_solve
from src.LLH.LLHolder import LLHolder
from train.config import Config, get_gym_env_args, build_env
from agents.AgentDQN import AgentDQN
from train.run import train_agent
from env import hh_env

gym.logger.set_level(40)  # Block warning

PROBLEM = 'MK07'
LLH_SET = 1
SOLVE_ITER = 7000
RENDER_TIMES = 20
ACTOR_PATH = f"./hh_env-v0_DQN_0_MK02"
MODULE = '/actor__000000307712.pt'
STATE = 'cla'
REWARD = 'reward3 + endReward4'

def run_dqn_for_hyper_heuristic(gpu_id=0):
    agent_class = AgentDQN  # DRL algorithm
    env_class = gym.make

    env_args = {
        'env_name': 'hh_env-v0',  # A pole is attached by an un-actuated joint to a cart.
        # Reward: keep the pole upright, a reward of `+1` for every step taken

        'state_dim': 3,
        'action_dim': len(LLHolder(LLH_SET)),  # (Push cart to the left, Push cart to the right)
        'if_discrete': True,  # discrete action space
        'problem': PROBLEM,
        'problem_path': os.path.join(os.getcwd(), "../../Brandimarte_Data/" + PROBLEM + ".fjs"),
        'llh_set': LLH_SET,
        'solve_iter': SOLVE_ITER
    }
    # get_gym_env_args(env=gym.make('hh_env-v0'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `config.py Arguments()` for hyperparameter explanation
    args.break_step = int(1e5)  # break training if 'total_step > break_step'
    args.net_dims = (64, 64, 32)  # the middle layer dimension of MultiLayer Perceptron
    args.gpu_id = gpu_id  # the ID of single GPU, -1 means CPU
    args.gamma = 0.95  # discount factor of future rewards
    args.eval_per_step = int(1e4)
    actor_path = ACTOR_PATH

    render_agent(env_class, env_args, args.net_dims, agent_class, actor_path, render_times=RENDER_TIMES)


def render_agent(env_class, env_args: dict, net_dims: [int], agent_class, actor_path: str, render_times: int = 10):
    env = build_env(env_class, env_args)

    state_dim = env_args['state_dim']
    action_dim = env_args['action_dim']
    agent = agent_class(net_dims, state_dim, action_dim, gpu_id=-1)
    # agent.save_or_load_agent(actor_path, if_save=False)
    actor = agent.act
    del agent

    print(f"| render and load actor from: {actor_path}")
    actor.load_state_dict(torch.load(actor_path + MODULE, map_location=lambda storage, loc: storage).state_dict())
    allResult = {}
    timeUsed = {}
    for i in range(render_times):
        t0 = time.time()
        cumulative_reward, episode_step, bestTime = get_rewards_and_steps_solve(env, actor, if_render=False)
        print(f"|{i:4}  cumulative_reward {cumulative_reward:9.3f}  episode_step {episode_step:5.0f}")
        t1 = time.time()
        allResult['test ' + str(i)] = bestTime
        timeUsed['test ' + str(i)] = t1 - t0
    print('problem: ', PROBLEM, 'LLH_SET: ', LLH_SET)
    print('state: ', STATE, 'reward: ', REWARD, 'actor_path: ', actor_path, '  MODULE: ', MODULE, 'net_dims: ', net_dims)
    print(allResult.values())
    print(timeUsed.values())
    print('average time: ', sum(allResult.values()) / len(allResult.values()))
    print('average time used: ', sum(timeUsed.values()) / len(timeUsed.values()))



if __name__ == "__main__":
    GPU_ID = 0
    run_dqn_for_hyper_heuristic()
