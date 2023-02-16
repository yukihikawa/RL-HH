import os

from src.LLH.LLHUtils import timeTaken
from src.LLH.LLHolder import LLHolder
import pandas as pd
import numpy as np

from src.utils.encoding import initializeResult
from src.utils.parser import parse

PROBLEM = ['MK01', 'MK02', 'MK03', 'MK04', 'MK05', 'MK06', 'MK07', 'MK08', 'MK09', 'MK10']
ORI_TIME = {'MK01': 88, 'MK02': 54, 'MK03': 398, 'MK04': 209, 'MK05': 339, 'MK06': 284, 'MK07': 369, 'MK08': 709, 'MK09': 613, 'MK10': 639}
IDEAL_TIME = {'MK01': (36, 42), 'MK02': (27, 32), 'MK03': (204, 211), 'MK04': (60, 81), 'MK05': (168, 186), 'MK06': (60, 86), 'MK07': (133, 157), 'MK08': (523, 523), 'MK09': (299, 369), 'MK10': (165, 296)}
BEST_TIME = {'MK01': 40, 'MK02': 26, 'MK03': 204, 'MK04': 60, 'MK05': 171, 'MK06': 57, 'MK07': 139, 'MK08': 523, 'MK09': 307, 'MK10': 197}

P_IDX = 1
problem_path = os.path.join(os.getcwd(), "../../../Brandimarte_Data/" + PROBLEM[P_IDX] + ".fjs")
LLH_SET = 1
holder = LLHolder(LLH_SET)

N_STATE = ORI_TIME[PROBLEM[P_IDX]] - BEST_TIME[PROBLEM[P_IDX]] + 1
ACTIONS = [i for i in range(len(holder.set.llh))]
EPSILON = 0.9
ALPHA = 0.1
GAMMA = 0.9
MAX_EPISODES = 500
FRESH_TIME = 0.3
TERMINATION_FLAG = 'terminal'


def build_q_table(n_states, actions):
    table = pd.DataFrame(# 用 pandas 建立 q_table
        np.zeros((n_states, len(actions))),  # q_table initial values
        columns=actions,  # action's name
    )
    return table

def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:  # act greedy
        action_name = state_actions.argmax()
    return action_name

