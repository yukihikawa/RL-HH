import os
import random

from src.HLS.ILS.actionILS import action

problem = 'MK06'
problem_str = os.path.join(os.getcwd(), "../../Brandimarte_Data/" + problem + ".fjs")
action_manager = action()
action_manager.llh_manager.reset(problem_str)
print(action_manager.llh_manager.previous_solution)
# for vnd in action_manager.llh_manager.vnd:
#     print(vnd)
#     for i in range(1000):
#         print('vnd: ', vnd, 'iter: ', i)
#         print(vnd(action_manager.llh_manager.previous_solution))
#
# for shaker in action_manager.llh_manager.shake:
#     print(shaker)
#     for i in range(1000):
#         print('vnd: ', shaker, 'iter: ', i)
#         print(shaker())
print(action_manager.actions)
for i in range(len(action_manager.actions)):
    print('i: ', i, 'action: ', action_manager.actions[i])
for turn in range(40):
    idx = random.randint(0, len(action_manager.actions) - 1)
    # print(action_manager.llh_manager.total_duration)
    # print(action_manager.llh_manager.total_improvement)
    # print(action_manager.llh_manager.total_Noop)
    # print(action_manager.llh_manager.total_selected)
    # print(action_manager.llh_manager.total_accepted)
    #
    # print(action_manager.llh_manager.eval_recent_improve)
    # print(action_manager.llh_manager.eval_by_accept)
    # print(action_manager.llh_manager.eval_improve_overtime)
    # print(action_manager.llh_manager.eval_by_speed)
    # print(action_manager.llh_manager.eval_by_speed_accepted)
    # print(action_manager.llh_manager.eval_by_speed_new)
    # print('action:', idx)
    # print(action_manager.actions[idx][0], ' and ', action_manager.actions[idx][1])
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('old_p: ', action_manager.llh_manager.previous_time)
    action_manager.execute(idx)
    print('new_p: ', action_manager.llh_manager.previous_time)
    # print(action_manager.actions)
    # for i in range(len(action_manager.actions)):
    #     print('i: ', i, 'action: ', action_manager.actions[i])
    print('best: ', action_manager.llh_manager.best_time)



# print('===========================================')
# print(action_manager.llh_manager.previous_time)
# action_manager.execute(19)
# print(action_manager.llh_manager.previous_time)