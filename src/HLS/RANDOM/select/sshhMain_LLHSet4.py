#!/usr/bin/env python
import random
# This script contains a high level overview of the proposed hybrid algorithm
# The code is strictly mirroring the section 4.1 of the attached paper

import sys
import time
from src.LLH.LLHSet4 import LLHSet4

from src.HLS.SSHH import sshh
from src.LLH.LLHolder import LLHolder
from src.utils import parser, gantt
from src.utils import encoding, decoding
from src.LLH import LLHUtils as llhUtils
from src.utils import config


# Beginning
# Parameters Setting
strs = '../../Brandimarte_Data/Mk06.fjs'
para = parser.parse(strs) # 导入数据
solution = (encoding.generateOS(para), encoding.generateMS(para))
holder =LLHolder(4, solution)
ss = sshh.SequenceSelection(holder.set.llh)


t0 = time.time()
# Initialize the Population
ss.best_solution = solution

ss.prevTime = oriTime = llhUtils.timeTaken(ss.best_solution, para)
print('Ori time:', oriTime)

for epoch in range(0, 5000):
    print('epoch:', epoch, 'os_tabu:', sum(holder.set.os_tabu), 'ms_tabu', sum(holder.set.ms_tabu))
    solution = ss.update_solution(solution, para)
    newTime = llhUtils.timeTaken(solution, para)
    print('time:', newTime)


# Termination Criteria Satisfied ?
gantt_data = decoding.translate_decoded_to_gantt(decoding.decode(para, ss.best_solution[0], ss.best_solution[1]))
print("================================")
t1 = time.time()
total_time = t1 - t0
print("Total time: ", total_time)
print('origin time:', oriTime)
print('final Best time:', llhUtils.timeTaken(ss.best_solution, para))
print(ss.transition_matrix)

if config.latex_export:
    gantt.export_latex(gantt_data)
else:
    gantt.draw_chart(gantt_data)


