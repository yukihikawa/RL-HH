#!/usr/bin/env python

from src.utils import encoding, decoding, gantt
from src.utils.encoding import initializeResult

# This script runs every non-trivial piece of code defined in this project to
# easily test their behavior


op11 = [{'machine': 0, 'processingTime': 2}, {'machine': 1, 'processingTime': 3}]
op12 = [{'machine': 2, 'processingTime': 1}, {'machine': 3, 'processingTime': 2}]
op13 = [{'machine': 3, 'processingTime': 1}, {'machine': 4, 'processingTime': 2}]
job1 = [op11, op12, op13]
op21 = [{'machine': 2, 'processingTime': 1}, {'machine': 3, 'processingTime': 2}]
op22 = [{'machine': 1, 'processingTime': 1}, {'machine': 4, 'processingTime': 2}]
job2 = [op21, op22]
op31 = [{'machine': 0, 'processingTime': 1}, {'machine': 4, 'processingTime': 2}]
job3 = [op31]
jobs = [job1, job2, job3]

parameters =  {'machinesNb': 5, 'jobs': jobs}

solution = initializeResult(parameters)
print(solution)
gantt_data = decoding.translate_decoded_to_gantt(decoding.decode(parameters, solution[0], solution[1]))
gantt.draw_chart(gantt_data)