# 解析fjs数据集文件
'''-in the first line there are (at least) 2 numbers: the first is the number of jobs and the second the number of
machines (the 3rd is not necessary, it is the average number of machines per operation)

-Every row represents one job: the first number is the number of operations of that job, the second number
(let's say k>=1) is the number of machines that can process the first operation; then according to k, there are k pairs
of numbers (machine,processing time) that specify which are the machines and the processing times; then the data for the
second operation and so on...
-在第一行中有（至少）2个数字：第一个是作业数量，第二个是机器数量（第三个不是必须的，它是每次操作的平均机器数量）

-每一行代表一个作业：第一个数字是该作业的操作数，第二个数字（假设k>=1）是可以处理第一个操作的机器数；然后根据k，有k对数字（机器、处理时间），
指定哪些是机器和处理时间；然后是第二次操作的数据等等。。。

返回结构：一个字典，包含机器数量machinesNb和作业的列表jobs
jobs是一个list，元素为所有工件/作业 operations。
operations 是一个list， 元素为每个工件的各个工序operation。
operation是一个list，元素为dict，是该工序在各个机器上的加工时间，每个dict包含两个键值对，分别是“机器序号”-机器序号，“加工时间”-加工时间

'''

def parse(path):
    file = open(path, 'r')

    # 读入第一行
    firstLine = file.readline()
    firstLineValue = list(map(int, firstLine.split()[0:2]))

    jobsNb = firstLineValue[0]  # 作业/工件数量
    machinesNb = firstLineValue[1] # 机器数量

    jobs = [] # 作业列表

    for i in range(jobsNb):
        currentLine = file.readline()
        currentLineValues = list(map(int, currentLine.split()))

        operations = [] # 作业列表

        j = 1 # 索引从1开始，首位为作业的操作数量
        while j < len(currentLineValues):
            k = currentLineValues[j]  # 可处理该操作的机器数
            j = j + 1

            operation = []

            for ik in range(k): # 处理k对
                machine = currentLineValues[j] # 机器序号
                j = j + 1
                processingTime = currentLineValues[j] # 该机器加工时间
                j = j + 1

                operation.append({'machine': machine, 'processingTime' : processingTime})
            operations.append(operation)
        jobs.append(operations)

    file.close()

    return {'machinesNb': machinesNb, 'jobs': jobs}


