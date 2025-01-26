import pandas as pd
import numpy as np
import os

tasks = ['cola', 'mrpc', 'qnli', 'rte', 'sst2', 'stsb']
lrs = [1e-05, 2e-05, 5e-05, 0.0001, 0.0002, 0.0005]
ex_num = 5

def one_task(task, dir):
    reses = {}
    for lr in lrs:
        tmp = []
        lr_dir = os.path.join(dir, 'df_logs', task, str(lr))
        if os.path.isdir(lr_dir):
            for i in range(ex_num):
                filepath = os.path.join(dir, 'df_logs', task, str(lr), str(i+1)+'.csv')
                df = pd.read_csv(filepath)
                tmp.append(df.iloc[-1]['valid_metrics'])
            res = sum(tmp)/len(tmp)
            reses[lr] = res
        else:
            continue
    print(task)
    print(reses)
    print(max(reses, key=reses.get))

dir = input('input directory: ')

for task in tasks:
    one_task(task, dir)

