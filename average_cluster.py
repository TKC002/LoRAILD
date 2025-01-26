import os
import sys
import pickle
import yaml
import glob
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans

from lib.util.result import *
from lib.util.define import *
from lib.util.conf import *

tasks = ['cola', 'mrpc', 'qnli', 'rte', 'sst2']
splits = ['train', 'valid', 'test']
models = ['teacher', 'student', 'teacher_all']
subjects = ['ild_features', 'intermediate_output']

args = sys.argv
if len(args) == 1:
  yaml_path = input('input path for yaml file: ')
else:
  yaml_path = args[1]
with open(yaml_path, 'r') as f:
    conf = yaml.safe_load(f)

# conf needs
# - work dir : /work2/analyze/roberta/clustering/all
# - outdirs
# - outdir
# - ex_num

for method in conf['out_dirs']:
    print(f'method = {method}')
    for task in tasks:
        print(f'task = {task}')
        for split in splits:
            for model in models:
                for subject in subjects:
                    data_frames = []
                    for i in range(1, conf['ex_num']+1):
                        file_path = f'{conf["work_dir"]}/{i}/{conf["out_dirs"][method]}/{task}/**/{split}/**/{model}/{subject}/metrics/*.csv'
                        print(file_path)
                        files = glob.glob(file_path, recursive=True)
                        if len(files) == 0:
                            print(files)
                            break
                        elif len(files) > 1:
                            print(files)
                            exit(1)
                        else:
                            print('averaging...')
                            df = pd.read_csv(files[0], index_col=0, header=0)
                            data_frames.append(df)
                    if len(data_frames)==conf['ex_num']:
                        for dfi in data_frames[1:]:
                            data_frames[0] = data_frames[0]+dfi
                            
                        data_frames[0] = data_frames[0]/len(data_frames)

                        outpath = f'{conf["outdir"]}/average/{method}/{task}/{split}/{model}/{subject}/metrics'
                        os.makedirs(outpath, exist_ok=True)
                        data_frames[0].to_csv(f'{outpath}/metrics.csv')

            

