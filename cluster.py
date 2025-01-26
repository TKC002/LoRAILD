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
args = sys.argv
if len(args) == 1:
  yaml_path = input('input path for yaml file: ')
else:
  yaml_path = args[1]
# yaml_path = input('input path for yaml file: ')
with open(yaml_path, 'r') as f:
    conf = yaml.safe_load(f)

def cluster_metric(feature, labels):
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    cluster_labels = kmeans.fit_predict(feature)
    
    print("ajusted_rand_score computing...")
    ars = metrics.adjusted_rand_score(labels, cluster_labels)

    print("adjusted_mutual_info_score computing")
    amis = metrics.adjusted_mutual_info_score(labels, cluster_labels)
    
    print("v_measure_score computing")
    vms = metrics.v_measure_score(labels, cluster_labels)

    print("fowlkes_mallows_score computing")
    fms = metrics.fowlkes_mallows_score(labels, cluster_labels)

    print("silhouette_score computing")
    ss = metrics.silhouette_score(feature, cluster_labels, metric='euclidean')

    print("calinski_harabasz_score computing...")
    chs = metrics.calinski_harabasz_score(feature, cluster_labels)

    print("davies_bouldin_score computing")
    dbs = metrics.davies_bouldin_score(feature, cluster_labels)
    
    metric_list = {'ars': ars, 'amis': amis, 'vms': vms, 'fms': fms, 'ss': ss, 'chs': chs, 'dbs': dbs}
    # metric_list = {'ars': ars}
    return cluster_labels, metric_list

def cluster_3d(k, l, features, result, matrices, metric_lists, labels):
    result[k][l] = {}
    matrices[k][l] = {}
    metric_lists[k][l] = {}
    layer_num = features[k][l].size()[1]
    for layer in range(layer_num):
        print(f'layer = {layer}')
        feature = features[k][l][:,layer, :].numpy()
        cluster_labels, metric_list = cluster_metric(feature, labels)
        result[k][l][layer] = cluster_labels
        metric_lists[k][l][layer] = metric_list
        matrices[k][l][layer] = defaultdict(lambda: defaultdict(int))
        for i in range(len(cluster_labels)):
            matrices[k][l][layer][f'labels_{labels[i]}'][f'culster_{cluster_labels[i]}'] +=1
    print('\n')

def cluster_2d(k, l, features, result, matrices, metric_lists, labels):
    print('no layer')
    feature = features[k][l].numpy()
    cluster_labels, metric_list = cluster_metric(feature, labels)
    result[k][l] = cluster_labels
    metric_lists[k][l] = metric_list
    matrices[k][l] = defaultdict(lambda: defaultdict(int))
    for i in range(len(cluster_labels)):
        matrices[k][l][f'labels_{labels[i]}'][f'culster_{cluster_labels[i]}'] +=1

def save(outpath, result, matrices, metric_lists):
    print('saving...')
    for k in result:
        for l in result[k]:
            # print(type(result[k][l]))
            if isinstance(result[k][l], dict):
                print('save 3d')

                saving_dir = f'{outpath}/{l}/{k}/metrics'
                os.makedirs(saving_dir, exist_ok=True)
                met_file = f'{saving_dir}/metrics.csv'
                df = pd.DataFrame.from_dict(metric_lists[k][l])
                df.to_csv(met_file)

                for layer in result[k][l]:
                    saving_dir = f'{outpath}/{l}/{k}/result'
                    os.makedirs(saving_dir, exist_ok=True)
                    res_file = f'{saving_dir}/{layer}.txt'
                    np.savetxt(res_file, np.array(result[k][l][layer]), delimiter=',', fmt='%d')
                    
                    saving_dir = f'{outpath}/{l}/{k}/matrices'
                    os.makedirs(saving_dir, exist_ok=True)
                    mat_file = f'{saving_dir}/{layer}.csv'
                    df = pd.DataFrame.from_dict(matrices[k][l][layer])
                    df.to_csv(mat_file)

            elif isinstance(result[k][l], list) or isinstance(result[k][l], np.ndarray):
                print('save 2d')
                saving_dir = f'{outpath}/{l}/{k}/'
                os.makedirs(saving_dir, exist_ok=True)
                res_file = f'{saving_dir}/result.txt'
                np.savetxt(res_file, np.array(result[k][l]), delimiter=',', fmt='%d')

                met_file = f'{saving_dir}/metrics.csv'
                df = pd.Series(metric_lists[k][l])
                df.to_csv(met_file)
                
                mat_file = f'{saving_dir}/matrices.csv'
                df = pd.DataFrame.from_dict(matrices[k][l])
                df.to_csv(mat_file)

def clustering(features):
    print('clustering...')
    result = {}
    matrices = {}
    metric_lists = {}
    labels = features['labels']
    # features = {}
    # features[k][l] = tensor size = (data, layer, feature)
    for k in features:
        if k != 'labels':
            result[k] = {}
            matrices[k] = {}
            metric_lists[k] = {}
            for l in features[k]: # teacher or student
                print(k, l)
                if features[k][l].dim() == 3:
                    cluster_3d(k, l, features, result, matrices, metric_lists, labels)
                elif features[k][l].dim() == 2:
                    cluster_2d(k, l, features, result, matrices, metric_lists, labels)

    return result, matrices, metric_lists

for method in conf['out_dirs']:
    print(f'method = {method}')
    for task in tasks:
        print(f'task = {task}')
        files = glob.glob(f'{conf["out_dirs"][method]}/**/{task}/**/*.pickle', recursive=True)
        for file in files: # train, valid test, batch_number
            print(f'file = {file}')
            with open(file, 'rb') as f:
                print('file loading')
                features = pickle.load(f) # dict of dict outer keys are [intermediate_output, ild_features, labels], inner keys are [teacher, student]
                print('loading complete!!')
            dirs = file.split('/')
            # dirs[-1] = dd-dd.pickle
            dirs[-1] = dirs[-1].rstrip('pickle')
            dirs[-1] = dirs[-1].rstrip('.')
            outpath = f"{conf['outdir']}/{method}/{task}/{dirs[-4]}/{dirs[-3]}/{dirs[-2]}/{dirs[-1]}"
            result, matrices, metric_lists = clustering(features)
            save(outpath, result, matrices, metric_lists)

            

