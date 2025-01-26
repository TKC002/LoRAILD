from scipy import stats
import pandas as pd
import glob
from collections import defaultdict
import statistics
import os
import copy
import itertools


def statistic(x, y):
    return sum(x)/len(x) - sum(y)/len(y)

def write_result(filepath, significant):
    with open(filepath, 'w') as f:
        for method in significant:
            f.write(method+'\n')
            for task in significant[method]:
                f.write('\t'+task+'\n')
                for small in significant[method][task]:
                    f.write('\t\t'+small+'\n')


def extract_result(results_dirs):
    valid_results = []
    test_results = []
    for i, dir in enumerate(results_dirs):
        df = pd.read_csv(dir)
        max_ind = df['valid_metrics'].idxmax()
        valid_results.append(df.loc[max_ind, 'valid_metrics'])
        test_results.append(df.loc[max_ind, 'test_metrics'])

    return valid_results, test_results

def average_result(results):
    averaged = {}
    for method in results:
        one_method = {}
        for task in results[method]:
            one_method[task] = sum(results[method][task])/len(results[method][task])
        averaged[method] = one_method
    return averaged

def permutaion_test(results, averaged, p):
    methods = list(results.keys())
    tasks = results[methods[0]].keys()
    method_combination = list(itertools.combinations(methods, 2))
    significant = {}
    for method in methods:
        significant[method] = {}
        for task in tasks:
            significant[method][task] = []
    
    for c in method_combination:
        # c is something like (nokd, KD)
        for task in tasks:
            if averaged[c[0]][task] >= averaged[c[1]][task]:
                larger = c[0]
                smaller = c[1]
            else:
                larger = c[1]
                smaller = c[0]
            ptest_res = stats.permutation_test((results[larger][task], results[smaller][task]), statistic, alternative='greater')
            if ptest_res.pvalue < p:
                significant[larger][task].append(smaller)
    return significant