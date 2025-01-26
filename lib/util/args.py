import argparse
import yaml

def prepro_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='task', required=True)
    parser.add_argument('--tokenizer', type=str, help='', required=False)
    parser.add_argument('--t_tokenizer', type=str, help='', required=False)
    parser.add_argument('--max_length', type=int, help='', required=True)
    parser.add_argument('--mask_prob', type=float, help='', required=False)
    args = parser.parse_args()
    return args

def dark_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all_conf', type=str, help='path of dark knowledge configuration', required=True)
    parser.add_argument('--task', type=str, help='', required=True)
    parser.add_argument('--max_epoch', type=int, help='', required=True)
    # ------------------------------------------------------------------------
    parser.add_argument('--method_conf', type=str, help='name or path of your configuration file', required=False)
    parser.add_argument('--task_conf', type=str, help='', required=False)
    parser.add_argument('--common2', type=str, help='path to second common config', required=False)
    parser.add_argument('--common3', type=str, help='path to second common config', required=False)
    parser.add_argument('--nep_token', type=str, help='path to second common config', required=False)
    args = parser.parse_args()
    return args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, help='learning rate', required=True)
    parser.add_argument('--task', type=str, help='task', required=True)
    parser.add_argument('--i', type=str, help='number of experiment', required=True)
    parser.add_argument('--seed', type=str, help='', required=True)
    parser.add_argument('--all_conf', type=str, help='name or path of your configuration file', required=True)
    parser.add_argument('--g_lr', type=float, help='generator learning rate', required=False)
    parser.add_argument('--l2alpha', type=float, help='l2 normalization hyper parameter', required=False)
    parser.add_argument('--outpath', type=str, help='after outdir', required=False)
    parser.add_argument('--method_conf', type=str, help='name or path of your configuration file', required=False)
    parser.add_argument('--task_conf', type=str, help='', required=False)
    parser.add_argument('--common2', type=str, help='path to second common config', required=False)
    parser.add_argument('--common3', type=str, help='path to second common config', required=False)
    parser.add_argument('--nep_token', type=str, help='path to second common config', required=False)
    parser.add_argument('--ild_start', type=int, help='when this epoch ends, ild starts.', required=False)
    parser.add_argument('--curriculum', type=str, nargs='+', help='curriculum mode', required=False)
    parser.add_argument('--curriculum_lr', type=float, help='learning rate while curriculum learning', required=False)
    parser.add_argument('--workdir', type=str, help='additonal path before conf[outdir]', required=False)
    args = parser.parse_args()
    return args

def get_conf(args):
    task = args.task
    conf = {}
    with open(args.all_conf, 'r') as f:
        conf.update(yaml.safe_load(f))
    if args.method_conf:
        with open(args.method_conf, 'r') as f:
            conf.update(yaml.safe_load(f))
    if args.task_conf:
        with open(args.task_conf, 'r') as f:
            tc = yaml.safe_load(f)
            conf.update(tc[task])
    if args.common2:
        with open(args.common2, 'r') as f:
            conf.update(yaml.safe_load(f))
    if args.common3:
        with open(args.common3, 'r') as f:
            conf.update(yaml.safe_load(f))
    if args.nep_token:
        with open(args.nep_token, 'r') as f:
            conf.update(yaml.safe_load(f))
    
    return conf
