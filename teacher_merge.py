import accelerate
from accelerate import DistributedDataParallelKwargs
import transformers
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, default_data_collator
from datasets import load_dataset, load_from_disk
import traceback
import os
import evaluate
import pickle

import torch
from torch.utils.data import DataLoader
import neptune

import lib.glue_trainer as mytrainer
import lib.util.args as myargs
import lib.models as mymodels
import lib.util.adv_tools as adv_tools
import lib.loraconfig as lc
from lib.util.conf import *
from peft import get_peft_model, LoraConfig, TaskType, PeftModelForSequenceClassification



def do_merge(args, conf):
    num_labels=2 if args.task != "stsb" else 1
    model_name = conf['model_name']
    t_name = conf_arg(conf, 't_name', model_name)
    teacher_peft_path = conf['teacher_peft'][t_name][args.task]
    teacher_path = conf['teacher'][t_name]
    outdir = teacher_peft_path.replace('LoRA', 'Merged')
    if args.outpath:
        outdir = outdir+'/'+args.outpath+'/'

    if outdir == teacher_peft_path:
        exit(1)
    print(outdir)
    os.makedirs(outdir, exist_ok=True)

    tconfig = AutoConfig.from_pretrained(teacher_path, num_labels=num_labels, finetuning_task=args.task)
    teacher = AutoModelForSequenceClassification.from_pretrained(teacher_path, config=tconfig)
    teacher = PeftModelForSequenceClassification.from_pretrained(teacher, teacher_peft_path)
    merged_teacher = teacher.merge_and_unload()
    merged_teacher.save_pretrained(outdir)



if __name__ == "__main__":
    args = myargs.parse_args()
    conf = myargs.get_conf(args)

    if args.workdir:
        # change workspace

        if 'teacher_peft' in conf:
            # LoRA
            for model in conf['teacher_peft']:
                for task in conf['teacher_peft'][model]:
                    conf['teacher_peft'][model][task] = args.workdir+'/'+conf['teacher_peft'][model][task]
        elif 'teacher' in conf:
            for model in conf['teacher']:
                for task in conf['teacher'][model]:
                    conf['teacher'][model][task] = args.workdir+'/'+conf['teacher'][model][task]
    
    
    # print('***************** conf ********************')
    # print(conf)
    # print('***************** args ********************')
    # print(args)
    do_merge(args, conf)