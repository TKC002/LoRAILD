import torch
import transformers
from transformers import AutoTokenizer
import datasets
from datasets import load_dataset, DatasetDict
import random

import pickle
import os
import traceback

import lib.util.args as myargs

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def add_pad(args, tokenizer):
    if 'Llama-2' in args.tokenizer:
        tokenizer.add_special_tokens({"pad_token":"<pad>"})
    return tokenizer


def preprocess(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)


    raw_datasets = load_dataset('glue', args.task)
    sentence1_key, sentence2_key = task_to_keys[args.task]
    # -----------------------------------------------------------------------
    def preprocess_function(examples):
        # Tokenize the texts
        #logger.info('tokenizing')
        padding = 'max_length'
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        result['labels'] = examples['label']
        return result
    # -----------------------------------------------------------------------
    print('Tokenizing datasets')
    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    os.makedirs('saved_datasets/num_labels/'+args.tokenizer+'/', exist_ok=True)
    os.makedirs('saved_datasets/data_inds/', exist_ok=True)
    os.makedirs('saved_datasets/normal/'+args.tokenizer+'/'+args.task, exist_ok=True)

    # data_inds ha dono you na dataset demo kyoutuu
    inds_path = 'saved_datasets/data_inds/'+args.task+'.pickle'
    if os.path.isfile(inds_path):
        with open(inds_path, 'rb') as f:
            train_inds, valid_inds = pickle.load(f)
    else:
        train_size = raw_datasets["train"].num_rows
        tmp = list(range(train_size))
        random.seed(0)
        random.shuffle(tmp)
        valid_size = train_size // 10
        valid_inds = tmp[:valid_size]
        train_inds = tmp[valid_size:]
        with open(inds_path, 'wb') as f:
            inds = (train_inds, valid_inds)
            pickle.dump(inds, f)

    num_labels_path = 'saved_datasets/num_labels/'+args.task+'.pickle'
    if os.path.isfile(num_labels_path):
        pass
    else:
        if args.task == 'stsb':
            num_labels = 1
        else:
            num_labels = len(raw_datasets["train"].features["label"].names)

        with open(num_labels_path, 'wb') as f:
            pickle.dump(num_labels, f)
    
    train_dataset = processed_datasets['train'].select(train_inds)
    valid_dataset = processed_datasets['train'].select(valid_inds)
    test_dataset = processed_datasets["validation_matched" if args.task == "mnli" else "validation"]

    saved_dataset = DatasetDict()
    saved_dataset['train'] = train_dataset
    saved_dataset['valid'] = valid_dataset
    saved_dataset['test'] = test_dataset


    saved_dataset.save_to_disk('saved_datasets/normal/'+args.tokenizer+'/'+args.task)

    train_size = int(train_dataset.num_rows)
    valid_size = int(valid_dataset.num_rows)
    test_size = int(test_dataset.num_rows)
    print('train size = ', train_size)
    print('valid_size = ', valid_size)
    print('test_size  = ', test_size)
        
        
if __name__ == "__main__":
    args = myargs.prepro_args()
    try:
        preprocess(args)
    except Exception as e:
        dir = '/log/datasets/normal/'+args.task+'/'
        os.makedirs(dir, exist_ok=True)
        with open(dir+'error.log', 'a') as f:
            traceback.print_exc(file=f)
            exit(1)
