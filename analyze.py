import transformers
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, default_data_collator
from datasets import load_dataset, load_from_disk
import traceback
import os
import evaluate
import pickle

import torch
from torch.utils.data import DataLoader

import lib.glue_analyzer as myanalyzer
import lib.util.args as myargs
import lib.models as mymodels
import lib.util.adv_tools as adv_tools
import lib.loraconfig as lc

advs = ['MATEKD', 'DEKD', 'L2DEKD', 'DKD']
dkds = ['DKD']

def do_analyze(args, conf):
    print('!!!!!!!!!!!!!! task: ', args.task, "!!!!!!!!!!!!!!")
    half = conf['half'] if 'half' in conf else False
    tokenizer = AutoTokenizer.from_pretrained(conf['tokenizer'])
    dataset_path = conf['dataset_path']+'/'+args.task
    loaded_dataset = load_from_disk(dataset_path)

    def make_adv_dataset(example):
        result = example
        mask_prob = conf['mask_prob'] if 'mask_prob' in conf else 0.3
        input_ids_permuted, labels_permuted, mask_permuted = adv_tools.mask_tokens(torch.tensor(result['input_ids']).clone(), tokenizer, mask_prob)
        result['input_ids_permuted'] = input_ids_permuted
        result['mask_permuted'] = mask_permuted
        return result


    if conf['method'] in advs:
        # print('maks adv dataset')
        # loaded_dataset = loaded_dataset.map(
        #     make_adv_dataset,
        #     batched=True,
        # )
        if 'g_lr' not in conf:
            if args.g_lr is not None:
                print('g_lr in args')
                conf['g_lr'] = args.g_lr
            else:
                print('no g_lr, exit')
                return

    train_dataset = loaded_dataset['train']
    valid_dataset = loaded_dataset['valid']
    test_dataset = loaded_dataset['test']
    
    data_collator = default_data_collator

    train_dataloader = DataLoader(train_dataset, shuffle=False, collate_fn=data_collator, batch_size=conf['batch_size'])
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, collate_fn=data_collator, batch_size=conf['batch_size'])
    test_dataloader  = DataLoader(test_dataset,  shuffle=False, collate_fn=data_collator, batch_size=conf['batch_size'])
    dataloaders = {'train': train_dataloader, 'valid': valid_dataloader, 'test': test_dataloader}

    labels_path = 'saved_datasets/num_labels/'+args.task+'.pickle'
    if args.workdir:
        labels_path = args.workdir+labels_path

    with open(labels_path, 'rb') as f:
        num_labels = pickle.load(f)
        conf['num_labels'] = num_labels
    
    if half:
        dtype = torch.float16
    else:
        dtype = torch.float32

    if conf['method']=='normal':
        config = AutoConfig.from_pretrained(conf['model'], num_labels=num_labels, finetuning_task=args.task)
        model = AutoModelForSequenceClassification.from_pretrained(conf['model'], config=config)
        analyzer = myanalyzer.NormalAnalyzer(
            conf=conf,
            model=model,
            dataloaders=dataloaders
        )
    elif conf['method']=='LoRAnormal':
        pass
    elif conf['method']=='KD':
        model = mymodels.KDModel(conf=conf, task=args.task, num_labels=num_labels)
        analyzer = myanalyzer.KDAnalyzer(
            conf=conf,
            model=model,
            dataloaders=dataloaders
        )
    elif conf['method']=='LoRAKD':
        model = mymodels.LoRAKDModel(conf=conf, task=args.task, num_labels=num_labels)
        analyzer = myanalyzer.LoRAKDAnalyzer(
            conf=conf,
            model=model,
            dataloaders=dataloaders
        )
    elif conf['method']=='MATEKD':
        pass
    elif conf['method']=='DEKD':
        pass
    elif conf['method']=='L2DEKD':
        pass
    elif conf['method']=='ADKD':
        pass
    elif conf['method']=='DKD':
        pass
    elif conf['method']=='LoRAILD' or conf['method']=='CurriculumLoRAILD':
        model = mymodels.LoRAILDModel(conf=conf, task=args.task, num_labels=num_labels)
        analyzer = myanalyzer.LoRAILDAnalyzer(
            conf=conf,
            model=model,
            dataloaders=dataloaders
        )
    # elif conf['method']=='CurriculumLoRAILD':
    #     pass
    elif conf['method']=='RAILKD' or conf['method']=='CurriculumRAILKD':
        model = mymodels.RAILKDModel(conf=conf, task=args.task, num_labels=num_labels)
        analyzer = myanalyzer.RAILKDAnalyzer(
            conf=conf,
            model=model,
            dataloaders=dataloaders
        )
    elif conf['method']=='RAILKD_l' or conf['method']=='CurriculumRAILKD_l':
        model = mymodels.RAILKD_l_Model(conf=conf, task=args.task, num_labels=num_labels)
        analyzer = myanalyzer.RAILKD_l_Analyzer(
            conf=conf,
            model=model,
            dataloaders=dataloaders
        )
    # elif conf['method']=='CurriculumRAILKD':
    #     pass
    
    analyzer.set_args(args)

    analyzer.analyze()

if __name__ == "__main__":
    args = myargs.parse_args()
    conf = myargs.get_conf(args)
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

    if args.workdir:
        # change workspace
        conf['outdir'] = args.workdir+'/'+conf['outdir']
        conf['dataset_path'] = args.workdir+'/'+conf['dataset_path']

        if 'teacher_peft' in conf:
            # LoRA
            for model in conf['teacher_peft']:
                for task in conf['teacher_peft'][model]:
                    conf['teacher_peft'][model][task] = args.workdir+'/'+conf['teacher_peft'][model][task]
        elif 'teacher' in conf:
            for model in conf['teacher']:
                for task in conf['teacher'][model]:
                    conf['teacher'][model][task] = args.workdir+'/'+conf['teacher'][model][task]
    
    if args.outpath:
        conf['outdir'] = conf['outdir']+'/'+args.outpath+'/'
    
    # print('***************** conf ********************')
    # print(conf)
    # print('***************** args ********************')
    # print(args)
    try:
        do_analyze(args, conf)
    except (ZeroDivisionError, TypeError, ModuleNotFoundError, SyntaxError, IndexError, KeyError, AttributeError) as e:
        dir = conf['outdir']+'/log/'+args.task+'/'
        os.makedirs(dir, exist_ok=True)
        with open(dir+'error.log', 'w') as f:
            traceback.print_exc(file=f)
            exit(1)