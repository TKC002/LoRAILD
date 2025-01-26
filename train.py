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

advs = ['MATEKD', 'DEKD', 'L2DEKD', 'DKD']
dkds = ['DKD']

def do_train(args, conf):
    half = conf['half'] if 'half' in conf else False

    if conf['method'] in dkds:
        print('find_unused_parameters')
        accelerator = accelerate.Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    elif 'LoRAILD' in conf['method'] and conf['full']:
        print('find_unused_parameters')
        accelerator = accelerate.Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    elif conf['method'] == 'RAILKD_l':
        print('find_unused_parameters')
        accelerator = accelerate.Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    elif 'LoRAKD' in conf['method'] and conf['full']:
        print('find_unused_parameters')
        accelerator = accelerate.Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    else:
        accelerator = accelerate.Accelerator()
    use_neptune = conf['use_neptune'] if 'use_neptune' in conf else True
    if use_neptune:
        if accelerator.is_main_process:
            run = neptune.init_run(
                # project=conf['nep_proj'],
                # api_token=conf['nep_token'],
                mode='offline'
            )
            neptune_params = {'method':conf['nep_method'], 'task' : args.task, 'batch_size' : conf['batch_size']*conf['device_num'], 'lr' : args.lr, 'ex_num': args.i, 'seed':args.seed, 'conf':conf}
            run['parameters'] = neptune_params
            print('neputune is initialized')
            if 'tags' in conf:
                run['sys/tags'].add(conf['tags'])
        else:
            run=None
    tokenizer = AutoTokenizer.from_pretrained(conf['tokenizer'])
    dataset_path = conf['dataset_path']+'/'+args.task
    loaded_dataset = load_from_disk(dataset_path)

    # print(loaded_dataset)

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

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=conf['batch_size'])
    valid_dataloader = DataLoader(valid_dataset, shuffle=True, collate_fn=data_collator, batch_size=conf['batch_size'])
    test_dataloader  = DataLoader(test_dataset,  shuffle=True, collate_fn=data_collator, batch_size=conf['batch_size'])
    dataloaders = (train_dataloader, valid_dataloader, test_dataloader)

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
        trainer = mytrainer.NormalTrainer(
            conf=conf,
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            dataloaders=dataloaders,
            run=run
        )
    elif conf['method']=='LoRAnormal':
        config = AutoConfig.from_pretrained(conf['model'], num_labels=num_labels, finetuning_task=args.task)
        model = AutoModelForSequenceClassification.from_pretrained(conf['model'], config=config, torch_dtype=dtype)
        pad_added = conf['pad_added'] if 'pad_added' in conf else False
        if pad_added:
            model.resize_token_embeddings(len(tokenizer))
            model.config.pad_token_id = tokenizer.pad_token_id
        if accelerator.is_main_process:
            print('parameters dtype: ', model.dtype)
        trainer = mytrainer.LoRANormalTrainer(
            conf=conf,
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            dataloaders=dataloaders,
            run=run,
            task=args.task
        )
    elif conf['method']=='KD':
        model = mymodels.KDModel(conf=conf, task=args.task, num_labels=num_labels)
        trainer = mytrainer.KDTrainer(
            conf=conf,
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            dataloaders=dataloaders,
            run=run
        )
    elif conf['method']=='LoRAKD':
        model = mymodels.LoRAKDModel(conf=conf, task=args.task, num_labels=num_labels)
        trainer = mytrainer.LoRAKDTrainer(
            conf=conf,
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            dataloaders=dataloaders,
            run=run
        )
    elif conf['method']=='MATEKD':
        model = mymodels.MATEKDModel(conf=conf, task=args.task, num_labels=num_labels)
        trainer = mytrainer.MATEKDTrainer(
            conf=conf,
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            dataloaders=dataloaders,
            run=run
        )
    elif conf['method']=='DEKD':
        model = mymodels.DEKDModel(conf=conf, task=args.task, num_labels=num_labels)
        trainer = mytrainer.DEKDTrainer(
            conf=conf,
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            dataloaders=dataloaders,
            run=run
        )
    elif conf['method']=='L2DEKD':
        if 'l2alpha' not in conf:
            if args.l2alpha is not None:
                print('l2alpha in args')
                conf['l2alpha'] = args.l2alpha
            else:
                print('no l2alpha')
        model = mymodels.L2DEKDModel(conf=conf, task=args.task, num_labels=num_labels)
        trainer = mytrainer.L2DEKDTrainer(
            conf=conf,
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            dataloaders=dataloaders,
            run=run
        )
    elif conf['method']=='ADKD':
        model = mymodels.ADKDModel(conf=conf, task=args.task, num_labels=num_labels)
        trainer = mytrainer.ADKDTrainer(
            conf=conf,
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            dataloaders=dataloaders,
            run=run
        )
    elif conf['method']=='DKD':
        model = mymodels.DKDModel(conf=conf, task=args.task, num_labels=num_labels)
        trainer = mytrainer.DKDTrainer(
            conf=conf,
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            dataloaders=dataloaders,
            run=run
        )
    elif conf['method']=='LoRAILD':
        model = mymodels.LoRAILDModel(conf=conf, task=args.task, num_labels=num_labels)
        trainer = mytrainer.LoRAILDTrainer(
            conf=conf,
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            dataloaders=dataloaders,
            run=run
        )
    elif conf['method']=='CurriculumLoRAILD':
        model = mymodels.CurriculumLoRAILDModel(conf=conf, task=args.task, num_labels=num_labels)
        trainer = mytrainer.CurriculumLoRAILDTrainer(
            conf=conf,
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            dataloaders=dataloaders,
            run=run
        )
    elif conf['method']=='RAILKD':
        model = mymodels.RAILKDModel(conf=conf, task=args.task, num_labels=num_labels)
        trainer = mytrainer.RAILKDTrainer(
            conf=conf,
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            dataloaders=dataloaders,
            run=run
        )
    elif conf['method']=='RAILKD_l':
        model = mymodels.RAILKD_l_Model(conf=conf, task=args.task, num_labels=num_labels)
        trainer = mytrainer.RAILKD_l_Trainer(
            conf=conf,
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            dataloaders=dataloaders,
            run=run
        )
    elif conf['method']=='CurriculumRAILKD':
        model = mymodels.CurriculumRAILKDModel(conf=conf, task=args.task, num_labels=num_labels)
        trainer = mytrainer.CurriculumRAILKDTrainer(
            conf=conf,
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            dataloaders=dataloaders,
            run=run
        )
    
    trainer.set_args(args)
    trainer.construct_optimizer()
    trainer.construct_scheduler()
    trainer.prepare()
    print(f'prepared, device = {accelerator.device}')
    # if accelerator.is_main_process:
    #     print('parameters dtype: ', model.dtype)

    trainer.train()
    if use_neptune and accelerator.is_main_process:
        run.stop()

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
    
    print('***************** conf ********************')
    print(conf)
    print('***************** args ********************')
    print(args)
    try:
        do_train(args, conf)
    except (ZeroDivisionError, TypeError, ModuleNotFoundError, SyntaxError, IndexError, KeyError) as e:
        dir = conf['outdir']+'/log/'+args.task+'/'
        os.makedirs(dir, exist_ok=True)
        with open(dir+'error.log', 'w') as f:
            traceback.print_exc(file=f)
            exit(1)