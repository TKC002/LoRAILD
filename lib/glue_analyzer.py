import random
import os
import pickle
# import gc

import numpy as np
import pandas as pd
import torch
import accelerate
from accelerate.utils import set_seed
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
import evaluate
from peft import get_peft_model, LoraConfig, TaskType

from .util import adv_tools as adv_tools
from .loraconfig import lora_configs
from .models import *

task_to_metrics = {
    "cola": 'matthews_correlation',
    "mnli": 'accuracy',
    "mrpc": 'f1',
    "qnli": 'accuracy',
    "qqp": 'accuracy',
    "rte": 'accuracy',
    "sst2": 'accuracy',
    "stsb": 'pearson',
    "wnli": 'accuracy',
}

class NormalAnalyzer:
    def __init__(self, conf, model, dataloaders):
        self.model = model
        self.dataloaders = dataloaders
        self.conf = conf
        # output of intermediate layer
        self.intermediate_output = {self.conf['save_mode']: []}
        # for feature to calculate ild loss
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.labels = []
        self.result = {}
        self.distribute = False

    def set_args(self, args):
        self.task = args.task
        self.lr = args.lr
        self.i = args.i
        self.seed = args.seed
        self.is_regression = True if args.task=='stsb' else False
        self.metric = evaluate.load('glue', args.task)
        self.conf['i'] = args.i
        self.conf['lr'] = args.lr
    
    def analyze_loop(self, dataloader, mode):
        print(f'**************{mode} dataloader**************')
        self.mode = mode
        self.model.to('cuda')
        previous = 0
        print(f'number of iteration is {len(dataloader[mode])}')
        for step, batch in enumerate(dataloader[mode]):
            batch = {k: v.to('cuda') for k, v in batch.items()}
            outputs = self.model(**batch, output_hidden_states=True)
            self.hidden_states = outputs.hidden_states[1:]
            self.labels.append(batch['labels'].detach().clone().to('cpu'))
            self.batch_features()
            if step % 100 == 99:
                self.labels = torch.cat(self.labels)
                self.concat_features()
                self.features_save(step, previous)
                previous = step+1
                return
        
        print('*************final saving*****************')
        self.labels = torch.cat(self.labels)
        self.concat_features()
        self.features_save(step, previous)
        self.labels = []
    
    def batch_features(self):
        for k in self.intermediate_output:
            tmp_layer = []
            for layer in self.hidden_states:
                sz = layer.size()
                tmp_layer.append(layer.detach().clone().to('cpu').reshape(sz[0], 1, -1))#(batch, 1, seq*hid)
            self.intermediate_output[k].append(torch.cat(tmp_layer, dim=1))

    def concat_features(self):
        # intermediate_output[k] = [tensor, tensor, ...]
        for k in self.intermediate_output:
            self.intermediate_output[k] = torch.cat(self.intermediate_output[k], dim=0)
            print(f'{k}\'s size of intermediate output is {self.intermediate_output[k].size()}')

        self.result = {'intermediate_output': self.intermediate_output, 'labels': self.labels}
        
        del self.intermediate_output, self.labels
        self.intermediate_output = {self.conf['save_mode']: []}
        self.labels = []

    def features_save(self, step, previous):
        print(f'current step {step}, saving...')
        outdir = os.path.join(self.conf['outdir'], self.task, str(self.lr), str(self.i), self.mode)
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, f'{previous}-{step}.pickle')
        with open(outpath, 'wb') as f:
            pickle.dump(self.result, f)
        
        del self.result
        self.result = {}
        # gc.collect()
        print(f'{previous}-{step} step saving complete!!')
    
    def analyze(self):
        self.analyze_loop(self.dataloaders, 'train')
        self.analyze_loop(self.dataloaders, 'valid')
        self.analyze_loop(self.dataloaders, 'test')

class KDAnalyzer:
    def __init__(self, conf, model:KDModel, dataloaders):
        self.model = model
        self.dataloaders = dataloaders
        self.conf = conf
        # output of intermediate layer
        self.intermediate_output = {'teacher': [], 'student': []}
        # for feature to calculate ild loss
        self.ild_features = {'teacher': [], 'student': []}
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.labels = []
        self.result = {}
        self.distribute = False
    
    def select_layers(self, model):
        pass

    def set_args(self, args):
        self.task = args.task
        self.lr = args.lr
        self.i = args.i
        self.seed = args.seed
        self.is_regression = True if args.task=='stsb' else False
        self.metric = evaluate.load('glue', args.task)
        self.conf['i'] = args.i
        self.conf['lr'] = args.lr
    
    def analyze_loop(self, dataloader, mode):
        print(f'**************{mode} dataloader**************')
        self.mode = mode
        self.model.to('cuda')
        self.select_layers(self.model)
        previous = 0
        print(f'number of iteration is {len(dataloader[mode])}')
        for step, batch in enumerate(dataloader[mode]):
            batch = {k: v.to('cuda') for k, v in batch.items()}
            outputs, loss, losses = self.model(**batch)
            self.labels.append(batch['labels'].detach().clone().to('cpu'))
            self.batch_features()
            if step % 100 == 99:
                self.labels = torch.cat(self.labels)
                self.concat_features()
                self.features_save(step, previous)
                previous = step+1
                return

        print('*************final saving*****************')
        self.labels = torch.cat(self.labels)
        self.concat_features()
        self.features_save(step, previous)
        self.labels = []

    def batch_features(self):
        for k in self.intermediate_output:
            tmp_layer = []
            for layer in self.model.hidden_states[k]:
                sz = layer.size()
                tmp_layer.append(layer.detach().clone().to('cpu').reshape(sz[0], 1, -1))#(batch, 1, seq*hid)
            self.intermediate_output[k].append(torch.cat(tmp_layer, dim=1))

    def concat_features(self):
        # intermediate_output[k] = [tensor, tensor, ...]
        for k in self.intermediate_output:
            self.intermediate_output[k] = torch.cat(self.intermediate_output[k], dim=0)
            print(f'{k}\'s size of intermediate output is {self.intermediate_output[k].size()}')

        self.result = {'intermediate_output': self.intermediate_output, 'labels': self.labels}
        del self.intermediate_output, self.labels
        self.intermediate_output = {'teacher': [], 'student': []}
        self.ild_features = {'teacher': [], 'student': []}
        self.labels = []

    def features_save(self, step, previous):
        print(f'current step {step}, saving...')
        outdir = os.path.join(self.conf['outdir'], self.task, str(self.lr), str(self.i), self.mode)
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, f'{previous}-{step}.pickle')
        with open(outpath, 'wb') as f:
            pickle.dump(self.result, f)
        
        del self.result
        self.result = {}
        # gc.collect()
        print(f'{previous}-{step} step saving complete!!')
        

    def analyze(self):
        self.model.from_check()
        self.analyze_loop(self.dataloaders, 'train')
        self.analyze_loop(self.dataloaders, 'valid')
        self.analyze_loop(self.dataloaders, 'test')
        
class LoRAKDAnalyzer(KDAnalyzer):
    def __init__(self, conf, model:LoRAKDModel, dataloaders):
        super().__init__(conf, model, dataloaders)

    def batch_features(self):
        # ild_feature = {teacher: tensor, student: tensor}
        # tensor.size() = (batch, sequence, intermediate)
        ild_feature = self.model.linear_states
        for k in self.intermediate_output:
            tmp_layer = []
            for layer in self.model.hidden_states[k]:
                sz = layer.size()
                tmp_layer.append(layer.detach().clone().to('cpu').reshape(sz[0], 1, -1))#(batch, 1, seq*hid)
            self.intermediate_output[k].append(torch.cat(tmp_layer, dim=1))
            # [tensor of(batch, sequence*intermediate), ...]
            if self.conf['is_lora'][1] == True: 
                # print(ild_feature)
                self.ild_features[k].append(ild_feature[k].detach().clone().to('cpu'))

    def concat_features(self):
        for k in self.intermediate_output:
            self.intermediate_output[k] = torch.cat(self.intermediate_output[k], dim=0)
            print(f'{k}\'s size of intermediate output is {self.intermediate_output[k].size()}')
            if self.conf['is_lora'][1] == True: # student is lora
                self.ild_features[k] = torch.cat(self.ild_features[k], dim=0)
                print(f'{k}\'s size of ild_feature is {self.ild_features[k].size()}')

        self.result= {'intermediate_output': self.intermediate_output, 'labels': self.labels}
        if self.conf['is_lora'][1] == True: # student is lora
            self.result['ild_features'] = self.ild_features

        del self.intermediate_output, self.ild_features, self.labels
        self.intermediate_output = {'teacher': [], 'student': []}
        self.ild_features = {'teacher': [], 'student': []}
        self.labels = []

class LoRAILDAnalyzer(KDAnalyzer):
    
    def __init__(self, conf, model:LoRAILDModel, dataloaders):
        super().__init__(conf, model, dataloaders)

    def select_layers(self, model:LoRAILDModel):
        if self.distribute:
            layers = list(range(model.module.teacher.config.num_hidden_layers))
            selected = sorted(random.sample(layers, model.module.student.config.num_hidden_layers))
            model.module.set_selected(selected)
        else:
            layers = list(range(model.teacher.config.num_hidden_layers))
            selected = sorted(random.sample(layers, model.student.config.num_hidden_layers))
            model.set_selected(selected)    

    def batch_features(self):
        # intermediate_output = {teacher: tensor, student: tensor}
        # tensor.size() = (batch, sequence, intermediate)
        ild_feature = self.model.linear_states
        for k in self.intermediate_output:
            tmp_layer = []
            for layer in self.model.hidden_states[k]:
                sz = layer.size()
                tmp_layer.append(layer.detach().clone().to('cpu').reshape(sz[0], 1, -1))#(batch, 1, seq*hid)
            self.intermediate_output[k].append(torch.cat(tmp_layer, dim=1))
            
        for l in self.ild_features:
            # LoRAILD: [tensor of(batch, layer*3, sequence*intermediate), ...] 
            # RAILKD:  [tensor of(batch, layer, intermediate), ...]
            self.ild_features[l].append(ild_feature[l].detach().clone().to('cpu'))
    
    def concat_features(self):
        for k in self.intermediate_output:
            self.intermediate_output[k] = torch.cat(self.intermediate_output[k], dim=0)
            print(f'{k}\'s size of intermediate output is {self.intermediate_output[k].size()}')
        for l in self.ild_features:
            self.ild_features[l] = torch.cat(self.ild_features[l], dim=0)
            print(f'{l}\'s size of ild_feature is {self.ild_features[l].size()}')

        self.result= {'intermediate_output': self.intermediate_output, 'ild_features': self.ild_features, 'labels': self.labels}
        del self.intermediate_output, self.ild_features, self.labels
        self.intermediate_output = {'teacher': [], 'student': []}
        self.ild_features = {'teacher': [], 'student': []}
        self.labels = []

class RAILKDAnalyzer(LoRAILDAnalyzer):
    def __init__(self, conf, model:RAILKDModel, dataloaders):
        super().__init__(conf, model, dataloaders)

class RAILKD_l_Analyzer(RAILKDAnalyzer):
    def __init__(self, conf, model:RAILKD_l_Model, dataloaders):
        super().__init__(conf, model, dataloaders)
        self.ild_features = {'teacher': [], 'teacher_all': [], 'student': []}

    def concat_features(self):
        super().concat_features()
        self.ild_features = {'teacher': [], 'teacher_all': [], 'student': []}
