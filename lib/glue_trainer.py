import random
import os

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

class NormalTrainer:
    def __init__(self, conf, accelerator, model, tokenizer, dataloaders, run):
        self.conf = conf
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader, self.valid_dataloader, self.test_dataloader = dataloaders
        self.run = run
        self.use_neptune = conf['use_neptune'] if 'use_neptune' in conf else False
        self.samples_seen = 0
        columns = ['epoch', 'train_loss', 'valid_loss', 'test_loss', 'valid_metrics', 'test_metrics']
        self.df_logs = pd.DataFrame(columns=columns)
        self.df_logs = self.df_logs.set_index('epoch')
        self.distribute = self.conf['device_num']!=1
        self.curriculum = False

    def set_args(self, args):
        self.task = args.task
        self.lr = args.lr
        self.i = args.i
        self.seed = args.seed
        self.is_regression = True if args.task=='stsb' else False
        self.metric = evaluate.load('glue', args.task)

        if self.run is not None:
            neptune_params = {'method':self.conf['nep_method'], 'task' : args.task, 'batch_size' : self.conf['batch_size']*self.conf['device_num'], 'lr' : args.lr, 'ex_num': args.i, 'seed':args.seed, 'conf':self.conf}
            self.run['parameters'] = neptune_params

    # will be overrided
    def construct_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf['wd'],
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)
    
    def construct_scheduler(self):
        self.scheduler = get_scheduler(
            name=self.conf['scheduler_type'],
            optimizer=self.optimizer,
            num_warmup_steps=self.conf['num_warmup_steps'],
            num_training_steps=self.conf['epoches']*len(self.train_dataloader)/self.conf['device_num'],
        )
        
    def prepare(self):
        # print('before', len(self.train_dataloader))
        model, optimizer, train_dataloader, valid_dataloader, test_dataloader, scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.valid_dataloader, self.test_dataloader, self.scheduler
        )
        self.model = model
        self.optimizer=optimizer
        self.train_dataloader=train_dataloader
        self.valid_dataloader=valid_dataloader
        self.test_dataloader=test_dataloader
        self.scheduler=scheduler
        set_seed(int(self.seed))

    # will be overrided
    def log_losses(self, losses, mode):
        # log each step loss
        self.run[mode+'losses/lce'].log(losses[0].item())

    # will be overrided
    def model_check(self):
        if self.distribute:
            return {'model':self.model.module.state_dict()}
        else:
            return {'model':self.model.state_dict()}
    
    def save_check(self, epoch):
        if self.conf['save_check']:
            checkpoint = {
                "epoch" : epoch,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "random": random.getstate(),
                "np_random": np.random.get_state(), # numpy.randomを使用する場合は必要
                "torch": torch.get_rng_state(),
                "torch_random": torch.random.get_rng_state(),
                "cuda_random": torch.cuda.get_rng_state(), # gpuを使用する場合は必要
                "cuda_random_all": torch.cuda.get_rng_state_all(), # 複数gpuを使用する場合は必要
            }
            checkpoint = self.model_check() | checkpoint
            print('saving checkpoint')
            check_dir = self.conf['outdir']+'/checkpoints/'+self.task+'/'+str(self.lr)+'/'+str(self.i)+'/'
            os.makedirs(check_dir, exist_ok=True)
            torch.save(checkpoint, check_dir+'/check.bin')

    def save_teacher(self, epoch=''):
        if self.conf['save_teacher']:
            model_dir = self.conf['outdir']+'/models/'+self.task+'/'+str(self.lr)+'/'+str(self.i)+'/'+str(epoch)
            os.makedirs(model_dir, exist_ok=True)
            save_model = self.accelerator.unwrap_model(self.model)
            save_model.save_pretrained(
                model_dir, is_main_process=self.accelerator.is_main_process, save_function=self.accelerator.save
            )

    def add_batch(self, step, batch, dataloader, logits):
        predictions = logits.argmax(dim=-1) if not self.is_regression else logits.squeeze()
        predictions, references = self.accelerator.gather((predictions, batch["labels"]))
        if self.accelerator.num_processes > 1:
            if step == len(dataloader) - 1:
                predictions = predictions[: len(dataloader.dataset) - self.samples_seen]
                references = references[: len(dataloader.dataset) - self.samples_seen]
            else:
                self.samples_seen += references.shape[0]
        self.metric.add_batch(
            predictions=predictions,
            references=references,
        )

    # will be overrided
    def train_loop(self, epoch):
        self.model.train()
        loss_sum = torch.tensor(0, dtype=torch.float, requires_grad=False).to(self.accelerator.device)

        print(f'epoch: {epoch}, training')
        for step, batch in enumerate(self.train_dataloader):
            with self.accelerator.accumulate(self.model):
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                self.accelerator.backward(outputs.loss)
                self.optimizer.step()
                self.scheduler.step()

                if self.accelerator.is_main_process and self.use_neptune:
                    self.log_losses([outputs.loss.detach().clone()], 'train')

                loss_sum += outputs.loss.detach().clone()

        train_loss = self.accelerator.gather(loss_sum)
        train_loss = torch.sum(train_loss)/(len(self.train_dataloader)*self.conf['device_num'])
        if self.accelerator.is_main_process and self.use_neptune:
            self.run['train_loss'].log(train_loss.item())
        return train_loss

    # will be overrided
    def valid_loop(self, epoch, mode, dataloader):
        self.model.eval()
        self.samples_seen = 0
        loss_sum = torch.tensor(0, dtype=torch.float, requires_grad=False).to(self.accelerator.device)
        if self.accelerator.is_main_process:
            print(f'epoch {epoch} : {mode}')

        for step, batch in enumerate(dataloader):
            with torch.no_grad():
                outputs = self.model(**batch)
                loss_sum += outputs.loss.detach().clone()
                if self.accelerator.is_main_process and self.use_neptune:
                    self.log_losses([outputs.loss.detach().clone()], mode)
            
            self.add_batch(step, batch, dataloader, outputs.logits)
            
        if self.accelerator.is_main_process:
            print(f'integrating {mode} loss')
        
        loss_result = self.accelerator.gather(loss_sum)
        loss_result = torch.sum(loss_result)/(len(dataloader)*self.conf['device_num'])
        metric_result = self.metric.compute()
        metric_result = metric_result[task_to_metrics[self.task]]
        if self.accelerator.is_main_process and self.use_neptune:
            self.run[mode+'_loss'].log(loss_result.item())
            self.run[mode+'_metrics'].log(metric_result)
        return loss_result, metric_result

    def train(self):
        best_metric = 0
        best_epoch = 0
        for epoch in range(self.conf['epoches']):
            # one_epoch_log : keys is 'train_loss', 'valid_loss', 'test_loss', 'valid_metrics', 'test_metrics'
            one_epoch_log = {}
            train_loss = self.train_loop(epoch)
            one_epoch_log['train_loss'] = train_loss.item()

            valid_loss, valid_metric = self.valid_loop(epoch, 'valid', self.valid_dataloader)
            
            one_epoch_log['valid_metrics'] = valid_metric
            if self.curriculum:
                if epoch < self.ild_start:
                    one_epoch_log['valid_metrics'] = 0
                    one_epoch_log['pre_valid_metrics'] = valid_metric
            one_epoch_log['valid_loss'] = valid_loss.item()

            test_loss, test_metric = self.valid_loop(epoch, 'test', self.test_dataloader)

            one_epoch_log['test_metrics'] = test_metric
            if self.curriculum:
                if epoch < self.ild_start:
                    one_epoch_log['valid_metrics'] = 0
                    one_epoch_log['pre_valid_metrics'] = valid_metric
            one_epoch_log['test_loss'] = test_loss.item()

            df_log = pd.DataFrame(one_epoch_log, index=[epoch])

            self.df_logs = pd.concat([self.df_logs, df_log])

            # each epoch save
            self.save_teacher(epoch)

            # for student
            if best_metric < one_epoch_log['valid_metrics']:
                best_epoch = epoch
                best_metric = one_epoch_log['valid_metrics']
                if self.accelerator.is_main_process:
                    self.save_check(epoch)
                    # self.save_teacher()

        print('best epoch = ', best_epoch)
        #for teacher choosing
        # if self.conf['save_last']:
        #     self.save_last()
        
        df_logs_dir = self.conf['outdir']+'/df_logs/'+self.task+'/'+str(self.lr)+'/'
        os.makedirs(df_logs_dir, exist_ok=True)
        self.df_logs.to_csv(df_logs_dir+f'{self.i}.csv')
    
class KDTrainer(NormalTrainer):
    def __init__(self, conf, accelerator, model, tokenizer, dataloaders, run):
        super().__init__(conf, accelerator, model, tokenizer, dataloaders, run)

    #override
    def construct_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.student.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf['wd'],
            },
            {
                "params": [p for n, p in self.model.student.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)
    
    # override
    def log_losses(self, losses, mode):
        # log each step loss
        self.run[mode+'losses/lce'].log(losses[0].item())
        self.run[mode+'losses/lkd'].log(losses[1].item())

    # override
    def model_check(self):
        if self.distribute:
            return {'student':self.model.module.student.state_dict()}
        else:
            return {'student':self.model.student.state_dict()}

    # override
    def train_loop(self, epoch):
        if self.distribute:
            self.model.module.train()
        else:
            self.model.train()
        loss_sum = torch.tensor(0, dtype=torch.float, requires_grad=False).to(self.accelerator.device)

        print(f'epoch: {epoch}, training')
        for step, batch in enumerate(self.train_dataloader):
            with self.accelerator.accumulate(self.model):
                self.optimizer.zero_grad()
                outputs, loss, losses = self.model(**batch)
                if self.accelerator.is_main_process and self.use_neptune:
                    self.log_losses(losses, 'train')
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.scheduler.step()

                loss_sum += loss.detach().clone()

        train_loss = self.accelerator.gather(loss_sum)
        train_loss = torch.sum(train_loss)/(len(self.train_dataloader)*self.conf['device_num'])
        if self.accelerator.is_main_process and self.use_neptune:
            self.run['train_loss'].log(train_loss.item())
        return train_loss
    
    # override
    def valid_loop(self, epoch, mode, dataloader):
        if self.distribute:
            self.model.module.eval()
        else:
            self.model.eval()
        self.samples_seen = 0
        loss_sum = torch.tensor(0, dtype=torch.float, requires_grad=False).to(self.accelerator.device)
        if self.accelerator.is_main_process:
            print(f'epoch {epoch} : {mode}')

        for step, batch in enumerate(dataloader):
            with torch.no_grad():
                outputs, loss, losses = self.model(**batch)
                loss_sum += loss.detach().clone()
                if self.accelerator.is_main_process and self.use_neptune:
                    self.log_losses(losses, mode)
            
            self.add_batch(step, batch, dataloader, outputs.logits)
            
        if self.accelerator.is_main_process:
            print(f'integrating {mode} loss')
        
        loss_result = self.accelerator.gather(loss_sum)
        loss_result = torch.sum(loss_result)/(len(dataloader)*self.conf['device_num'])
        metric_result = self.metric.compute()
        metric_result = metric_result[task_to_metrics[self.task]]
        if self.accelerator.is_main_process and self.use_neptune:
            self.run[mode+'_loss'].log(loss_result.item())
            self.run[mode+'_metrics'].log(metric_result)
        return loss_result, metric_result

# this is base of KD trainer using adversarial training

class LoRANormalTrainer(NormalTrainer):
    def __init__(self, conf, accelerator, model, tokenizer, dataloaders, run, task=None):
        super().__init__(conf, accelerator, model, tokenizer, dataloaders, run)
        self.half = conf['half'] if 'half' in conf else False
        task_type = TaskType.SEQ_CLS
        if 'llama2' in self.conf['model_name']:
            self.lora_config = lora_configs['llama2']
        elif 'roberta' in self.conf['model_name']:
            self.lora_config = lora_configs[self.conf['model_name']]
        self.model = get_peft_model(self.model, self.lora_config)
        if self.accelerator.is_main_process:
            print('trainable parameters')
            self.model.print_trainable_parameters()
            print("**********lora_config************")
            print(self.lora_config)
    
    def construct_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
    
    def save_check(self, epoch):
        # return super().save_check(epoch)
        if self.conf['save_check']:
            checkpoint = {
                "epoch" : epoch,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "random": random.getstate(),
                "np_random": np.random.get_state(), # numpy.randomを使用する場合は必要
                "torch": torch.get_rng_state(),
                "torch_random": torch.random.get_rng_state(),
                "cuda_random": torch.cuda.get_rng_state(), # gpuを使用する場合は必要
                "cuda_random_all": torch.cuda.get_rng_state_all(), # 複数gpuを使用する場合は必要
            }
            print('saving checkpoint')
            check_dir = self.conf['outdir']+'/checkpoints/'+self.task+'/'+str(self.lr)+'/'+str(self.i)+'/'
            os.makedirs(check_dir, exist_ok=True)
            torch.save(checkpoint, check_dir+'/check.bin')
            save_model = self.accelerator.unwrap_model(self.model)
            save_model.save_pretrained(
                check_dir+'/model/',
                is_main_process=self.accelerator.is_main_process,
                save_function=self.accelerator.save
            )

    def save_teacher(self, epoch=''):
        if self.conf['save_teacher']:
            model_dir = self.conf['outdir']+'/models/'+self.task+'/'+str(self.lr)+'/'+str(self.i)+'/'+str(epoch)
            os.makedirs(model_dir, exist_ok=True)
            save_model = self.accelerator.unwrap_model(self.model)
            save_model.save_pretrained(
                model_dir, 
                is_main_process=self.accelerator.is_main_process,
                save_function=self.accelerator.save
            )
    
    # will be overrided
    def train_loop(self, epoch):
        if self.accelerator.is_main_process:
            print(self.accelerator.scaler)
        self.model.train()
        loss_sum = torch.tensor(0, dtype=torch.float, requires_grad=False).to(self.accelerator.device)

        print(f'epoch: {epoch}, training')
        for step, batch in enumerate(self.train_dataloader):
            with self.accelerator.accumulate(self.model):
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                self.accelerator.backward(outputs.loss)
                self.optimizer.step()
                self.scheduler.step()

                if self.accelerator.is_main_process and self.use_neptune:
                    self.log_losses([outputs.loss.detach().clone()], 'train')

                loss_sum += outputs.loss.detach().clone()

        train_loss = self.accelerator.gather(loss_sum)
        train_loss = torch.sum(train_loss)/(len(self.train_dataloader)*self.conf['device_num'])
        if self.accelerator.is_main_process and self.use_neptune:
            self.run['train_loss'].log(train_loss.item())
        return train_loss

class LoRAKDTrainer(KDTrainer):
    def __init__(self, conf, accelerator, model, tokenizer, dataloaders, run):
        super().__init__(conf, accelerator, model, tokenizer, dataloaders, run)

    #override
    def construct_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.model.student.parameters(), lr=self.lr)
    
    def save_check(self, epoch):
        if self.conf['save_check']:
            checkpoint = {
                "epoch" : epoch,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "random": random.getstate(),
                "np_random": np.random.get_state(), # numpy.randomを使用する場合は必要
                "torch": torch.get_rng_state(),
                "torch_random": torch.random.get_rng_state(),
                "cuda_random": torch.cuda.get_rng_state(), # gpuを使用する場合は必要
                "cuda_random_all": torch.cuda.get_rng_state_all(), # 複数gpuを使用する場合は必要
            }
            check_dir = self.conf['outdir']+'/checkpoints/'+self.task+'/'+str(self.lr)+'/'+str(self.i)+'/'
            os.makedirs(check_dir, exist_ok=True)
            if self.conf['is_lora'][1] == True and 'full' in self.conf and self.conf['full']:
                checkpoint = self.model_check() | checkpoint
                print('saving checkpoint, lora full')
                torch.save(checkpoint, check_dir+'/check.bin')
            else:
                if self.conf['is_lora'][1] == True: # student is also lora
                    torch.save(checkpoint, check_dir+'/check.bin')
                    save_model = self.accelerator.unwrap_model(self.model)
                    save_model.student.save_pretrained(check_dir+'/model/')
                else:
                    checkpoint = self.model_check() | checkpoint
                    print('saving checkpoint')
                    torch.save(checkpoint, check_dir+'/check.bin')

class LoRAILDTrainer(LoRAKDTrainer):
    def __init__(self, conf, accelerator, model, tokenizer, dataloaders, run):
        super().__init__(conf, accelerator, model, tokenizer, dataloaders, run) 
    
        # override
    def log_losses(self, losses, mode):
        # log each step loss
        self.run[mode+'losses/lce'].log(losses[0].item())
        self.run[mode+'losses/lkd'].log(losses[1].item())
        self.run[mode+'losses/lild'].log(losses[2].item())

    def save_check(self, epoch):
        if self.conf['save_check']:
            checkpoint = {
                "epoch" : epoch,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "random": random.getstate(),
                "np_random": np.random.get_state(), # numpy.randomを使用する場合は必要
                "torch": torch.get_rng_state(),
                "torch_random": torch.random.get_rng_state(),
                "cuda_random": torch.cuda.get_rng_state(), # gpuを使用する場合は必要
                "cuda_random_all": torch.cuda.get_rng_state_all(), # 複数gpuを使用する場合は必要
            }
            check_dir = self.conf['outdir']+'/checkpoints/'+self.task+'/'+str(self.lr)+'/'+str(self.i)+'/'
            os.makedirs(check_dir, exist_ok=True)
            if 'full' in self.conf and self.conf['full']:
                checkpoint = self.model_check() | checkpoint
                print('saving checkpoint, lora full')
                torch.save(checkpoint, check_dir+'/check.bin')
            else:
                if self.conf['is_lora'][1] != False: # student is also lora
                    print('saving checkpoint, lora')
                    torch.save(checkpoint, check_dir+'/check.bin')
                    save_model = self.accelerator.unwrap_model(self.model)
                    save_model.student.save_pretrained(check_dir+'/model/')
                else:
                    checkpoint = self.model_check() | checkpoint
                    print('saving checkpoint, not lora')
                    torch.save(checkpoint, check_dir+'/check.bin')
    
    def select_layers(self, model):
        if self.distribute:
            layers = list(range(model.module.teacher.config.num_hidden_layers))
            selected = sorted(random.sample(layers, model.module.student.config.num_hidden_layers))
            model.module.set_selected(selected)
        else:
            layers = list(range(model.teacher.config.num_hidden_layers))
            selected = sorted(random.sample(layers, model.student.config.num_hidden_layers))
            model.set_selected(selected)   
    # override
    def train_loop(self, epoch):
        if self.distribute:
            self.model.module.train()
        else:
            self.model.train()
        loss_sum = torch.tensor(0, dtype=torch.float, requires_grad=False).to(self.accelerator.device)
        self.select_layers(self.model)
        print(f'epoch: {epoch}, training')
        for step, batch in enumerate(self.train_dataloader):
            with self.accelerator.accumulate(self.model):
                self.optimizer.zero_grad()
                outputs, loss, losses = self.model(**batch)
                if self.accelerator.is_main_process and self.use_neptune:
                    self.log_losses(losses, 'train')
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.scheduler.step()

                loss_sum += loss.detach().clone()

        train_loss = self.accelerator.gather(loss_sum)
        train_loss = torch.sum(train_loss)/(len(self.train_dataloader)*self.conf['device_num'])
        if self.accelerator.is_main_process and self.use_neptune:
            self.run['train_loss'].log(train_loss.item())
        return train_loss

class CurriculumLoRAILDTrainer(LoRAILDTrainer):
    def __init__(self, conf, accelerator, model, tokenizer, dataloaders, run):
        super().__init__(conf, accelerator, model, tokenizer, dataloaders, run) 
        self.curriculum = True
    
    def set_args(self, args):
        print('setting args for Trainer')
        self.task = args.task
        self.lr = args.lr
        self.i = args.i
        self.seed = args.seed
        self.is_regression = True if args.task=='stsb' else False
        self.metric = evaluate.load('glue', args.task)

        # self.curriculum[0]:
        #   - normal: [0.5, 0.5, 0] -> [0.333, 0.333, 0.333]
        #   - kd_rem: [0.5, 0.5, 0] -> [0.5, 0, 0.5]
        # self.curriculum[1]
        #   - sharp:
        #   - smooth:
        if args.ild_start and args.curriculum:
            self.ild_start = args.ild_start
            self.curriculum = args.curriculum
            self.curriculum_lr = args.curriculum_lr
            print(self.conf['outdir'])
        else:
            print("args.ild_start or args.curriculum is None.")
            exit(1)

        if self.run is not None:
            neptune_params = {'method':self.conf['nep_method'], 'task' : args.task, 'batch_size' : self.conf['batch_size']*self.conf['device_num'], 'lr' : args.lr, 'ex_num': args.i, 'seed':args.seed, 'conf':self.conf}
            self.run['parameters'] = neptune_params

    def set_lambdas(self, epoch):
        k = 0
        n = 0
        initial_lambdas = [0.5, 0.5, 0]
        if self.curriculum[0] == 'normal':
            distination = [0.333, 0.333, 0.333]
        elif self.curriculum[0] == 'kd_rem':
            distination = [0.5, 0, 0.5]

        if epoch < self.ild_start:
            lambdas = initial_lambdas
        else:
            if self.curriculum[1] == 'sharp':
                lambdas = distination
            elif self.curriculum[1] == 'smooth':
                n = self.conf['epoches'] - self.ild_start
                k = min((epoch-self.ild_start+1) / n, 1)

                lambdas = [(1-k)*initial_lambdas[i]+k*distination[i] for i in range(3)]
        
        if self.distribute:
            self.model.module.set_lambdas(lambdas)
        else:
            self.model.set_lambdas(lambdas)
        
        print(f'epoch: {epoch}, lambdas: {lambdas}')

    # override
    def train_loop(self, epoch):
        if self.distribute:
            self.model.module.train()
        else:
            self.model.train()
        loss_sum = torch.tensor(0, dtype=torch.float, requires_grad=False).to(self.accelerator.device)
        self.select_layers(self.model)
        print(f'epoch: {epoch}, training')
        self.set_lambdas(epoch) # <----- this is the difference

        if epoch == self.ild_start:
            for g in self.optimizer.param_groups:
                g['lr'] == self.curriculum_lr

        for step, batch in enumerate(self.train_dataloader):
            with self.accelerator.accumulate(self.model):
                self.optimizer.zero_grad()
                outputs, loss, losses = self.model(**batch)
                if self.accelerator.is_main_process and self.use_neptune:
                    self.log_losses(losses, 'train')
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.scheduler.step()

                loss_sum += loss.detach().clone()

        train_loss = self.accelerator.gather(loss_sum)
        train_loss = torch.sum(train_loss)/(len(self.train_dataloader)*self.conf['device_num'])
        if self.accelerator.is_main_process and self.use_neptune:
            self.run['train_loss'].log(train_loss.item())
        return train_loss
    
class RAILKDTrainer(KDTrainer):
    def __init__(self, conf, accelerator, model, tokenizer, dataloaders, run):
        super().__init__(conf, accelerator, model, tokenizer, dataloaders, run)

    def construct_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = []
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in self.model.W_t.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf['wd'],
            }
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in self.model.W_t.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in self.model.W_s.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf['wd'],
            }
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in self.model.W_s.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in self.model.student.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf['wd'],
            },
            {
                "params": [p for n, p in self.model.student.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)
    
    def select_layers(self, model):
        if self.distribute:
            layers = list(range(model.module.teacher.config.num_hidden_layers))
            selected = sorted(random.sample(layers, model.module.student.config.num_hidden_layers))
            model.module.set_selected(selected)
        else:
            layers = list(range(model.teacher.config.num_hidden_layers))
            selected = sorted(random.sample(layers, model.student.config.num_hidden_layers))
            model.set_selected(selected)            

    # override
    def model_check(self):
        if self.distribute:
            return {'student':self.model.module.student.state_dict(), 'W_t': self.model.module.W_t.state_dict(), 'W_s': self.model.module.W_s.state_dict()}
        else:
            return {'student':self.model.student.state_dict(), 'W_t': self.model.W_t.state_dict(), 'W_s': self.model.W_s.state_dict()}

    # override
    def log_losses(self, losses, mode):
        # log each step loss
        self.run[mode+'losses/lce'].log(losses[0].item())
        self.run[mode+'losses/lkd'].log(losses[1].item())
        self.run[mode+'losses/lild'].log(losses[2].item())
    
    # override
    def train_loop(self, epoch):
        if self.distribute:
            self.model.module.train()
        else:
            self.model.train()
        loss_sum = torch.tensor(0, dtype=torch.float, requires_grad=False).to(self.accelerator.device)
        self.select_layers(self.model)
        print(f'epoch: {epoch}, training')
        for step, batch in enumerate(self.train_dataloader):
            with self.accelerator.accumulate(self.model):
                self.optimizer.zero_grad()
                outputs, loss, losses = self.model(**batch)
                if self.accelerator.is_main_process and self.use_neptune:
                    self.log_losses(losses, 'train')
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.scheduler.step()

                loss_sum += loss.detach().clone()

        train_loss = self.accelerator.gather(loss_sum)
        train_loss = torch.sum(train_loss)/(len(self.train_dataloader)*self.conf['device_num'])
        if self.accelerator.is_main_process and self.use_neptune:
            self.run['train_loss'].log(train_loss.item())
        return train_loss

class RAILKD_l_Trainer(RAILKDTrainer):
    def __init__(self, conf, accelerator, model, tokenizer, dataloaders, run):
        super().__init__(conf, accelerator, model, tokenizer, dataloaders, run)

    def construct_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = []
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in W.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf['wd'],
            } for W in self.model.W_t
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in W.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            } for W in self.model.W_t
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in W.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf['wd'],
            } for W in self.model.W_s
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in W.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            } for W in self.model.W_s
        ]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in self.model.student.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.conf['wd'],
            },
            {
                "params": [p for n, p in self.model.student.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)

class CurriculumRAILKDTrainer(RAILKDTrainer):
    def __init__(self, conf, accelerator, model, tokenizer, dataloaders, run):
        super().__init__(conf, accelerator, model, tokenizer, dataloaders, run)
        self.curriculum = True
    
    def set_args(self, args):
        print('setting args for Trainer')
        self.task = args.task
        self.lr = args.lr
        self.i = args.i
        self.seed = args.seed
        self.is_regression = True if args.task=='stsb' else False
        self.metric = evaluate.load('glue', args.task)

        # self.curriculum[0]:
        #   - normal: [0.5, 0.5, 0] -> [0.333, 0.333, 0.333]
        #   - kd_rem: [0.5, 0.5, 0] -> [0.5, 0, 0.5]
        # self.curriculum[1]
        #   - sharp:
        #   - smooth:
        if args.ild_start and args.curriculum:
            self.ild_start = args.ild_start
            self.curriculum = args.curriculum
            self.curriculum_lr = args.curriculum_lr
            print(self.conf['outdir'])
        else:
            print("args.ild_start or args.curriculum is None.")
            exit(1)

        if self.run is not None:
            neptune_params = {'method':self.conf['nep_method'], 'task' : args.task, 'batch_size' : self.conf['batch_size']*self.conf['device_num'], 'lr' : args.lr, 'ex_num': args.i, 'seed':args.seed, 'conf':self.conf}
            self.run['parameters'] = neptune_params
        
    def set_lambdas(self, epoch):
        k = 0
        n = 0
        initial_lambdas = [0.5, 0.5, 0]
        if self.curriculum[0] == 'normal':
            distination = [0.333, 0.333, 0.333]
        elif self.curriculum[0] == 'kd_rem':
            distination = [0.5, 0, 0.5]

        if epoch < self.ild_start:
            lambdas = initial_lambdas
        else:
            if self.curriculum[1] == 'sharp':
                lambdas = distination
            elif self.curriculum[1] == 'smooth':
                n = self.conf['epoches'] - self.ild_start
                k = min((epoch-self.ild_start+1) / n, 1)

                lambdas = [(1-k)*initial_lambdas[i]+k*distination[i] for i in range(3)]
    
        if self.distribute:
            self.model.module.set_lambdas(lambdas)
        else:
            self.model.set_lambdas(lambdas)
        
        print(f'epoch: {epoch}, lambdas: {lambdas}')

    def train_loop(self, epoch):
        if self.distribute:
            self.model.module.train()
        else:
            self.model.train()
        loss_sum = torch.tensor(0, dtype=torch.float, requires_grad=False).to(self.accelerator.device)
        self.select_layers(self.model)
        print(f'epoch: {epoch}, training')
        self.set_lambdas(epoch) # <----- this is the difference

        if epoch == self.ild_start:
            for g in self.optimizer.param_groups:
                g['lr'] == self.curriculum_lr

        for step, batch in enumerate(self.train_dataloader):
            with self.accelerator.accumulate(self.model):
                self.optimizer.zero_grad()
                outputs, loss, losses = self.model(**batch)
                if self.accelerator.is_main_process and self.use_neptune:
                    self.log_losses(losses, 'train')
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.scheduler.step()

                loss_sum += loss.detach().clone()

        train_loss = self.accelerator.gather(loss_sum)
        train_loss = torch.sum(train_loss)/(len(self.train_dataloader)*self.conf['device_num'])
        if self.accelerator.is_main_process and self.use_neptune:
            self.run['train_loss'].log(train_loss.item())
        return train_loss
