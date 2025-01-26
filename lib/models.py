from typing import List, Optional, Tuple, Union
import os
import random

import torch
from torch import nn
import transformers
from transformers import AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoConfig, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, PeftModelForSequenceClassification

from . import functional as myfunc
from .loraconfig import lora_configs

class KDModel(nn.Module):
    def __init__(self, conf, task, num_labels):
        super(KDModel, self).__init__()

        self.model_name = conf['model_name']# if teacher and student is different, this is student model name
        self.t_name = conf['t_name'] if 't_name' in conf else self.model_name
        teacher_path = conf['teacher'][self.t_name][task]
        tconfig = AutoConfig.from_pretrained(teacher_path, num_labels=num_labels, finetuning_task=task)
        sconfig = AutoConfig.from_pretrained(conf['student'], num_labels=num_labels, finetuning_task=task)
        print('teacher_path:', teacher_path)
        self.T = conf['T'] if 'T' in conf else 1.0

        self.conf=conf
        self.task=task

        self.num_labels = num_labels

        if 'lambdas' in conf:
            self.lambdas = conf['lambdas']
        else:
            self.lambdas = [0.5, 0.5]
        
        self.student = AutoModelForSequenceClassification.from_pretrained(
            conf['student'],
            config=sconfig
        )
        self.teacher = AutoModelForSequenceClassification.from_pretrained(
            teacher_path,
            config=tconfig
        )
        # teacher freeze
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.hidden_states = {}
    
    def normal_check(self, check_path):
        checkpoint = torch.load(check_path+'/check.bin', map_location='cpu')
        # because of version
        invalid_keys = ['roberta.embeddings.position_ids']
        for i in invalid_keys:
            if i in checkpoint['student']:
                checkpoint['student'].pop(i)
        self.student.load_state_dict(checkpoint['student'])

    # both teacher and student is trained and no further training is needed
    def from_check(self):
        print('***************fron check*******************')
        check_path = os.path.join(self.conf['checkpoint'], self.task, str(self.conf['lr']), str(self.conf['i']))
        self.normal_check(check_path)

        for param in self.student.parameters():
            param.requires_grad = False
        print('********************************************')

    def train(self):
        self.teacher.eval()
        self.student.train()
    
    def eval(self):
        self.teacher.eval()
        self.student.eval()

    def loss_kd(self, teacher_logit, student_logit):
        if self.task == 'stsb':
            loss = torch.nn.functional.mse_loss(teacher_logit, student_logit)
        else:
            tp = nn.functional.softmax(teacher_logit, dim=1)
            sp = nn.functional.softmax(student_logit, dim=1)
            tp = tp/self.T
            sp = sp/self.T
            loss = myfunc.kl_div(tp, sp)
        return loss
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        soft_label = self.teacher(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, output_hidden_states=True)
        outputs = self.student(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        lkd = self.loss_kd(soft_label.logits, outputs.logits)
        loss = self.lambdas[0]*outputs.loss+self.lambdas[1]*lkd
        losses = [outputs.loss.detach().clone(), lkd.detach().clone()]

        # for analyzer
        self.hidden_states = {}
        self.hidden_states['teacher'] = soft_label.hidden_states[1:]
        self.hidden_states['student'] = outputs.hidden_states[1:]
        return outputs, loss, losses

class LoRAKDModel(nn.Module):
    def __init__(self, conf, task, num_labels):
        # sequence: 
        #   1. load pretrained model
        #   2. add padding token if there is no pad
        #   3. load peft model

        super(LoRAKDModel, self).__init__()

        self.T = conf['T'] if 'T' in conf else 1.0
        self.conf=conf
        self.task=task
        self.num_labels = num_labels
        if 'lambdas' in conf:
            self.lambdas = conf['lambdas']
        else:
            self.lambdas = [0.5, 0.5]
        # 1 ---------------------------------------------------------
        self.model_name = conf['model_name']# if teacher and student is different, this is student model name
        self.t_name = conf['t_name'] if 't_name' in conf else self.model_name
        teacher_peft_path = conf['teacher_peft'][self.t_name][task]
        tconfig = AutoConfig.from_pretrained(conf['teacher'][self.t_name], num_labels=num_labels, finetuning_task=task)
        sconfig = AutoConfig.from_pretrained(conf['student'], num_labels=num_labels, finetuning_task=task)
        
        self.student = AutoModelForSequenceClassification.from_pretrained(
            conf['student'],
            config=sconfig
        )
        self.teacher = AutoModelForSequenceClassification.from_pretrained(
            conf['teacher'][self.t_name],
            config=tconfig
        )
        # ----------------------------------------------------------

        # self.half = conf['half'] if 'half' in conf else False
        pad_added = conf['pad_added'] if 'pad_added' in conf else [False, False] # pad does sometimes not exist.
        # 2 ---------------------------------------------------------
        t_tokenizer = AutoTokenizer.from_pretrained(conf['t_tokenizer'])
        tokenizer = AutoTokenizer.from_pretrained(conf['tokenizer'])
        if pad_added[0] != False: # teacher
            self.teacher.resize_token_embeddings(len(t_tokenizer))
            self.teacher.config.pad_token_id = t_tokenizer.pad_token_id
        if pad_added[1] != False: # student
            self.student.resize_token_embeddings(len(tokenizer))
            self.student.config.pad_token_id = tokenizer.pad_token_id
        # -------------------------------------------------------------
        if 'llama2' in self.t_name:
            self.tlora_config = lora_configs['llama2']
        elif 'roberta' in self.t_name:
            self.tlora_config = lora_configs[self.t_name]

        print('loading teacher peft model')
        self.teacher = PeftModelForSequenceClassification.from_pretrained(self.teacher, teacher_peft_path)

        if self.conf['is_lora'][1] == True: # student is also lora
            if 'llama2' in conf['model_name']:
                self.slora_config = lora_configs['llama2']
            elif 'roberta' in conf['model_name']:
                self.slora_config = lora_configs[self.model_name]
            else:
                print('there are no config for', conf['model_name'])
                exit(1)
            print('constructing student peft model')
            self.student = get_peft_model(self.student, self.slora_config)
            print('trainable parameters')
            self.student.print_trainable_parameters()
            # full para tuning
            if 'full' in conf and conf['full']:
                for param in self.student.parameters():
                    param.requires_grad = True
                print('full trainable parameters')
                self.student.print_trainable_parameters()
        print(teacher_peft_path)
        for param in self.teacher.parameters():
            param.requires_grad = False

        # for analyzer
        self.target_modules = self.tlora_config.target_modules 
        self.lora_features = {'teacher':{k:[] for k in self.target_modules}, 'student': {k:[] for k in self.target_modules}}
        self.hidden_states = {}
        self.linear_states = {}

    # for analyzer
    def extract_lora_features(self):
        sz = self.lora_features['teacher']['query'][0].shape
        device = self.lora_features['teacher']['query'][0].device
        t_cat_feature = torch.empty(size = (sz[0], sz[1], 0), device=device)
        s_cat_feature = torch.empty(size = (sz[0], sz[1], 0), device=device)
        for k in self.target_modules:
            bs = t_cat_feature.shape[0]
            length = t_cat_feature.shape[1]
            for layer in range(len(self.lora_features['teacher'][k])):
                t_cat_feature = torch.cat((t_cat_feature, self.lora_features['teacher'][k][layer]), dim=2)
            
            for layer in range(len(self.lora_features['student'][k])):
                s_cat_feature = torch.cat((s_cat_feature, self.lora_features['student'][k][layer]), dim=2)

        t_tmp = t_cat_feature.detach().clone().reshape(bs, length, -1, self.tlora_config.r)
        t_tmp = t_tmp.permute(0, 2, 1, 3)
        s_tmp = s_cat_feature.detach().clone().reshape(bs, length, -1, self.slora_config.r)
        s_tmp = s_tmp.permute(0, 2, 1, 3)
                
        self.linear_states['teacher'] = t_tmp.reshape(bs, -1, length*self.tlora_config.r)
        self.linear_states['student'] = s_tmp.reshape(bs, -1, length*self.slora_config.r)

    def normal_check(self, check_path):
        checkpoint = torch.load(check_path+'/check.bin', map_location='cpu')
        self.student.load_state_dict(checkpoint['student'])

    def lora_check(self, check_path):
        check_path = os.path.join(check_path, 'model')
        del self.student
        sconfig = AutoConfig.from_pretrained(self.conf['student'], num_labels=self.num_labels, finetuning_task=self.task)
        self.student = AutoModelForSequenceClassification.from_pretrained(
            self.conf['student'],
            config=sconfig
        )
        self.student = PeftModelForSequenceClassification.from_pretrained(self.student, check_path)

    def from_check(self):
        print('***************fron check*******************')
        check_path = os.path.join(self.conf['checkpoint'], self.task, str(self.conf['lr']), str(self.conf['i']))
        if self.conf['is_lora'][1] == False or self.conf['full']: # student is not lora or full
            self.normal_check(check_path)
        else: # student is also lora
            self.lora_check(check_path)

        for param in self.student.parameters():
            param.requires_grad = False
        
        print('********************************************')
        if self.conf['is_lora'][1] == True:
            print('trainable parameters')
            self.student.print_trainable_parameters()
        
        def teacher_q_hook(module, input, output):
            self.lora_features['teacher']['query'].append(output)
        def teacher_k_hook(module, input, output):
            self.lora_features['teacher']['key'].append(output)
        def teacher_v_hook(module, input, output):
            self.lora_features['teacher']['value'].append(output)
        
        def student_q_hook(module, input, output):
            self.lora_features['student']['query'].append(output)
        def student_k_hook(module, input, output):
            self.lora_features['student']['key'].append(output)
        def student_v_hook(module, input, output):
            self.lora_features['student']['value'].append(output)
        
        if self.conf['is_lora'][1] == True: # student is also lora
            for layer in self.teacher.base_model.model.roberta.encoder.layer:
                layer.attention.self.query.lora_A.default.register_forward_hook(teacher_q_hook)
                layer.attention.self.key.lora_A.default.register_forward_hook(teacher_k_hook)
                layer.attention.self.value.lora_A.default.register_forward_hook(teacher_v_hook)

            for layer in self.student.base_model.model.roberta.encoder.layer:
                layer.attention.self.query.lora_A.default.register_forward_hook(student_q_hook)
                layer.attention.self.key.lora_A.default.register_forward_hook(student_k_hook)
                layer.attention.self.value.lora_A.default.register_forward_hook(student_v_hook)

    def train(self):
        self.teacher.eval()
        self.student.train()
    
    def eval(self):
        self.teacher.eval()
        self.student.eval()

    def loss_kd(self, teacher_logit, student_logit):
        if self.task == 'stsb':
            loss = torch.nn.functional.mse_loss(teacher_logit, student_logit)
        else:
            tp = nn.functional.softmax(teacher_logit, dim=1)
            sp = nn.functional.softmax(student_logit, dim=1)
            tp = tp/self.T
            sp = sp/self.T
            loss = myfunc.kl_div(tp, sp)
        return loss
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        teacher_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        teacher_attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        teacher_token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        self.lora_features = {'teacher':{k:[] for k in self.target_modules}, 'student': {k:[] for k in self.target_modules}}
        self.hidden_states = {}
        self.linear_states = {}

        if teacher_input_ids is not None:
            # teacher and student is not same model
            if teacher_token_type_ids is None:
                # teacher.forward() does not have argument token_type_ids
                soft_label = self.teacher(input_ids=teacher_input_ids, attention_mask=teacher_attention_mask, labels=labels, output_hidden_states=True)
            else:
                soft_label = self.teacher(input_ids=teacher_input_ids, attention_mask=teacher_attention_mask, token_type_ids = teacher_token_type_ids, labels=labels, output_hidden_states=True)
        else:
            soft_label = self.teacher(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, output_hidden_states=True)
        outputs = self.student(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        lkd = self.loss_kd(soft_label.logits, outputs.logits)
        loss = self.lambdas[0]*outputs.loss+self.lambdas[1]*lkd
        losses = [outputs.loss.detach().clone(), lkd.detach().clone()]

        # for analyzer
        if 'analyze' in self.conf and self.conf['analyze']:
            if self.conf['is_lora'][1] == True:
                self.extract_lora_features()
            self.hidden_states['teacher'] = soft_label.hidden_states[1:]
            self.hidden_states['student'] = outputs.hidden_states[1:]

        return outputs, loss, losses

## only for LoRA-LoRA KD and roberta architecture
class LoRAILDModel(LoRAKDModel):
    def __init__(self, conf, task, num_labels):
        super().__init__(conf, task, num_labels)
        # only for case that tlora_config and slora_config is the same.
        # it is something like {'key', 'query', 'value'}
        self.target_modules = self.tlora_config.target_modules 
        self.lora_features = {'teacher':{k:[] for k in self.target_modules}, 'student': {k:[] for k in self.target_modules}}

        # for analyzer
        self.hidden_states = {}
        self.linear_states = {}

        self.ild_mode = conf['ild_mode'] if 'ild_mode' in conf else 'random'
        self.pretrain_adapter = conf['pretrain_adapter'] if 'pretrain_adapter' in conf else False
        self.pretrain_step = conf['pretrain_step'] if 'pretrain_step' in conf else 20
        self.step_counter = 0
        self.pooling = conf['pooling'] if 'pooling' in conf else False
    
        def teacher_q_hook(module, input, output):
            self.lora_features['teacher']['query'].append(output)
        def teacher_k_hook(module, input, output):
            self.lora_features['teacher']['key'].append(output)
        def teacher_v_hook(module, input, output):
            self.lora_features['teacher']['value'].append(output)
        
        def student_q_hook(module, input, output):
            self.lora_features['student']['query'].append(output)
        def student_k_hook(module, input, output):
            self.lora_features['student']['key'].append(output)
        def student_v_hook(module, input, output):
            self.lora_features['student']['value'].append(output)
        
        for layer in self.teacher.base_model.model.roberta.encoder.layer:
            layer.attention.self.query.lora_A.default.register_forward_hook(teacher_q_hook)
            layer.attention.self.key.lora_A.default.register_forward_hook(teacher_k_hook)
            layer.attention.self.value.lora_A.default.register_forward_hook(teacher_v_hook)

        for layer in self.student.base_model.model.roberta.encoder.layer:
            layer.attention.self.query.lora_A.default.register_forward_hook(student_q_hook)
            layer.attention.self.key.lora_A.default.register_forward_hook(student_k_hook)
            layer.attention.self.value.lora_A.default.register_forward_hook(student_v_hook)
        
        # full para tuning
        if 'full' in conf and conf['full']:
            for param in self.student.parameters():
                param.requires_grad = True
            print('full trainable parameters')
            self.student.print_trainable_parameters()

    def from_check(self):
        print('***************fron check*******************')
        check_path = os.path.join(self.conf['checkpoint'], self.task, str(self.conf['lr']), str(self.conf['i']))
        if self.conf['full']:
            self.normal_check(check_path)
        else:
            self.lora_check(check_path)

        for param in self.student.parameters():
            param.requires_grad = False
        print('********************************************')
        print('trainable parameters')
        self.student.print_trainable_parameters()

    def set_selected(self, selected):
        self.selected = selected

    def loss_ild_calc(self, t_cat_feature, s_cat_feature):
        dim=2
        # if self.pooling:
        #     dim=1
        #     # shape is (batch size, sequence length, intermadiate) -> (batch size, intermadiate)
        #     t_cat_feature = torch.mean(t_cat_feature, dim=1)
        #     s_cat_feature = torch.mean(s_cat_feature, dim=1)
        #     # print('mean pooling.....')

        if self.conf['regularize']==True:
            # regularize
            r_t = t_cat_feature / torch.linalg.norm(t_cat_feature, dim=dim).unsqueeze(dim=dim)
            r_s = s_cat_feature / torch.linalg.norm(s_cat_feature, dim=dim).unsqueeze(dim=dim)
            # each l2 norm ^ 2, in other words, MSE
            lild = torch.pow(torch.linalg.norm(r_t-r_s, dim=dim), 2)
            lild = torch.mean(lild)

        else:
            r_t = t_cat_feature
            r_s = s_cat_feature
            # shape is (batch size, length of sequence)
            lild = torch.linalg.norm(r_t-r_s, dim=dim)
            lild = torch.mean(lild)
        
        # for analyzer
        bs = r_t.shape[0]
        length = r_t.shape[1]
        # (batch, length, layer*linear) -> (batch, length, layer, linear)
        t_tmp = r_t.detach().clone().reshape(bs, length, -1, self.tlora_config.r)
        s_tmp = r_s.detach().clone().reshape(bs, length, -1, self.slora_config.r)

        # (batch, length, layer, linear) -> (batch, layer, length, linear)
        t_tmp = t_tmp.permute(0, 2, 1, 3)
        s_tmp = s_tmp.permute(0, 2, 1, 3)

        if self.ild_mode != 'fixed':
            self.linear_states['teacher'] = t_tmp.reshape(bs, -1, length*self.tlora_config.r)
        self.linear_states['student'] = s_tmp.reshape(bs, -1, length*self.slora_config.r)
        return lild

    # for analyzer
    def fixed_linear_states(self):
        sz = self.lora_features['teacher']['query'][0].shape
        device = self.lora_features['teacher']['query'][0].device
        t_cat_feature = torch.empty(size = (sz[0], sz[1], 0), device=device)
        for k in self.target_modules:
            for layer in range(len(self.lora_features['teacher'][k])):
                t_cat_feature = torch.cat((t_cat_feature, self.lora_features['teacher'][k][layer]), dim=2)
                bs = t_cat_feature.shape[0]
                length = t_cat_feature.shape[1]
        t_tmp = t_cat_feature.detach().clone().reshape(bs, length, -1, self.tlora_config.r)
        t_tmp = t_tmp.permute(0, 2, 1, 3)
        # size (bs, layer*3, ild)
        self.linear_states['teacher'] = t_tmp.reshape(bs, -1, length*self.tlora_config.r)

    def loss_ild(self, teacher_layers, mode):
        sz = self.lora_features['teacher']['query'][0].shape
        device = self.lora_features['teacher']['query'][0].device
        t_cat_feature = torch.empty(size = (sz[0], sz[1], 0), device=device)
        s_cat_feature = torch.empty(size = (sz[0], sz[1], 0), device=device)
        if mode=='random' or mode=='random_epoch':
            for k in self.target_modules:
                for i, l in enumerate(teacher_layers):
                    t_cat_feature = torch.cat((t_cat_feature, self.lora_features['teacher'][k][l]), dim=2)
                    s_cat_feature = torch.cat((s_cat_feature, self.lora_features['student'][k][i]), dim=2)
            
            # # shape is (batch size, length of sequence)
            # lild = torch.linalg.norm(t_cat_feature-s_cat_feature, dim=2)
            # lild = torch.mean(lild)
            lild = self.loss_ild_calc(t_cat_feature, s_cat_feature)
        elif mode=='fixed':
            for k in self.target_modules:
                for i, l in teacher_layers.items():
                    t_cat_feature = torch.cat((t_cat_feature, self.lora_features['teacher'][k][l]), dim=2)
                    s_cat_feature = torch.cat((s_cat_feature, self.lora_features['student'][k][i]), dim=2)
                    
            lild = self.loss_ild_calc(t_cat_feature, s_cat_feature)
            self.fixed_linear_states()

        elif mode=='average':
            for k in self.target_modules:
                for i, ls in enumerate(teacher_layers):
                    t_feacher = torch.zeros_like(self.lora_features['teacher'][k][ls[0]])
                    for l in ls:
                        t_feacher = t_feacher + self.lora_features['teacher'][k][l]
                    t_feacher = t_feacher / len(ls)
                    t_cat_feature = torch.cat((t_cat_feature, t_feacher), dim=2)
                    s_cat_feature = torch.cat((s_cat_feature, self.lora_features['student'][k][i]), dim=2)
            lild = self.loss_ild_calc(t_cat_feature, s_cat_feature)
        return lild

    def forward(self,
        input_ids: Optional[torch.Tensor] = None,
        teacher_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        teacher_attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        teacher_token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        self.lora_features = {'teacher':{k:[] for k in self.target_modules}, 'student': {k:[] for k in self.target_modules}}
        self.hidden_states = {}
        self.linear_states = {}
        if teacher_input_ids is not None:
            # teacher and student is not same model
            if teacher_token_type_ids is None:
                # teacher.forward() does not have argument token_type_ids
                soft_label = self.teacher(input_ids=teacher_input_ids, attention_mask=teacher_attention_mask, labels=labels, output_hidden_states=True)
            else:
                soft_label = self.teacher(input_ids=teacher_input_ids, attention_mask=teacher_attention_mask, token_type_ids = teacher_token_type_ids, labels=labels, output_hidden_states=True)
        else:
            soft_label = self.teacher(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, output_hidden_states=True)

        outputs = self.student(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        lkd = self.loss_kd(soft_label.logits, outputs.logits)

        # for analyzer
        self.hidden_states['teacher'] = soft_label.hidden_states[1:]
        self.hidden_states['student'] = outputs.hidden_states[1:]

        # ----- ILD process --------------------------------------------- #
        if self.ild_mode == 'average':
            teacher_layers = self.conf['teacher_layers'] # matrix teacher_layers[0] is layers for student 0th layer 
        else:
            if self.ild_mode == 'random':
                num_teacher_layers = self.teacher.config.num_hidden_layers
                num_student_layers = self.student.config.num_hidden_layers
                teacher_layers = sorted(random.sample(range(num_teacher_layers), num_student_layers))
            elif self.ild_mode == 'fixed':
                teacher_layers = self.conf['teacher_layers']
            elif self.ild_mode == 'random_epoch':
                teacher_layers = self.selected
        
        lild = self.loss_ild(teacher_layers=teacher_layers, mode=self.ild_mode)

        if self.pretrain_adapter and self.step_counter < self.pretrain_step:
            lambda0 = 0
            lambda1 = 0
        else:
            if self.pretrain_adapter and self.step_counter == self.pretrain_step:
                print('entering main training')
            lambda0 = self.lambdas[0]
            lambda1 = self.lambdas[1]
        loss = lambda0*outputs.loss+lambda1*lkd+self.lambdas[2]*lild
        losses = [outputs.loss.detach().clone(), lkd.detach().clone(), lild.detach().clone()]
        
        self.step_counter += 1
        return outputs, loss, losses

class CurriculumLoRAILDModel(LoRAILDModel):
    def __init__(self, conf, task, num_labels):
        super().__init__(conf, task, num_labels)
    
    def set_lambdas(self, lambdas):
        self.lambdas = lambdas

class RAILKDModel(KDModel):
    def __init__(self, conf, task, num_labels):
        super().__init__(conf, task, num_labels)
        self.W_t = torch.nn.Linear(in_features=self.teacher.config.hidden_size*self.student.config.num_hidden_layers, out_features=conf['linear'])
        self.W_s = torch.nn.Linear(in_features=self.student.config.hidden_size*self.student.config.num_hidden_layers, out_features=conf['linear'])

        # for analyzer
        self.hidden_states = {}
        self.linear_states = {}

        if 'fixed_linear' in conf and conf['fixed_linear']:
            print('***** no ILD mapping training *****')
            for param in self.W_t.parameters():
                param.requires_grad = False
            for param in self.W_s.parameters():
                param.requires_grad = False

    def from_check(self):
        print('***************fron check*******************')
        check_path = os.path.join(self.conf['checkpoint'], self.task, str(self.conf['lr']), str(self.conf['i']))
        checkpoint = torch.load(check_path+'/check.bin', map_location='cpu')
        self.student.load_state_dict(checkpoint['student'])
        self.W_t.load_state_dict(checkpoint['W_t'])
        self.W_s.load_state_dict(checkpoint['W_s'])

        for param in self.student.parameters():
            param.requires_grad = False
        for param in self.W_t.parameters():
            param.requires_grad = False
        for param in self.W_s.parameters():
            param.requires_grad = False
        print('********************************************')

    def mean_pooling(self, hidden_states):
        # shape (batch_size, sequence_length, hidden_size) -> (batch_size, hidden_size)
        return torch.mean(hidden_states, 1)
    
    def set_selected(self, selected):
        self.selected = selected

    def RAIL_c_loss(self, soft_label, outputs):
        # tuple of tensor
        # hs_*[0] is embedding layer output
        # hs_* shape (batch_size, sequence_length, hidden_size)
        # h_bar_* shape (batch_size, hidden_size)

        # for analyzer
        self.hidden_states = {}
        self.linear_states = {}

        hs_t = soft_label.hidden_states[1:]
        hs_s = outputs.hidden_states[1:]
        h_bar_t = list(map(self.mean_pooling, hs_t))
        h_bar_s = list(map(self.mean_pooling, hs_s))
        # if self.sent_pool:
        #     h_bar_t = list(map(self.sent_mean_pooling, hs_t))
        #     h_bar_s = list(map(self.sent_mean_pooling, hs_s))
        # else:
        #     h_bar_t = list(map(self.mean_pooling, hs_t))
        #     h_bar_s = list(map(self.mean_pooling, hs_s))

        # shape is (batch size, len(selected)*hidden_size)
        h_bar_t_c = torch.empty(0, device=soft_label.logits.device)
        h_bar_s_c = torch.empty(0, device=soft_label.logits.device)
        for i, s in enumerate(self.selected):
            h_bar_t_c = torch.cat([h_bar_t_c, h_bar_t[s]], dim=1)
            h_bar_s_c = torch.cat([h_bar_s_c, h_bar_s[i]], dim=1)

        # assert h_bar_t_c.shape == torch.Size([self.conf['batch_size'], len(self.selected)*self.teacher.config.hidden_size]), f'h_bar_t_c size is not correct, size{h_bar_t_c.shape}'
        # assert h_bar_s_c.shape == torch.Size([self.conf['batch_size'], len(self.selected)*self.student.config.hidden_size]), f'h_bar_s_c size is not correct, size{h_bar_t_s.shape}'

        # shape is (batch_size, conf['linear'])
        h_hat_t = self.W_t(h_bar_t_c)
        h_hat_s = self.W_s(h_bar_s_c)

        # for analyzer
        self.hidden_states['teacher'] = hs_t
        self.hidden_states['student'] = hs_s
        self.linear_states['teacher'] = h_hat_t
        self.linear_states['student'] = h_hat_s

        # shape is (batch_size, 1)
        deno_t = torch.linalg.norm(h_hat_t, dim=1).reshape(-1,1)
        deno_s = torch.linalg.norm(h_hat_s, dim=1).reshape(-1,1)

        return torch.linalg.norm(h_hat_t/deno_t - h_hat_s/deno_s, dim=1).mean()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # self.attention_mask = attention_mask
        soft_label = self.teacher(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, output_hidden_states=True)
        outputs = self.student(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        lkd = self.loss_kd(soft_label.logits, outputs.logits)
        lrail = self.RAIL_c_loss(soft_label, outputs)
        loss = self.lambdas[0]*outputs.loss+self.lambdas[1]*lkd+self.lambdas[2]*lrail
        losses = [outputs.loss.detach().clone(), lkd.detach().clone(), lrail.detach().clone()]
        return outputs, loss, losses

    def ILD_loss(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # use when you need only intermadiate layer loss
        soft_label = self.teacher(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, output_hidden_states=True)
        outputs = self.student(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        loss = self.RAIL_c_loss(soft_label, outputs)
        return loss

class RAILKD_l_Model(RAILKDModel):
    def __init__(self, conf, task, num_labels):
        super().__init__(conf, task, num_labels)
        self.W_t = torch.nn.ModuleList(torch.nn.Linear(in_features=self.teacher.config.hidden_size, out_features=conf['linear']) for i in range(self.teacher.config.num_hidden_layers))
        self.W_s = torch.nn.ModuleList(torch.nn.Linear(in_features=self.student.config.hidden_size, out_features=conf['linear']) for i in range(self.student.config.num_hidden_layers))

        # for analyzer
        self.hidden_states = {}
        self.linear_states = {}

        if 'fixed_linear' in conf and conf['fixed_linear']:
            print('***** no ILD mapping training *****')
            for param in self.W_t.parameters():
                param.requires_grad = False
            for param in self.W_s.parameters():
                param.requires_grad = False
    
    def RAIL_l_loss(self, soft_label, outputs):
        # tuple of tensor
        # hs_*[0] is embedding layer output
        # hs_* shape (batch_size, sequence_length, hidden_size)
        # h_hat_* shape (batch_size, conf['linear'])

        # for analyzer
        self.hidden_states = {}
        self.linear_states = {}

        hs_t = soft_label.hidden_states[1:]
        hs_s = outputs.hidden_states[1:]
        batch_size = hs_t[0].shape[0]

        # (batch_size, sequence_length, hidden_size) -> (batch_size, hidden_size)
        h_bar_t = list(map(self.mean_pooling, hs_t))
        h_bar_s = list(map(self.mean_pooling, hs_s))
        
        h_hats_t = []
        all_ild_t = []

        for i, hs in enumerate(hs_t):
            h_hats_t.append(self.W_t[i](h_bar_t[i]))
            all_ild_t.append(h_hats_t[i].detach().clone().reshape(batch_size, 1, self.conf['linear']))

        ild_t = []
        ild_s = []

        loss = torch.tensor(0, dtype=torch.float, requires_grad=False).to(hs_t[0].device)
        for i, layer in enumerate(self.selected):
            h_hat_t = h_hats_t[layer]
            h_hat_s = self.W_s[i](h_bar_s[i])
            deno_t = torch.linalg.norm(h_hat_t, dim=1).reshape(batch_size, 1)
            deno_s = torch.linalg.norm(h_hat_s, dim=1).reshape(batch_size, 1)
            loss +=torch.linalg.norm(h_hat_t/deno_t - h_hat_s/deno_s, dim=1).mean()

            # for analyzer
            ild_t.append(h_hat_t.detach().clone().reshape(batch_size, 1, self.conf['linear']))
            ild_s.append(h_hat_s.detach().reshape(batch_size, 1, self.conf['linear']))

        loss = loss / len(self.selected)

        # for analyzer
        self.hidden_states['teacher'] = hs_t
        self.hidden_states['student'] = hs_s
        self.linear_states['teacher'] = torch.cat(ild_t, dim=1)
        self.linear_states['student'] = torch.cat(ild_s, dim=1)
        self.linear_states['teacher_all'] = torch.cat(all_ild_t, dim=1)

        return loss

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # self.attention_mask = attention_mask
        soft_label = self.teacher(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, output_hidden_states=True)
        outputs = self.student(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        lkd = self.loss_kd(soft_label.logits, outputs.logits)
        lrail = self.RAIL_l_loss(soft_label, outputs)
        loss = self.lambdas[0]*outputs.loss+self.lambdas[1]*lkd+self.lambdas[2]*lrail
        losses = [outputs.loss.detach().clone(), lkd.detach().clone(), lrail.detach().clone()]
        return outputs, loss, losses

    def ILD_loss(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # use when you need only intermadiate layer loss
        soft_label = self.teacher(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, output_hidden_states=True)
        outputs = self.student(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        loss = self.RAIL_l_loss(soft_label, outputs)
        return loss

class CurriculumRAILKDModel(RAILKDModel):
    def __init__(self, conf, task, num_labels):
        super().__init__(conf, task, num_labels)
    
    def set_lambdas(self, lambdas):
        self.lambdas = lambdas