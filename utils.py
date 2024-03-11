# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import random
import re
import json
import numpy as np
import torch
import torch.nn as nn
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import pickle
import os

class Config(nn.Module): 
    def __init__(self, args):
        super().__init__()
        self.alpha = args.alpha  # balance factor
        self.max_seq_length = args.max_seq_length
        with open('../data/{}/tag.json'.format(args.data_name), 'r', encoding='utf-8') as fp:
            self.label2id = json.load(fp)
        self.id2label = {i:label for label, i in self.label2id.items()}
        self.num_labels = len(self.label2id) + 1  # We do not use 0

        # BERT
        self.output_model_dir = args.output_model_dir
        self.bert_model = args.bert_model
        self.cache_dir = args.cache_dir if args.cache_dir else str(PYTORCH_PRETRAINED_BERT_CACHE)
        

class Process(nn.Module): 

    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.label2id = config.label2id
        self.max_seq_length = config.max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True)

    def load_data(self, data_path, do_predict=False):
        return self.read_chinese_ner_data(data_path)

    def convert_examples_to_features(self, examples, entity_weight, context_weight, use_adaptive_weight):
        max_seq_length = min(max([len(e[0]) for e in examples])+2, self.max_seq_length)
        features = []
        len_entity_token = []
        len_context_token = []
        
        for (tokens, labels) in examples:   
            if len(tokens) >= max_seq_length - 1:
                tokens = tokens[0:(max_seq_length - 2)]
                labels = labels[0:(max_seq_length - 2)]
            
            ntokens = []
            label_ids = []
            weights = []
            
            entity_token_count = 0
            context_token_count = 0

            ntokens.append("[CLS]")
            label_ids.append(self.label2id["[CLS]"])
            weights.append(1)

            for i, token in enumerate(tokens):
                token = self.tokenizer.tokenize(token)
                if len(token) == 0:
                    continue
                ntokens.extend(token)
                label_ids.append(self.label2id[labels[i]])
                if labels[i]!='O':
                    entity_token_count += 1
                    if use_adaptive_weight==False:
                        weights.append(entity_weight)
                else:
                    context_token_count += 1
                    if use_adaptive_weight==False:
                        weights.append(context_weight)
            
            if use_adaptive_weight==True:
                if entity_token_count!=0:
                    context_weight = entity_weight*entity_token_count/(context_token_count*1.0)
                else:
                    context_weight = 1.0
                for i, token in enumerate(tokens):
                    token = self.tokenizer.tokenize(token)
                    if len(token) == 0:
                        continue
                    if labels[i]!='O':
                        weights.append(entity_weight)
                    else:
                        weights.append(context_weight)

            ntokens.append("[SEP]")
            label_ids.append(self.label2id["[SEP]"])
            input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
            weights.append(1)

            input_mask = [1] * len(input_ids)

            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                label_ids.append(0)
                input_mask.append(0)
                weights.append(0)
            
            assert len(input_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(weights) == max_seq_length

            features.append((input_ids, label_ids, input_mask, weights, entity_token_count, context_token_count))
        return features

    def feature2input(self, device, feature):
        input_ids = torch.tensor([f[0] for f in feature], dtype=torch.long).to(device)
        label_ids = torch.tensor([f[1] for f in feature], dtype=torch.long).to(device)
        input_mask = torch.tensor([f[2] for f in feature], dtype=torch.long).to(device)
        weights = torch.tensor([f[3] for f in feature], dtype=torch.float).to(device)
        entity_token_count = torch.tensor([f[4] for f in feature], dtype=torch.long).to(device)
        context_token_count = torch.tensor([f[5] for f in feature], dtype=torch.long).to(device)
        return input_ids, label_ids, input_mask, weights, entity_token_count, context_token_count

    def get_entity_dict_drop(self, data_path):
        f = open(data_path, 'r', encoding='utf-8')
        entity = ''
        entity_dict = {}
        entity_type = None
        text = ''
        texts = []
        label = []
        labels = []
        for line in f:
            line = line.strip('\n')
            if len(line)<3:
                if len(text)>0:
                    texts.append(text)
                    labels.append(label)
                    text = ''
                    label = []
            else:
                splits = line.split(' ')
                text += splits[0]
                label.append(splits[1])
        if len(text)>0:
            texts.append(text)
            labels.append(label)
        count_wrong_entity = 0
        for i in range(len(texts)):
            include_entity = False
            for j in range(len(labels[i])):
                if labels[i][j]!='O':
                    include_entity = True
                    break
            wrong_entity = False
            for j in range(1, len(labels[i])):
                if 'M-' in labels[i][j] and 'O'==labels[i][j-1]:
                    wrong_entity = True
                    count_wrong_entity += 1
                    break
                if (labels[i][j][2:]!=labels[i][j-1][2:]) and 'M-' in labels[i][j] and 'M-' in labels[i][j-1]:
                    wrong_entity = True
                    count_wrong_entity += 1
                    break
            if include_entity==True and wrong_entity==False:
                for j in range(len(texts[i])):
                    if 'B-' in labels[i][j] or 'M-' in labels[i][j]:
                        entity += texts[i][j]
                        entity_type = labels[i][j][2:]
                    else:
                        if 'E-' in labels[i][j] or 'S-' in labels[i][j]:
                            entity += texts[i][j]
                            entity_type = labels[i][j][2:]
                            if entity_type is not None and entity!='':
                                if entity_type in entity_dict.keys():
                                    entity_dict[entity_type].append(entity)
                                else:
                                    entity_dict[entity_type] = [entity]
                            entity = ''
                            entity_type = None
        for k,v in entity_dict.items():
            entities = entity_dict[k]
            tuple_List = []
            temp = {}
            new_list = []
            for e in entities:
                if e in temp.keys():
                    temp[e][1] += 1
                else:
                    temp[e] = [len(e), 0]
            for a,b in temp.items():
                tuple_List.append([a,b[0],b[1]])
            tuple_List = sorted(tuple_List, key=lambda x: (x[1], -x[2]))
            for [entity, length, cipin] in tuple_List:
                new_list.append(entity)
            entity_dict[k] = new_list
        file = open('./entity_dict_drop.pickle', 'wb')
        pickle.dump(entity_dict, file)
        file.close()
        return
    
    def read_chinese_ner_data(self, data_path, mode='train', read_drop = True, shuffle_dict = False):
        f = open(data_path, 'r', encoding='utf-8')
        data = []
        sentence = []
        label = []
        start_end = []
        if mode=='train':
            if read_drop == True:
                if os.path.exists('./entity_dict_drop.pickle')==False:
                    self.get_entity_dict_drop(data_path)
                with open('./entity_dict_drop.pickle', 'rb') as file:
                    entity_dict = pickle.load(file)     
            if shuffle_dict == True:
                for k,v in entity_dict.items():
                    entity_dict[k] = random.shuffle(entity_dict[k])
                    entity_dict[k] = sorted(entity_dict[k], key=lambda x:len(x))
                if read_drop == True:
                    if os.path.exists('./entity_dict_drop_shuffle.pickle')==False:
                        with open('./entity_dict_drop_shuffle.pickle', 'wb') as file:
                            pickle.dump(entity_dict, file)
            if read_drop==True:
                texts = []
                labels = []
                text = ''
                label = []
                for line in f:
                    line = line.strip('\n')
                    if len(line)<3:
                        if len(text)>0:
                            texts.append(text)
                            labels.append(label)
                            text = ''
                            label = []
                    else:
                        splits = line.split(' ')
                        text += splits[0]
                        label.append(splits[1])
                if len(text)>0:
                    texts.append(text)
                    labels.append(label)
                for i in range(len(texts)):
                    start_end = []
                    sub_sentence = []
                    sub_label = []
                    include_entity = False
                    for j in range(len(labels[i])):
                        if labels[i][j]!='O':
                            include_entity = True
                            break
                    wrong_entity = False
                    count_wrong_entity = 0
                    for j in range(1, len(labels[i])):
                        if ('M-' in labels[i][j] and 'O'==labels[i][j-1]):
                            wrong_entity = True
                            count_wrong_entity += 1
                            break
                        if (labels[i][j][2:]!=labels[i][j-1][2:]) and 'M-' in labels[i][j] and 'M-' in labels[i][j-1]!='O':
                            wrong_entity = True
                            count_wrong_entity += 1
                            break
                    
                    if include_entity==True and wrong_entity==False:
                        for j in range(len(labels[i])):
                            l = labels[i][j]
                            if 'S-' in l:
                                start_end.append((j, j))
                            elif 'B-' in l:
                                start = j
                            elif 'E-' in l:
                                end = j
                                start_end.append((start, end))
                        temp_i = 0
                        for (s,e) in start_end:
                            for j in range(temp_i, s):
                                sub_sentence.append(texts[i][j])
                                sub_label.append(labels[i][j])
                            typ = labels[i][s][2:]
                            entity = ''
                            for j in range(s, e+1):
                                entity += texts[i][j]
                            entity_l = entity_dict[typ]
                            index = entity_l.index(entity)
                            if index==len(entity_l)-1:
                                sub_word = entity_l[index-1]
                            elif index==0:
                                sub_word = entity_l[1]
                            elif len(entity_l[index+1])==len(entity):
                                sub_word = entity_l[index+1]
                            else:
                                sub_word = entity_l[index-1]
                            if len(sub_word) == 1:
                                sub_sentence.append(sub_word)
                                sub_label.append('S-'+typ)
                            else:
                                for j in range(len(sub_word)):
                                    sub_sentence.append(sub_word[j])
                                    if j==0:
                                        sub_label.append('B-'+typ)
                                    elif j==len(sub_word)-1:
                                        sub_label.append('E-'+typ)
                                    else:
                                        sub_label.append('M-'+typ)
                            temp_i = e+1
                        if temp_i < len(texts[i]):
                            for j in range(temp_i, len(texts[i])):
                                sub_sentence.append(texts[i][j])
                                sub_label.append(labels[i][j])
                        data.append((sub_sentence, sub_label))
                    text_split = []
                    for q in texts[i]:
                        text_split.append(q)
                    data.append((text_split,labels[i]))
        else:
            for line in f:
                line = line.strip('\n')
                if len(line)<3:
                    if len(sentence)>0:
                        data.append((sentence, label))
                        sentence = []
                        label = []
                else:
                    splits = line.split(' ')
                    c = splits[0]
                    l = splits[1]
                    sentence.append(c)
                    label.append(l)
        return data




