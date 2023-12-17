import torch, os, argparse, random
from transformers import Blip2Processor, Blip2ForConditionalGeneration, Blip2QFormerConfig, AutoTokenizer
import numpy as np
import json
import torch.nn as nn
import sys
import numpy as np
from torch.nn import DataParallel
from torch.nn.modules import BatchNorm2d
from tqdm import tqdm
from Dataloader.dataloader import *
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import datetime
from transformers.optimization import Adafactor, AdafactorSchedule
from utils import *
from esnli_BLIP_eval import *
VERSION = "Salesforce/blip2-flan-t5-xxl"
COLOR_MASKS = ['red', 'yellow', 'blue', 'green', 'purple', 'white', 'black', 'brown', 'orange', 'pink', 'gold', 'silver','gray']
COLOR_MASKS_NAME = ['red box', 'yellow box', 'blue box', 'green box', 'purple box', 'white box', 'black box', 'brown box', 'orange box', 'pink box', 'gold box', 'silver box','gray box']

def get_and_freeze_model(path=None, finetune_vision_encoder=True):
    processor = Blip2Processor.from_pretrained(VERSION)
    device = "cuda"
    model = Blip2ForConditionalGeneration.from_pretrained(VERSION, device_map="auto", max_memory = {0: "4GIB", 1: "18GIB", 2: "18GIB", 3: "18GIB", 4: "18GIB", 5: "18GIB", 6: "18GIB"}) #, load_in_8bit=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(VERSION)

    if path is not None:
        checkpoint = torch.load(path, map_location='cpu')
        ckpt_pth = checkpoint['model_state_dict']
        model.qformer.load_state_dict(ckpt_pth)
        ckpt_pth = checkpoint['model_vision_encoder_state_dict']
        model.vision_model.load_state_dict(ckpt_pth)

    model.train()
    # finetune the qformer and the image encoder
    for param in model.parameters():
        param.requires_grad = False
    if finetune_vision_encoder:
        for param in model.vision_model.parameters():
            param.requires_grad = True
    for param in model.qformer.parameters():
        param.requires_grad = True
    return processor, model

def finetune_rationale_answer_esnli(path=None, use_boundingbox=True, finetune_vision_encoder=True, add_rationale=False):
    processor, model = get_and_freeze_model(path, finetune_vision_encoder=finetune_vision_encoder)
    train_dataset = eSNLITrainDataset(processor=processor, data_path="data")
    n_samples = len(train_dataset)
    print(n_samples)
    n_choice = int(n_samples*0.03)
    tokenizer = AutoTokenizer.from_pretrained(VERSION)
    batch_size = 2
    random.seed(1)
    indexes_list = random.choices(list(range(n_samples)), k=n_choice)
    indexes = [tuple([indexes_list[i+j] for j in range(0, min(batch_size, n_choice-i))]) for i in range(0, n_choice, batch_size)]
    num_epochs = 5
    num_training_steps = int(num_epochs * n_choice / batch_size)
    progress_bar = tqdm(range(num_training_steps))
    timeNow = datetime.datetime.now().strftime("%I:%M%p_on_%B_%d_%Y_")
    optimizer = AdamW(model.parameters(), lr=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0, last_epoch=-1, verbose=False)
    ckpt_path = 'ckpts/checkpoint_VQAX'+"_"+str(finetune_vision_encoder)+"_"+str(timeNow)+'.pth'
    loss = 0
    for epoch in range(num_epochs):
        tot_loss = 0
        j = 0
        for index_tuple in indexes:
            inputs_image = []
            question = []
            labels = []
            for i in index_tuple:
                element = train_dataset[i]
                inputs_image.append(element[0])
                if not add_rationale:
                    question_element = "Question: "+element[1]+"\n\nOptions: "+ "(A) "+element[4][0]+ " (B) "+element[4][1]+" (C) "+element[4][2]+ " (D) "+element[4][3]+"\n\nAnswer: "
                else:
                    question_element = "Based on the image can we conclude that '"+element[1]+"'?"+"\n\nAnswer: "
                question.append(question_element)
                if not add_rationale:
                    labels.append(convert_index_to_letter(element[5]))
                else:
                    labels.append(element[2]+" because "+element[3])
            print(labels)
            inputs_image = torch.stack(inputs_image).squeeze(1)
            question_encoded = tokenizer(question, padding='longest', return_tensors='pt')
            question_id = question_encoded["input_ids"]
            question_mask = question_encoded["attention_mask"]
            labels_ids = tokenizer(labels, padding='longest', return_tensors='pt')["input_ids"]
            labels_ids[labels_ids == tokenizer.pad_token_id] = -100 
            outputs = model(inputs_image, question_id, question_mask, labels=labels_ids)

            loss = outputs.loss
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        print(tot_loss/(n_choice))
        loss = tot_loss/(n_choice)
        torch.save({
            'model_state_dict': model.qformer.state_dict(),
            'model_vision_encoder_state_dict' : model.vision_model.state_dict(),
            'loss': loss,
            }, './ckpts/checkpoint_eSNLI'+"_3%"+str(finetune_vision_encoder)+"_"+str(timeNow)+'.pth')
    return ckpt_path

def finetune_rationale_answer_esnli_self_train(path=None, use_boundingbox=True, finetune_vision_encoder=True, add_rationale=False):
    processor, model = get_and_freeze_model(path, finetune_vision_encoder=finetune_vision_encoder)
    train_dataset = eSNLIEvalDataset(processor=processor, data_path="data")
    n_choice = 8
    tokenizer = AutoTokenizer.from_pretrained(VERSION)
    batch_size = 4
    random.seed(1)
    num_epochs = 8
    num_training_steps = int(num_epochs * n_choice / batch_size)
    progress_bar = tqdm(range(num_training_steps))
    timeNow = datetime.datetime.now().strftime("%I:%M%p_on_%B_%d_%Y_")
    optimizer = AdamW(model.parameters(), lr=4e-6)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0, last_epoch=-1, verbose=False)
    ckpt_path = 'ckpts/checkpoint_VQAX'+"_"+str(finetune_vision_encoder)+"_"+str(timeNow)+'.pth'
    loss = 0
    with open('./filtered_train/esnli_correct_on_2', 'r') as f:
        data_dict = json.load(f)
    
    indexes = list(data_dict.keys())
    indexes_list = random.choices(indexes, k=n_choice)
    indexes = [tuple([indexes_list[i+j] for j in range(0, min(batch_size, n_choice-i))]) for i in range(0, n_choice, batch_size)]
    for epoch in range(num_epochs):
        tot_loss = 0
        j = 0
        for index_tuple in indexes:
            inputs_image = []
            question = []
            labels = []
            for i in index_tuple:
                element = train_dataset[int(i)]
                caption = data_dict[i]["caption"]
                inputs_image.append(element[0])
                if not add_rationale:
                    question_element = "Question: "+element[1]+"\n\nOptions: "+ "(A) "+element[4][0]+ " (B) "+element[4][1]+" (C) "+element[4][2]+ " (D) "+element[4][3]+"\n\nAnswer: "
                else:
                    question_element = "Based on the image can we conclude that '"+element[1]+"'?"+"\n\nAnswer: "
                question.append(question_element)
                if not add_rationale:
                    labels.append(convert_index_to_letter(element[5]))
                else:
                    labels.append(element[2]+" because "+caption)
            inputs_image = torch.stack(inputs_image).squeeze(1)
            question_encoded = tokenizer(question, padding='longest', return_tensors='pt')
            question_id = question_encoded["input_ids"]
            question_mask = question_encoded["attention_mask"]
            labels_ids = tokenizer(labels, padding='longest', return_tensors='pt')["input_ids"]
            labels_ids[labels_ids == tokenizer.pad_token_id] = -100 
            outputs = model(inputs_image, question_id, question_mask, labels=labels_ids)

            loss = outputs.loss
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        print(tot_loss/(n_choice))
        loss = tot_loss/(n_choice)
        torch.save({
            'model_state_dict': model.qformer.state_dict(),
            'loss': loss,
            }, './ckpts/checkpoint_esnli_selftrain_given_wVGCoT_8shot_'+str(finetune_vision_encoder)+"_"+str(timeNow)+'.pth')
    return ckpt_path

if __name__=='__main__':
    """finetune on train"""
    path = finetune_rationale_answer_esnli(finetune_vision_encoder=True, use_boundingbox=True, add_rationale=True)
    """finetune self train"""
    path = finetune_rationale_answer_esnli_self_train(path='ckpts/...', use_boundingbox=True, finetune_vision_encoder=False, add_rationale=True)
