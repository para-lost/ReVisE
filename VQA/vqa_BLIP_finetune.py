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
from Dataloader.dataloader import AOKVQATrainDataset,VQAXTrainDataset
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import datetime
from transformers.optimization import Adafactor, AdafactorSchedule
from utils import *
from vqa_BLIP_eval import *
VERSION = "Salesforce/blip2-flan-t5-xxl"
COLOR_MASKS = ['red', 'yellow', 'blue', 'green', 'purple', 'white', 'black', 'brown', 'orange', 'pink', 'gold', 'silver','gray']
COLOR_MASKS_NAME = ['red box', 'yellow box', 'blue box', 'green box', 'purple box', 'white box', 'black box', 'brown box', 'orange box', 'pink box', 'gold box', 'silver box','gray box']

def get_and_freeze_model(path=None, finetune_vision_encoder=True):
    processor = Blip2Processor.from_pretrained(VERSION)
    device = "cuda"
    model = Blip2ForConditionalGeneration.from_pretrained(VERSION, device_map="auto") #, load_in_8bit=True, device_map="auto")
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

def finetune(model, finetune_type="non-cot"):
    if finetune_type == "non-cot":
        finetune_noncot(model)
    elif finetune_type == "cot-text":
        finetune_textcot(model)
    elif finetune_type == "cot-multimodal":
        finetune_mmcot(model)
    else:
        finetune_contrastcot(model)
def convert_index_to_letter(index):
    if index == 0:
        return "(A)"
    elif index == 1:
        return "(B)"
    elif index == 2:
        return "(C)"
    return "(D)"

def collate_tokenize(data, tokenizer, finetuneType="mmcot-step1"): 
    image_batch = torch.stack([element[0] for element in data]).squeeze(1)
    if finetuneType=='mmcot-step1':
        question_batch_with_answer = ["Question: "+element[3]+" Choice: "+ "(A) "+element[4][0]+ " (B) "+element[4][1]+" (C) "+element[4][2]+ " (D) "+element[4][3]+" Rationale: " for element in data]
        question_tokenized = tokenizer(question_batch_with_answer, padding='longest', truncation=True, return_tensors='pt')
        answer_rationale_batch = [element[6][element[7]] for element in data]
        answer_rationale_tokenized = tokenizer(answer_rationale_batch, padding='longest', truncation=True, return_tensors='pt')
        question_batch = ["Question: "+element[3] for element in data]
        question_only_tokenized = tokenizer(question_batch, padding='longest', truncation=True, return_tensors='pt')
        # question_len = torch.sum((question_only_tokenized['attention_mask'] == 1), dim=1)
        # print(question_len)
        choice_batch = [" Choice: "+ "(A) "+element[4][0]+ " (B) "+element[4][1]+" (C) "+element[4][2]+ " (D) "+element[4][3]+" Rationale: " for element in data]
        choice_only_tokenized = tokenizer(choice_batch, padding='longest', truncation=True, return_tensors='pt')
        # choice_len = torch.sum((choice_only_tokenized['attention_mask'] == 1), dim=1)
        # print(choice_len)
        rationale_batch_with_answer = ["Question: "+element[3]+" Rationale: "+element[6][element[7]]+" Choice: "+ "(A) "+element[4][0]+ " (B) "+element[4][1]+" (C) "+element[4][2]+ " (D) "+element[4][3]+" Answer: " for element in data]
        rationale_tokenized = tokenizer(rationale_batch_with_answer, padding='longest', truncation=True, return_tensors='pt')
        answer_batch = [convert_index_to_letter(element[5]) for element in data]
        answer_tokenized = tokenizer(answer_batch, padding='longest', truncation=True, return_tensors='pt')
        
        return image_batch, question_tokenized, answer_rationale_tokenized,rationale_tokenized, answer_tokenized, question_only_tokenized, choice_only_tokenized#, question_len, choice_len

    elif finetuneType=='mmcot-step2':
        question_batch = ["Question: "+element[3]+" "+element[6][element[7]]+" Choice: "+ "(A) "+element[4][0]+ " (B) "+element[4][1]+" (C) "+element[4][2]+ " (D) "+element[4][3]+" Answer: " for element in data]
        question_tokenized = tokenizer(question_batch, padding='longest', truncation=True, return_tensors='pt')
        answer_rationale_batch = [convert_index_to_letter(element[5]) for element in data]
        answer_rationale_tokenized = tokenizer(answer_rationale_batch,  padding='longest', truncation=True, return_tensors='pt')

    else:
        question_batch = ["Question: "+element[3]+" Choice: "+ "(A) "+element[4][0]+ " (B) "+element[4][1]+" (C) "+element[4][2]+ " (D) "+element[4][3]+" Answer: " for element in data]
        question_tokenized = tokenizer(question_batch, padding='longest', truncation=True, return_tensors='pt')
        answer_rationale_batch = [convert_index_to_letter(element[5]) for element in data]
        answer_rationale_tokenized = tokenizer(answer_rationale_batch,  padding='longest', truncation=True, return_tensors='pt')

    return image_batch, question_tokenized, answer_rationale_tokenized

def self_forward_with_cot(model, pixel_values,input_ids,attention_mask,labels,cot_ids=None, cot_attns=None, isTrain=True):
    return_dict = True
    batch_size = pixel_values.shape[0]
    vision_outputs = model.vision_model(
        pixel_values=pixel_values,
        return_dict=return_dict,
    )
    image_embeds = vision_outputs[0]

    # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
    image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

    query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
    query_outputs = model.qformer(
        query_embeds=query_tokens,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_attention_mask,
    )
    query_output = query_outputs[0]

    # step 3: use the language model, conditioned on the query outputs and the prompt
    language_model_inputs = model.language_projection(query_output)
    language_model_attention_mask = torch.ones(
        language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
    )
    inputs_embeds = model.language_model.get_input_embeddings()(input_ids)
    inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    expected_device = language_model_attention_mask.device
    attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(expected_device)], dim=1)

    if cot_ids != None:
        cot_ids = cot_ids.expand(batch_size, -1, -1)
        cot_attns = cot_attns.expand(batch_size,  -1)
        inputs_embeds = torch.cat([cot_ids, inputs_embeds], dim=1)
        attention_mask = torch.cat([cot_attns, attention_mask], dim=1)

    outputs = model.language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
    )
    loss = outputs.loss 
    logits = outputs.logits 
    if isTrain:
        with torch.no_grad():
            outputs_text = model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_length = 100
                ) 
        return loss, logits, outputs_text

    return loss, logits

def finetune_mmcot(use_cot=True, finetuneType="mmcot-step1", path=None):
    processor, model = get_and_freeze_model(path)
    train_dataset = VCRTrainDataset(processor=processor, data_path="data/vcr")
    n_samples = len(train_dataset)
    n_choice = 1000
    tokenizer = AutoTokenizer.from_pretrained(VERSION)
    batch_size = 2
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=lambda b: collate_tokenize(b, tokenizer, finetuneType))
    print(train_dataloader)
    device = model.device
    num_epochs = 10
    num_training_steps = int(num_epochs * 1000 / batch_size)
    progress_bar = tqdm(range(num_training_steps))
    timeNow = datetime.datetime.now().strftime("%I:%M%p_on_%B_%d_%Y_")
    
    if finetuneType=="mmcot-step1":
        optimizer1 = AdamW(model.parameters(), lr=1e-7)
        lr_scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=num_epochs, eta_min=0, last_epoch=-1, verbose=False)
        optimizer2 = AdamW(model.parameters(), lr=1e-6)
        lr_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=num_epochs, eta_min=0, last_epoch=-1, verbose=False)
    else:
        optimizer = AdamW(model.parameters(), lr=1e-6)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0, last_epoch=-1, verbose=False)
    
    if use_cot:
        inputs, attns = construst_cot_examples(model, train_dataset, tokenizer)
    for epoch in range(num_epochs):
        num = 0 
        tot_loss1 = 0
        tot_loss2 = 0
        tot_loss = 0
        for batch in train_dataloader:
            num += 1
            if num == 1000/batch_size:
                break
            inputs_image = batch[0]
            question_id = batch[1]['input_ids']
            question_mask = batch[1]['attention_mask']
            answer_rationale_id = batch[2]['input_ids']
            answer_rationale_mask = batch[2]['attention_mask']

            if use_cot:
                loss, _, _ = self_forward_with_cot(model, inputs_image, question_id, question_mask, answer_rationale_id)
                # loss = outputs.loss
                loss.mean().backward()
                optimizer.step()
                lr_scheduler.step()
                tot_loss += loss.item()
                optimizer.zero_grad()
                progress_bar.update(1)
            else:
                outputs = model(inputs_image, question_id, question_mask, labels=answer_rationale_id)
                if finetuneType == "mmcot-step1":
                    # train the model to generate a better rationale
                    loss1 = outputs.loss
                    loss1.backward()
                    optimizer1.step()
                    lr_scheduler1.step()
                    optimizer1.zero_grad()
                    # train the model to predict a correct answer using its own rationale; joint training
                    with torch.no_grad():
                        rationale_id = batch[3]['input_ids']
                        rationale_mask = batch[3]['attention_mask']
                        answer_id = batch[4]['input_ids']
                        answer_mask = batch[4]['attention_mask']
                        question_only_id = batch[5]['input_ids']
                        question_only_mask = batch[5]['attention_mask']
                        choice_id = batch[6]['input_ids']
                        choice_mask = batch[6]['attention_mask']
                        output = model.generate(inputs_image, question_id, question_mask)
                        output_text = processor.batch_decode(output, skip_special_tokens=True)
                        print(output_text)
                        output_text = [output.split('.')[0] for output in output_text]
                        tokenized_output_text = tokenizer(output_text, padding='longest', truncation=True, return_tensors='pt')
                        rationale_id = torch.cat((question_only_id,tokenized_output_text['input_ids'],choice_id), dim=1)
                        rationale_mask = torch.cat((question_only_mask,tokenized_output_text['attention_mask'], choice_mask), dim=1)
                        print(rationale_id)
                        print(rationale_mask)
                    outputs = model(inputs_image, rationale_id, rationale_mask, labels=answer_id)
                    loss2 = outputs.loss
                    loss2.backward()
                    tot_loss1 += loss1.item()
                    tot_loss2 += loss2.item()
                    tot_loss=tot_loss1+tot_loss2
                    optimizer2.step()
                    lr_scheduler2.step()
                    optimizer2.zero_grad()
                    progress_bar.update(1)

                else:
                    loss = outputs.loss
                    loss.mean().backward()
                    optimizer.step()
                    lr_scheduler.step()
                    tot_loss += loss.item()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    
            

        print(tot_loss1/(num*batch_size))
        print(tot_loss2/(num*batch_size))
        print(tot_loss/(num*batch_size))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.qformer.state_dict(),
            'loss': tot_loss/(num*batch_size),
            }, './ckpts/checkpoint_'+finetuneType+'_'+str(timeNow)+str(epoch)+'.pth')
    return model

# cot text only finetune
def finetune_mmnoncot():
    processor, model = get_and_freeze_model()
    train_dataset = VCRTrainDataset(processor=processor, data_path="data/vcr")
    n_samples = len(train_dataset)
    n_choice = 1000
    tokenizer = AutoTokenizer.from_pretrained(VERSION)
    batch_size = 2
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=lambda b: collate_tokenize(b, tokenizer, "mmnoncot"))
    print(train_dataloader)
    optimizer = AdamW(model.parameters(), lr=1e-6)
    num_epochs = 5
    num_training_steps = int(num_epochs * 1000 / batch_size)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0, last_epoch=-1, verbose=False)
    # lr_scheduler = get_scheduler(
    #                     name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps,
    #                 )
    timeNow = datetime.datetime.now().strftime("%I:%M%p_on_%B_%d_%Y_")
    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(num_epochs):
        num = 0
        tot_loss = 0
        for batch in train_dataloader:
            num += 1
            if num == 1000/batch_size:
                break
            inputs_image = batch[0]
            question_id = batch[1]['input_ids']
            question_mask = batch[1]['attention_mask']
            answer_rationale_id = batch[2]['input_ids']
            answer_rationale_mask = batch[2]['attention_mask']
            outputs = model(inputs_image, question_id, question_mask, labels=answer_rationale_id, return_dict=True)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            tot_loss += loss.item()
            optimizer.zero_grad()
            progress_bar.update(1)

    # only save the qformer
        print(tot_loss/(num*batch_size))
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.qformer.state_dict(),
            'loss': tot_loss/(num*batch_size),
            }, './ckpts/checkpoint_mmnoncot_'+str(timeNow)+str(epoch)+'.pth')
    return model

# multi-modal cot finetune

def finetune_rationale_answer(finetuneType="two-step", path=None, use_cot=True):
    processor, model = get_and_freeze_model(path)
    train_dataset = VCRTrainDataset(processor=processor, data_path="data/vcr")
    n_samples = len(train_dataset)
    n_choice = 1000
    tokenizer = AutoTokenizer.from_pretrained(VERSION)
    indexes_list_cot = [173870,97677]
    batch_size = 1
    num_epochs = 1
    num_training_steps = int(num_epochs * 1000 / batch_size)
    optimizer1 = AdamW(model.parameters(), lr=1e-7)
    lr_scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=num_epochs, eta_min=0, last_epoch=-1, verbose=False)
    optimizer2 = AdamW(model.parameters(), lr=1e-6)
    lr_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=num_epochs, eta_min=0, last_epoch=-1, verbose=False)
    progress_bar = tqdm(range(num_training_steps))
    timeNow = datetime.datetime.now().strftime("%I:%M%p_on_%B_%d_%Y_")
    indexes_list = random.choices(list(range(n_samples)), k=n_choice)
    indexes = [tuple([indexes_list[i+j] for j in range(0, batch_size)]) for i in range(0, n_choice, batch_size)]
    print(indexes)
    output_list = []
    output_front = []
    num_tot = 0
    cot_ids, cot_attns, cot_ids2, cot_attns2 = construct_cot_examples(model, tokenizer, processor, indexes_list=indexes_list_cot)
    with open('rationale', 'w') as f1:
        for index_tuple in indexes:
            inputs_image = []
            question = []
            rationale = []
            for i in index_tuple:
                element = train_dataset[i]
                inputs_image.append(element[0])
                question_element = "Question: "+element[3] + " Generate a rationale"
                cnt = 0
                idx = 100000
                want_name = ''
                for box_name in COLOR_MASKS_NAME:
                    if box_name in element[3]:
                        if element[3].find(box_name) < idx:
                            idx = element[3].find(box_name)
                            want_name = box_name
                        if cnt == 0:
                            question_element += " about the person in "
                        else:
                            question_element += " and the person in "
                        question_element += box_name
                        cnt += 1
                question_element += ":" 
                question_element += " the person with "
                question_element += want_name
                question.append(question_element)

                if cnt != 0:
                    output_front.append("the person with "+want_name)
                else:
                    output_front.append("")
                    
                rationale.append(element[6][element[7]])
            inputs_image = torch.stack(inputs_image).squeeze(1)
            question_encoded = tokenizer(question, padding='longest', truncation=True, return_tensors='pt')
            rationale_encoded = tokenizer(rationale, padding='longest', truncation=True, return_tensors='pt')
            question_id = question_encoded["input_ids"]
            question_mask = question_encoded["attention_mask"]
            rationale_id = rationale_encoded["input_ids"]
            
            # loss1, _, output = self_forward_with_cot(model, inputs_image, input_ids=question_id, attention_mask=question_mask, labels=rationale_id ,cot_ids=cot_ids, cot_attns=cot_attns, isTrain=True)
            output = model_generate(model, inputs_image, question_id, question_mask, cot_ids, cot_attns)
            with torch.no_grad():
                output_text = processor.batch_decode(output, skip_special_tokens=True)
                print(output_text)
                for output_text_element in output_text:
                    output_list.append(output_text_element)
                    f1.write("the person with "+want_name + " "+output_text_element + "\n")
    
    train_dataset = VCRTrainDataset(processor=processor, data_path="data/vcr")
    batch_size = 4
    num_epochs = 10
    num_training_steps = int(num_epochs * 1000 / batch_size)
    optimizer2 = AdamW(model.parameters(), lr=2e-6)
    lr_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=num_epochs, eta_min=0, last_epoch=-1, verbose=False)
    progress_bar = tqdm(range(num_training_steps))
    timeNow = datetime.datetime.now().strftime("%I:%M%p_on_%B_%d_%Y_")
    indexes = [tuple([indexes_list[i+j] for j in range(0, batch_size)]) for i in range(0, n_choice, batch_size)]
    print(indexes)
    for epoch in range(num_epochs):
        num_tot = 0   
        tot_loss1 = 0
        tot_loss2 = 0
        tot_loss = 0
        for index_tuple in indexes:
            inputs_image = []
            
            rationale_ans = []
            labels = []
            for i in index_tuple:
                element = train_dataset[i]
                inputs_image.append(element[0])
                rationale_ans.append("Question: "+element[3]+" "+output_front[num_tot] + " "+output_list[num_tot]+". Choice: "+ "(A) "+element[4][0]+ " (B) "+element[4][1]+" (C) "+element[4][2]+ " (D) "+element[4][3]+" Answer: ")
                labels.append(convert_index_to_letter(element[5]))
                num_tot += 1
                
            inputs_image = torch.stack(inputs_image).squeeze(1)      
            print(rationale_ans)
            tokenized_rationale_ans = tokenizer(rationale_ans, padding='longest', truncation=True, return_tensors='pt')
            rationale_ans_id = tokenized_rationale_ans["input_ids"]
            rationale_ans_mask = tokenized_rationale_ans["attention_mask"]
            labels_ids = tokenizer(labels, padding='longest', truncation=True, return_tensors='pt')["input_ids"]
            loss2, _= self_forward_with_cot(model, inputs_image, rationale_ans_id, rationale_ans_mask, labels=labels_ids, isTrain=False)

            #loss2 = outputs.loss
            loss2.backward()
            tot_loss2 += loss2.item()
            optimizer2.step()
            lr_scheduler2.step()
            optimizer2.zero_grad()
            progress_bar.update(1)

        print(tot_loss1/n_choice)
        print(tot_loss2/n_choice)
        print(tot_loss/n_choice)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.qformer.state_dict(),
            'loss1': tot_loss1/(n_choice),
            'loss2': tot_loss2/(n_choice),
            }, './ckpts/checkpoint_'+finetuneType+'_'+str(timeNow)+str(epoch)+'.pth')
    return model

def model_generate(model, pixel_values, input_ids, attention_mask, cot_ids=None, cot_attns=None):
    with torch.no_grad():
        batch_size = pixel_values.shape[0]
        image_embeds = model.vision_model(pixel_values, return_dict=True).last_hidden_state
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
        )
        query_output = query_outputs.last_hidden_state
        language_model_inputs = model.language_projection(query_output) 
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        inputs_embeds = model.get_input_embeddings()(input_ids)
        
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)
        attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)

        if cot_ids != None:
            cot_ids = cot_ids.expand(batch_size, -1, -1)
            cot_attns = cot_attns.expand(batch_size, -1)
            inputs_embeds = torch.cat([cot_ids, inputs_embeds], dim=1)
            attention_mask = torch.cat([cot_attns, attention_mask], dim=1)

        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_length = 100,
        )
            
        return outputs



def output_text_edit(output_text):
    if '.' not in output_text or '-' in output_text or "'" in output_text:
        return ""
    else:
        return " Rationale: " + "Let's think step by step, " + output_text.split('.')[0] + "."

def finetune_rationale_answer_step2(path=None, finetune_vision_encoder=True):
    processor, model = get_and_freeze_model(path, finetune_vision_encoder=finetune_vision_encoder)
    train_dataset = VCRTrainDataset(processor=processor, data_path="data/vcr", use_boundingbox=True,use_imagedict=False)
    n_samples = len(train_dataset)
    n_choice = 1000
    tokenizer = AutoTokenizer.from_pretrained(VERSION)
    batch_size = 4
    indexes_list = random.choices(list(range(n_samples)), k=n_choice)
    indexes = [tuple([indexes_list[i+j] for j in range(0, min(batch_size, n_choice-i))]) for i in range(0, n_choice, batch_size)]
    num_epochs = 8
    num_training_steps = int(num_epochs * 1000 / batch_size)
    progress_bar = tqdm(range(num_training_steps))
    timeNow = datetime.datetime.now().strftime("%I:%M%p_on_%B_%d_%Y_")
    optimizer = AdamW(model.parameters(), lr=1e-6)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0, last_epoch=-1, verbose=False)
    ckpt_path = 'ckpts/checkpoint_'+str(timeNow)+'.pth'
    loss = 0
    for epoch in range(num_epochs):
        tot_loss = 0
        j = 0
        for index_tuple in indexes:
            inputs_image = []
            question = []
            for i in index_tuple:
                element = train_dataset[i]
                inputs_image.append(element[0])
                question_element = get_question_element_method5(element)
                question.append(question_element)
            inputs_image = torch.stack(inputs_image).squeeze(1)
               
            num = 0
            rationale_ans = []
            labels = []
            for i in index_tuple:
                element = train_dataset[i]
                rationale_ans.append("Question: "+element[3]+"\n\nOptions: "+ "(A) "+element[4][0]+ " (B) "+element[4][1]+" (C) "+element[4][2]+ " (D) "+element[4][3]+"\n\nAnswer: ")
                num += 1
                j+=1
                labels.append(convert_index_to_letter(element[5]))
           
            tokenized_rationale_ans = tokenizer(rationale_ans, padding='longest', truncation=True, return_tensors='pt')
            rationale_ans_id = tokenized_rationale_ans["input_ids"]
            rationale_ans_mask = tokenized_rationale_ans["attention_mask"]
            labels_ids = tokenizer(labels, padding='longest', truncation=True, return_tensors='pt')["input_ids"]
            labels_ids[labels_ids == tokenizer.pad_token_id] = -100 
            outputs = model(inputs_image, rationale_ans_id, rationale_ans_mask, labels=labels_ids)

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
            # 'model_vision_encoder_state_dict' : model.vision_model.state_dict(),
            'loss': loss,
            }, './ckpts/checkpoint_'+str(timeNow)+'.pth')
    return ckpt_path

def self_forward_t5(model, inputs_embeds,attention_mask,labels, cot_ids=None, cot_attns=None, return_logits=False):
    batch_size = inputs_embeds.shape[0]
    if cot_ids != None:
        cot_ids = cot_ids.expand(batch_size, -1, -1)
        cot_attns = cot_attns.expand(batch_size,  -1)
        inputs_embeds = torch.cat([cot_ids.to(inputs_embeds.device), inputs_embeds], dim=1)
        attention_mask = torch.cat([cot_attns.to(attention_mask.device), attention_mask], dim=1)
    outputs = model.language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
    )
    loss = outputs.loss 
    if return_logits:
        return loss, outputs.logits
    return loss

def finetune_t5(path=None): 
    processor, tokenizer, model = get_and_freeze_model_t5(path)
    train_dataset = VCRTrainDataset(processor=processor, data_path="data/vcr", use_boundingbox=False)
    n_samples = len(train_dataset)
    n_choice = 16
    batch_size = 1
    indexes_list = random.choices(list(range(n_samples)), k=n_choice)
    indexes = [tuple([indexes_list[i+j] for j in range(0, min(batch_size, n_samples-i-1))]) for i in range(0, n_choice, batch_size)]
    index_list_cot = [666, 890, 15678, 334]
    with torch.no_grad():
        cot_ids, cot_attns = construct_cot_element(model, tokenizer, processor,  indexes_list=index_list_cot)
    num_epochs = 10
    num_training_steps = int(num_epochs * n_choice / batch_size)
    optimizer = Adafactor(model.language_model.parameters(), relative_step=True, warmup_init=True)
    lr_scheduler = AdafactorSchedule(optimizer)
    progress_bar = tqdm(range(num_training_steps))
    timeNow = datetime.datetime.now().strftime("%I:%M%p_on_%B_%d_%Y_")
    for epoch in range(num_epochs):
        tot_loss = 0
        for index_tuple in indexes:
            questions = []
            questions_attns = []
            rationales = []
            rationales_attns = []
            inputs_image_id = []
            inputs_image_attn = []
            with torch.no_grad():
                for i in index_tuple:
                    element = train_dataset[i]
                    img_id, img_attn = get_original_img(element, model, tokenizer)
                    inputs_image_id.append(img_id)
                    inputs_image_attn.append(img_attn)
                    all_id, all_attn = get_mixed_question(element, model, tokenizer)
                    questions.append(all_id)
                    questions_attns.append(all_attn)
                    rat_id, rat_attn = get_tokenized_rationale(element, model, tokenizer)
                    rationales.append(rat_id)
                    rationales_attns.append(rat_attn)
                img_id = make_tensor(inputs_image_id)
                img_attn = make_tensor(inputs_image_attn)
                question_id = make_tensor(questions)
                question_mask = make_tensor(questions_attns)
                rationale_id = make_tensor(rationales)
                rationale_mask = make_tensor(rationales_attns)
                # inputs_image = torch.stack(inputs_image).squeeze(1)
                # language_model_inputs, language_attention_mask = get_image_queries(inputs_image, model)
                question_id = torch.cat((img_id.to(question_id.device), question_id), dim=1)
                question_mask = torch.cat((img_attn.to(question_mask.device), question_mask), dim=1)
                del rationale_mask,img_id, img_attn
                torch.cuda.empty_cache()
            
            loss = self_forward_t5(model,question_id,question_mask, rationale_id, cot_ids, cot_attns)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            tot_loss += loss.item()
            optimizer.zero_grad()
            progress_bar.update(1)
        print(tot_loss/n_choice)

    torch.save({
        'model_state_dict': model.language_model.state_dict(),
        'loss': tot_loss/(n_choice),
    }, './ckpts/checkpoint_language_model_'+str(timeNow)+'.pth')

def finetune_t5_2(path=None): 
    processor, tokenizer, model = get_and_freeze_model_t5(path)
    train_dataset = VCRTrainDataset(processor=processor, data_path="data/vcr", use_imagedict=False)
    n_samples = len(train_dataset)
    n_choice = 1000
    batch_size = 4
    indexes_list = random.choices(list(range(n_samples)), k=n_choice)
    indexes = [tuple([indexes_list[i+j] for j in range(0, min(batch_size, n_choice-i))]) for i in range(0, n_choice, batch_size)]
    # index_list_cot = [666, 890, 15678, 334]
    # with torch.no_grad():
    #     cot_ids, cot_attns = construct_cot_element(model, tokenizer, processor,  indexes_list=index_list_cot)
    num_epochs = 5
    num_training_steps = int(num_epochs * n_choice / batch_size)
    optimizer = Adafactor(model.language_model.parameters(), relative_step=True, warmup_init=True)
    lr_scheduler = AdafactorSchedule(optimizer)
    progress_bar = tqdm(range(num_training_steps))
    timeNow = datetime.datetime.now().strftime("%I:%M%p_on_%B_%d_%Y_")
    for epoch in range(num_epochs):
        tot_loss = 0
        for index_tuple in indexes:
            question = []
            rationales = []
            inputs_image = []
            for i in index_tuple:
                element = train_dataset[i]
                inputs_image.append(element[0])
                question_element = get_question_element_method5(element)
                rationale_element = element[6][element[7]] + " " + element[4][element[5]]
                question.append(question_element)
                rationales.append(rationale_element)
            inputs_image = torch.stack(inputs_image).squeeze(1)
            tokenized_rationale_ans = tokenizer(rationales, padding='longest', truncation=True, return_tensors='pt')
            rationale_id = tokenized_rationale_ans["input_ids"]
            rationale_mask = tokenized_rationale_ans["attention_mask"]
            question_encoded = tokenizer(question, padding='longest', truncation=True, return_tensors='pt')
            question_id = question_encoded["input_ids"]
            question_mask = question_encoded["attention_mask"]
            with torch.no_grad():
                question_embeds = model.get_input_embeddings()(question_id) 
            loss = self_forward_t5(model,question_embeds,question_mask, rationale_id)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            tot_loss += loss.item()
            optimizer.zero_grad()
            progress_bar.update(1)
        print(tot_loss/n_choice)

    torch.save({
        'model_state_dict': model.language_model.state_dict(),
        'loss': tot_loss/(n_choice),
    }, './ckpts/checkpoint_language_model_'+str(timeNow)+'.pth')
      

def finetune_rationale_answer_v2(path=None, use_boundingbox=True, finetune_vision_encoder=True, add_rationale=False):
    processor, model = get_and_freeze_model(path, finetune_vision_encoder=finetune_vision_encoder)
    train_dataset = AOKVQATrainDataset(processor=processor, data_path="data")
    n_samples = len(train_dataset)
    print(n_samples)
    n_choice = int(n_samples * 0.05)
    tokenizer = AutoTokenizer.from_pretrained(VERSION)
    batch_size = 2
    indexes_list = random.choices(list(range(n_samples)), k=n_choice)
    indexes = [tuple([indexes_list[i+j] for j in range(0, min(batch_size, n_choice-i))]) for i in range(0, n_choice, batch_size)]
    num_epochs = 8
    num_training_steps = int(num_epochs * n_choice / batch_size)
    progress_bar = tqdm(range(num_training_steps))
    timeNow = datetime.datetime.now().strftime("%I:%M%p_on_%B_%d_%Y_")
    optimizer = AdamW(model.parameters(), lr=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0, last_epoch=-1, verbose=False)
    ckpt_path = 'ckpts/checkpoint_'+str(use_boundingbox)+"_"+str(finetune_vision_encoder)+"_"+str(timeNow)+'.pth'
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
                    question_element = "Answer the following question by reasoning step by step. Question: "+element[1]+"\n\nAnswer: "
                question.append(question_element)
                if not add_rationale:
                    labels.append(convert_index_to_letter(element[5]))
                else:
                    labels.append(element[5]+" because "+element[3][0])
            inputs_image = torch.stack(inputs_image).squeeze(1)
            question_encoded = tokenizer(question, padding='longest', truncation=True, return_tensors='pt')
            question_id = question_encoded["input_ids"]
            question_mask = question_encoded["attention_mask"]
            labels_ids = tokenizer(labels, padding='longest', truncation=True, return_tensors='pt')["input_ids"]
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
            }, './ckpts/checkpoint_AOKVQA'+"_"+str(finetune_vision_encoder)+"_"+str(timeNow)+'.pth')
    return ckpt_path

def finetune_rationale_answer_vqax(path=None, use_boundingbox=True, finetune_vision_encoder=True, add_rationale=False):
    processor, model = get_and_freeze_model(path, finetune_vision_encoder=finetune_vision_encoder)
    train_dataset = VQAXTrainDataset(processor=processor, data_path="data")
    n_samples = len(train_dataset)
    print(n_samples)
    n_choice = int(n_samples)
    tokenizer = AutoTokenizer.from_pretrained(VERSION)
    batch_size = 2
    indexes_list = random.choices(list(range(n_samples)), k=n_choice)
    indexes = [tuple([indexes_list[i+j] for j in range(0, min(batch_size, n_choice-i))]) for i in range(0, n_choice, batch_size)]
    num_epochs = 3
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
                    question_element = "Answer the following question by reasoning step by step. Question: "+element[1]+"\n\nAnswer: "
                question.append(question_element)
                if not add_rationale:
                    labels.append(convert_index_to_letter(element[5]))
                else:
                    labels.append(element[4]+" because "+element[3][0])
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
            }, './ckpts/checkpoint_VQAX'+"_"+str(finetune_vision_encoder)+"_"+str(timeNow)+'.pth')
    return ckpt_path

def finetune_rationale_answer_vqax_self_train(path=None, use_boundingbox=True, finetune_vision_encoder=True, add_rationale=False):
    processor, model = get_and_freeze_model(path, finetune_vision_encoder=finetune_vision_encoder)
    train_dataset = VQAXEvalDataset(processor=processor, data_path="data")
    n_choice = 32
    tokenizer = AutoTokenizer.from_pretrained(VERSION)
    batch_size = 4
    num_epochs = 8
    num_training_steps = int(num_epochs * n_choice / batch_size)
    progress_bar = tqdm(range(num_training_steps))
    timeNow = datetime.datetime.now().strftime("%I:%M%p_on_%B_%d_%Y_")
    optimizer = AdamW(model.parameters(), lr=1e-6)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0, last_epoch=-1, verbose=False)
    ckpt_path = 'ckpts/checkpoint_VQAX'+"_"+str(finetune_vision_encoder)+"_"+str(timeNow)+'.pth'
    loss = 0
    with open('./filtered_train/vqax_correct_on_2', 'r') as f:
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
                    question_element = "Answer the following question by reasoning step by step. Question: "+element[1]+"\n\nAnswer: "
                question.append(question_element)
                if not add_rationale:
                    labels.append(convert_index_to_letter(element[5]))
                else:
                    labels.append(element[4]+" because "+caption)
                    print(element[4]+" because "+caption)
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
            }, './ckpts/checkpoint_VQAX_selftrain_givenright_wVGCoT_32shot'+"_"+str(finetune_vision_encoder)+"_"+str(timeNow)+'.pth')
    return ckpt_path

def finetune_rationale_answer_aokvqa_self_train(path=None, use_boundingbox=True, finetune_vision_encoder=True, add_rationale=False):
    processor, model = get_and_freeze_model(path, finetune_vision_encoder=finetune_vision_encoder)
    train_dataset = AOKVQAEvalDataset(processor=processor, data_path="data")
    n_choice = 32
    tokenizer = AutoTokenizer.from_pretrained(VERSION)
    batch_size = 4
    with open('./filtered_train/aokvqax_correct_on_1', 'r') as f:
        data_dict = json.load(f)
    indexes = list(data_dict.keys())
    indexes_list = random.choices(indexes, k=n_choice)
    indexes = [tuple([indexes_list[i+j] for j in range(0, min(batch_size, n_choice-i))]) for i in range(0, n_choice, batch_size)] 
    num_epochs = 8
    num_training_steps = int(num_epochs * n_choice / batch_size)
    progress_bar = tqdm(range(num_training_steps))
    timeNow = datetime.datetime.now().strftime("%I:%M%p_on_%B_%d_%Y_")
    optimizer = AdamW(model.parameters(), lr=1e-6)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0, last_epoch=-1, verbose=False)
    ckpt_path = 'ckpts/checkpoint_'+str(use_boundingbox)+"_"+str(finetune_vision_encoder)+"_"+str(timeNow)+'.pth'
    loss = 0
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
                    question_element = "Answer the following question by reasoning step by step. Question: "+element[1]+"\n\nAnswer: "
                question.append(question_element)
                if not add_rationale:
                    labels.append(convert_index_to_letter(element[5]))
                else:
                    labels.append(element[5]+" because "+caption)
            print(labels)
            inputs_image = torch.stack(inputs_image).squeeze(1)
            question_encoded = tokenizer(question, padding='longest', truncation=True, return_tensors='pt')
            question_id = question_encoded["input_ids"]
            question_mask = question_encoded["attention_mask"]
            labels_ids = tokenizer(labels, padding='longest', truncation=True, return_tensors='pt')["input_ids"]
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
            }, './ckpts/checkpoint_AOKVQA'+"_rightbefore"+str(finetune_vision_encoder)+"_"+str(timeNow)+'.pth')
    return ckpt_path

if __name__=='__main__':

    """vqax finetune"""
    path = finetune_rationale_answer_vqax(path='',finetune_vision_encoder=True, use_boundingbox=True, add_rationale=True)
    

    

