import torch, os, argparse
from transformers import Blip2Processor, Blip2Config, Blip2ForConditionalGeneration, Blip2QFormerConfig, AutoTokenizer
import numpy as np
import datasets
import json
import nltk
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
import sacrebleu
from accelerate import load_checkpoint_and_dispatch
from torchmetrics.functional.text.rouge import rouge_score
from sentence_transformers import SentenceTransformer, util
from PIL import Image
from torch.nn.functional import pad 
from lavis.models import load_model_and_preprocess
import PIL
from evaluate import load
VERSION = "Salesforce/blip2-flan-t5-xxl"
# different ways to construct a question prompt
def get_question_element_method1(element, colored_mask):
    question_element = "Question: "+element[3] + " Generate a rationale"
    cnt = 0
    for idx in colored_mask:
        if cnt == 0:
            question_element += " about the person in "
        else:
            question_element += " and the person in "
        question_element += COLOR_MASKS[idx] + " box"
        cnt += 1
    question_element += ":" + element[6][element[7]]
    return question_element


def convert_index_to_letter(index):
    if index == 0:
        return "(A)"
    elif index == 1:
        return "(B)"
    elif index == 2:
        return "(C)"
    return "(D)"

def get_question_element_method2(element):
    question_batch_with_answer = "Question: "+element[3] + " Generate a rationale"
    cnt = 0
    for person_name, box_name in zip(GENDER_NEUTRAL_NAMES,COLOR_MASKS_NAME):
        if box_name in element[3]:
            if cnt == 0:
                question_batch_with_answer += " about " + person_name + " in "
            else:
                question_batch_with_answer += " and "+person_name+" in "
            question_batch_with_answer += box_name
            cnt += 1
    question_batch_with_answer += ": " + element[6][element[7]]
    return question_batch_with_answer


def get_question_element_method3(element, use_img_region=True):
    question_element = "Question: " + element[3] + " Generate a rationale"
    cnt = 0
    idx = 100000
    want_name = ''
    want_person_name = ''
    for person_name, box_name in zip(GENDER_NEUTRAL_NAMES,COLOR_MASKS_NAME):
        if box_name in element[3]:
            if element[3].find(box_name) < idx:
                idx = element[3].find(box_name)
                want_name = box_name
                want_person_name = person_name
            if cnt == 0:
                question_element += " about "+person_name+" in "
            else:
                question_element += " and "+person_name+" in "
            question_element += box_name
            cnt += 1
    question_element += ": " 
    output_front_append = ''
    if cnt != 0:
        if use_img_region:
            img_dict = element[9]
            img_region = img_dict[want_person_name + " with "+want_name]
        else:
            img_region = None
        question_element += want_person_name + " with "
        question_element += want_name    
        output_front_append = want_person_name + " with "+want_name
    else:
        img_region = None
    
    return question_element, output_front_append, img_region

def get_tokenized_list(tokenizer):
    names = []
    for name in GENDER_NEUTRAL_NAMES:
        names.append(tokenizer.tokenize(name))
    return names

def get_mixed_question(element,model, tokenizer, not_cot=True):
    with torch.no_grad():
        img_dict = element[9]
        question = "Question: "+element[3]+"\n\nOptions: "+ "(A) "+element[4][0]+" (B) "+element[4][1]+" (C) " +element[4][2]+ " (D) "+element[4][3]+"\n\nAnswer: "
        # for name in GENDER_NEUTRAL_NAMES_ORG:
        #     if name not in img_dict and name in question:
        #         question = question.replace(name, 'Jessie')
        question_list = tokenizer.tokenize(question)
        tokenized_question = tokenizer(question, truncation=True, return_tensors='pt')
        question_only_list = tokenizer.tokenize("Question: "+element[3])
        input_id = tokenized_question['input_ids']
        inputs_embeds = model.get_input_embeddings()(input_id)
        input_attn = tokenized_question['attention_mask']
        img_ids_dict = {}
        all_id = []
        all_attn = []
        for i,o in enumerate(question_list):
            all_id.append(inputs_embeds[:, i:i+1, :])
            all_attn.append(input_attn[:, i:i+1])
            if o in GENDER_NEUTRAL_NAMES:
                o = o[1:]
                if o not in img_ids_dict:
                    img1 = img_dict[o]
                    language_model_inputs, language_attention_mask = get_image_queries(img1, model)
                    img_ids_dict[o] = (language_model_inputs, language_attention_mask)
                language_model_inputs, language_attention_mask = img_ids_dict[o]
                all_id.append(language_model_inputs[:, 3:8, :].to(inputs_embeds.device))
                all_attn.append(language_attention_mask[:, 3:8].to(input_attn.device))

        all_id.append(inputs_embeds[:, i+1:, :])
        all_attn.append(input_attn[:, i+1:])
    return torch.cat(tuple(all_id), dim=1).squeeze(0), torch.cat(tuple(all_attn), dim=1).squeeze(0)

def get_original_img(element, model, tokenizer):
    with torch.no_grad():
        img = element[0]
        language_model_inputs, language_attention_mask = get_image_queries(img, model)
        return language_model_inputs.squeeze(0), language_attention_mask.squeeze(0)

def make_tensor(tensor_list):
    out = torch.nn.utils.rnn.pad_sequence(tensor_list, batch_first=True)
    return out

def get_question_element_method4(element):
    question_element = "Question: "+element[3] + "\n\nChoices: "+ "(A) "+element[4][0]+" (B) "+element[4][1]+" (C) " +element[4][2]+ " (D) "+element[4][3] + "\n\nAnswer:"
    return question_element

def get_question_element_method5(element):
    question_element = "\n\nAnswer the following question by reasoning step by step. "+element[3]+"\n\nAnswer: "
    # question_element = "\n\nQuestion: "+element[3]+" Give the rationale before answering. Answer: "
    
    return question_element

def get_question_element_method6(element, rationale=''):
    question_element = "\n\nQuestion: "+element[3]+"\n\nOptions: "+ "(A) "+element[4][0]+" (B) "+element[4][1]+" (C) " +element[4][2]+ " (D) "+element[4][3]+ "\n\nAnswer: "+rationale+" so the choice is:"
    return question_element

def get_image_queries(pixel_values, model):
    with torch.no_grad():
        image_embeds = model.vision_model(pixel_values).last_hidden_state
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state
        language_model_inputs = model.language_projection(query_output) 
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        return language_model_inputs, language_attention_mask

# only keep the pixels inside the bounding box, and keep all the parts of the image zero
# choose the index with the largest 
def get_masked_image(path, boxes=None, processor=None):
    if processor is None:
        processor = Blip2Processor.from_pretrained(VERSION)
    x1 = int(boxes[0])
    y1 = int(boxes[1])
    x2 = int(boxes[2])
    y2 = int(boxes[3])
    image = Image.open(path).convert('RGB')
    img_shape = (image.size[0], image.size[1], 3)
    mask = np.zeros(img_shape)
    for i in range(x1, x2):
        for j in range(y1, y2):
            mask[i, j, :] = 1.0
    mask = np.transpose(mask, (1, 0, 2))
    image = image * mask
    image_pixels = processor(images=image,  return_tensors="pt")['pixel_values']
    return image_pixels

def get_most_relevent_query(model, image_pixels, top_k = 5):
    batch_size = image_pixels.shape[0]
    image_embeds = model.vision_model(image_pixels, return_dict=True).last_hidden_state
    image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

    query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
    query_outputs = model.qformer(
        query_embeds=query_tokens,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_attention_mask,
        return_dict=True,
    )
    query_output = query_outputs.last_hidden_state
    dot_product = query_output * query_output
    dot_product = dot_product.sum(2)
    sorted_dot_product, sorted_indices = torch.topk(dot_product, top_k, largest=False, dim=-1)
    sorted_indices, _ = torch.sort(sorted_indices, descending=False)
    return sorted_indices

def print_loss(path, rang):
    for i in range(rang):
        ckpt = path+str(i)+'.pth'
        checkpoint = torch.load(ckpt)
        loss1 = checkpoint['loss1']
        loss2 = checkpoint['loss2']
        print(loss1, loss2)

def get_mixed_rationale(element, model, tokenizer):
    with torch.no_grad():
        img_dict = element[9]
        question = "Question: "+element[3] + "\n\nGenerate a rationale"
        idx = 10000
        cnt = 0
        want_name = ''
        for name in GENDER_NEUTRAL_NAMES_ORG:
            if name in question:
                if question.find(name) < idx:
                    idx = question.find(name)
                    want_name = name
                cnt += 1
        if cnt == 0:
            question += ":"
        else:
            question += " about "+want_name + ":"
        question_list = tokenizer.tokenize(question)
        tokenized_question = tokenizer(question, truncation=True, return_tensors='pt')
        input_id = tokenized_question['input_ids']
        inputs_embeds = model.get_input_embeddings()(input_id)
        input_attn = tokenized_question['attention_mask']
        img_ids_dict = {}
        all_id = []
        all_attn = []
        for i,o in enumerate(question_list):
            all_id.append(inputs_embeds[:, i:i+1, :])
            all_attn.append(input_attn[:, i:i+1])
            if o in GENDER_NEUTRAL_NAMES:
                o = o[1:]
                if o not in img_ids_dict:
                    img1 = img_dict[o]
                    language_model_inputs, language_attention_mask = get_image_queries(img1, model)
                    img_ids_dict[o] = (language_model_inputs, language_attention_mask)
                language_model_inputs, language_attention_mask = img_ids_dict[o]
                all_id.append(language_model_inputs[:, 3:8, :].to(inputs_embeds.device))
                all_attn.append(language_attention_mask[:, 3:8].to(input_attn.device))
        all_id.append(inputs_embeds[:, i+1:, :])
        all_attn.append(input_attn[:, i+1:])
    return torch.cat(tuple(all_id), dim=1).squeeze(0), torch.cat(tuple(all_attn), dim=1).squeeze(0)

def get_tokenized_rationale(element, model, tokenizer):
    with torch.no_grad():
        rationale =  element[6][element[7]] + " so " + element[4][element[5]]
        tokenized_rationale = tokenizer(rationale, truncation=True, return_tensors='pt')
        input_id = tokenized_rationale['input_ids']
        input_attn = tokenized_rationale['attention_mask']
        return input_id.squeeze(0), input_attn.squeeze(0)

def get_and_freeze_model_t5(path=None):
    processor = Blip2Processor.from_pretrained(VERSION)
    device = "cuda"
    model = Blip2ForConditionalGeneration.from_pretrained(VERSION, device_map="auto") #, load_in_8bit=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(VERSION)

    if path is not None:
        checkpoint = torch.load(path, map_location='cpu')
        ckpt_pth = checkpoint['model_state_dict']
        model.qformer.load_state_dict(ckpt_pth)

    model.train()
    for param in model.parameters():
        param.requires_grad = False
    for param in model.language_model.parameters():
        param.requires_grad = True

    return processor, tokenizer, model

def construct_cot_examples(model, tokenizer, processor, path=None, indexes_list=None, use_boundingbox=False):
    train_dataset = VCRTrainDataset(processor=processor, data_path="data/vcr", use_boundingbox=use_boundingbox, use_imagedict=False)
    n_samples = len(train_dataset)
    if indexes_list == None:
        n_choice = 4
        indexes_list = random.choices(list(range(n_samples)), k=n_choice)
        with open('cot_choices', 'a') as f:
            for i in indexes_list:
                f.write(str(i)+'\n')
    with torch.no_grad():
        num = 0
        for idx in indexes_list:
            element = train_dataset[idx]
            pixel_values = element[0]
            colored_mask = element[8]
            question_batch_with_answer = "Answer the following question by reasoning step by step.  "+element[3]+"\n\nOptions: "+ "(A) "+element[4][0]+" (B) "+element[4][1]+" (C) " +element[4][2]+ " (D) "+element[4][3]+"\n\nAnswer: " + element[6][element[7]] + " so the final answer is: "+convert_index_to_letter(element[5])
            print(question_batch_with_answer)
            #question_batch_with_answer = "\n\nAnswer the question by reasoning step by step. "+element[3]+"\n\nOptions: "+ "(A) "+element[4][0]+" (B) "+element[4][1]+" (C) " +element[4][2]+ " (D) "+element[4][3]+ "\n\nAnswer: " + convert_index_to_letter(element[5])+" because "+element[6][element[7]] 
            question_batch_with_answer_2 = "Question: "+element[3] + " Generate a rationale"
            if "Why" in element[3]:
                question_batch_with_answer_2 += " about why"
            question_batch_with_answer_2 += ": " + element[6][element[7]] 
            
            inputs = tokenizer(question_batch_with_answer, padding='longest', truncation=True, return_tensors='pt')
            inputs_2 = tokenizer(question_batch_with_answer_2, padding='longest', truncation=True, return_tensors='pt')
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            input_ids_2 = inputs_2["input_ids"]
            attention_mask_2 = inputs_2["attention_mask"]
            image_embeds = model.vision_model(pixel_values, return_dict=True).last_hidden_state
            image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

            query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs = model.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                return_dict=True,
            )
            query_output = query_outputs.last_hidden_state
            language_model_inputs = model.language_projection(query_output) 
            language_attention_mask = torch.ones(
                language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
            )
            inputs_embeds = model.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)
            attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)

            inputs_embeds_2 = model.get_input_embeddings()(input_ids_2)
            inputs_embeds_2 = torch.cat([language_model_inputs, inputs_embeds_2.to(language_model_inputs.device)], dim=1)
            attention_mask_2 = torch.cat([language_attention_mask, attention_mask_2.to(language_attention_mask.device)], dim=1)
            
            if num == 0:
                inputs_all_ids = inputs_embeds
                attns_all_ids = attention_mask
                inputs_all_ids_2 = inputs_embeds_2
                attns_all_ids_2 = attention_mask_2
            else:
                inputs_all_ids = torch.cat([inputs_all_ids, inputs_embeds], dim=1)
                attns_all_ids = torch.cat([attns_all_ids, attention_mask], dim=1)
                inputs_all_ids_2 = torch.cat([inputs_all_ids_2, inputs_embeds_2], dim=1)
                attns_all_ids_2 = torch.cat([attns_all_ids_2, attention_mask_2], dim=1)
            num += 1
            print(question_batch_with_answer)
        return inputs_all_ids, attns_all_ids, inputs_all_ids_2, attns_all_ids_2

def construct_cot_element(model, tokenizer, processor, indexes_list):
    train_dataset = VCRTrainDataset(processor=processor, data_path="data/vcr", use_boundingbox=False)
    n_samples = len(train_dataset)
    for num, idx in enumerate(indexes_list):
        element = train_dataset[idx]
        inputs_embeds, attention_mask = get_mixed_question(element,model, tokenizer, not_cot=False)
        language_model_inputs, language_attention_mask =  get_image_queries(element[0], model)
        rationale_ids, rationale_attention_mask = get_tokenized_rationale(element, model, tokenizer)
        rationale_embeddings = model.get_input_embeddings()(rationale_ids)
        rationale_embeddings = rationale_embeddings.unsqueeze(0)
        rationale_attention_mask = rationale_attention_mask.unsqueeze(0)
        inputs_embeds = inputs_embeds.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        inputs_embeds = torch.cat([inputs_embeds, rationale_embeddings.to(inputs_embeds.device)], dim=1)
        attention_mask = torch.cat([attention_mask, rationale_attention_mask.to(attention_mask.device)], dim=1)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)
        attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)
        
        if num == 0:
            inputs_all_ids = inputs_embeds
            attns_all_ids = attention_mask
        else:
            inputs_all_ids = torch.cat([inputs_all_ids, inputs_embeds], dim=1)
            attns_all_ids = torch.cat([attns_all_ids, attention_mask], dim=1)
    return inputs_all_ids, attns_all_ids


def get_feature_model():
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device='cpu')
    print(txt_processors)
    return model, txt_processors

model_feature, txt_processors = get_feature_model()
model_feature.eval()
def get_word_embeddings(text, tokenizer=None, model=model_feature):
    if tokenizer == None:
        tokenizer = model.tokenizer
    input_ids = tokenizer(text, truncation=True, padding=True, return_tensors='pt')
    embeddings = model.Qformer.bert.embeddings.word_embeddings(input_ids["input_ids"])
    return embeddings, input_ids["attention_mask"]


