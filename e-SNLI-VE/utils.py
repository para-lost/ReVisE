import torch, os, argparse
from transformers import Blip2Processor, Blip2Config, Blip2ForConditionalGeneration, Blip2QFormerConfig, AutoTokenizer
import numpy as np
import datasets
import json
import nltk
import torch.nn as nn
import sys
import random
import numpy as np
from torch.nn import DataParallel
from torch.nn.modules import BatchNorm2d
from tqdm import tqdm
from Dataloader.dataloader import *
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from accelerate import load_checkpoint_and_dispatch
from torchmetrics.functional.text.rouge import rouge_score
from sentence_transformers import SentenceTransformer, util
from PIL import Image
from torch.nn.functional import pad 
from lavis.models import load_model_and_preprocess
import PIL
from evaluate import load
VERSION = "Salesforce/blip2-flan-t5-xxl"


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

def construct_cot_examples(model, tokenizer, processor, path=None, indexes_list=None, use_boundingbox=False):
    train_dataset = eSNLITrainDataset(processor=processor, data_path="data", use_boundingbox=use_boundingbox, use_imagedict=False)
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
            question_batch_with_answer = "Based on the image can we conclude that '"+element[1]+"'?"+"\n\nAnswer: "+element[2]+" because "+element[3]
            print(question_batch_with_answer)
            
            inputs = tokenizer(question_batch_with_answer, padding='longest', truncation=True, return_tensors='pt')
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
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

            if num == 0:
                inputs_all_ids = inputs_embeds
                attns_all_ids = attention_mask
            else:
                inputs_all_ids = torch.cat([inputs_all_ids, inputs_embeds], dim=1)
                attns_all_ids = torch.cat([attns_all_ids, attention_mask], dim=1)   
            num += 1
            print(question_batch_with_answer)
        return inputs_all_ids, attns_all_ids



