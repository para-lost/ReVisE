import torch
from torch.utils.data import Dataset
from torchvision import datasets
import os, jsonlines, json
from copy import deepcopy
from PIL import Image, ImageColor, ImageDraw
from skimage.draw import polygon
from transformers import Blip2Processor
import re
contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve":
    "could've", "couldnt": "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt":
    "doesn't", "dont": "don't", "hadnt": "hadn't", "hadnt've":
    "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent":
    "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve":
    "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im":
    "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've":
    "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've":
    "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat":
    "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve":
    "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt":
    "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve":
    "shouldn't've", "somebody'd": "somebodyd", "somebodyd've":
    "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll":
    "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd":
    "something'd", "somethingd've": "something'd've", "something'dve":
    "something'd've", "somethingll": "something'll", "thats":
    "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres":
    "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve":
    "they'd've", "theyll": "they'll", "theyre": "they're", "theyve":
    "they've", "twas": "'twas", "wasnt": "wasn't", "wed've":
    "we'd've", "we'dve": "we'd've", "weve": "we've", "werent":
    "weren't", "whatll": "what'll", "whatre": "what're", "whats":
    "what's", "whatve": "what've", "whens": "when's", "whered":
    "where'd", "wheres": "where's", "whereve": "where've", "whod":
    "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl":
    "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve":
    "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll":
    "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd":
    "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll":
    "you'll", "youre": "you're", "youve": "you've"
}

manual_map = {'none': '0',
              'zero': '0',
              'one': '1',
              'two': '2',
              'three': '3',
              'four': '4',
              'five': '5',
              'six': '6',
              'seven': '7',
              'eight': '8',
               'nine': '9',
              'ten': '10'}
articles = ['a', 'an', 'the']
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [';', r"/", '[', ']', '"', '{', '}',
                '(', ')', '=', '+', '\\', '_', '-',
                '>', '<', '@', '`', ',', '?', '!']
def process_punctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) \
           or (re.search(comma_strip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText

def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText

def prep_ans(answer):
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(',', '')
    return answer
def proc_ans(ans):
    ans_prob_dict = {}

    for ans_ in ans:
        ans_proc = prep_ans(ans_['answer'])
        if ans_proc not in ans_prob_dict:
            ans_prob_dict[ans_proc] = 1
        else:
            ans_prob_dict[ans_proc] += 1

    confident_answer = max(ans_prob_dict, key=ans_prob_dict.get)
    return confident_answer
class VQAEvalDataset(Dataset):
    def __init__(self, processor, ann_file='OpenEnded_mscoco_val2014_questions.json', ann_file_detailed='mscoco_val2014_annotations.json', image_set='val2014', data_path='../data', transform=None, 
                 n_samples=1000,use_boundingbox=True, 
                 **kwargs):
        
        super(VQAEvalDataset, self).__init__()
        self.data_path = data_path
        self.image_set = image_set
        self.ann_file = os.path.join(data_path, ann_file)
        self.ann_file_detailed = os.path.join(data_path, ann_file_detailed)
        self.vqa_answers, self.vqa_questions = self.load_annotations(self.ann_file, self.ann_file_detailed)
        self.processor = processor
        self.use_boundingbox = use_boundingbox
        
    def load_annotations(self, ann_file, ann_file_detailed):
        with open(ann_file, 'r') as json_file:
            vqa_val = json.load(json_file)
        vqa_test_questions = vqa_val["questions"]

        with open(ann_file_detailed, 'r') as json_file:
            vqa_val_ans = json.load(json_file)
            vqa_val_ans = vqa_val_ans["annotations"]
        qqa = {}
        for ann in vqa_val_ans:
            answer_dict = []
            for answer in ann['answers']:
                if answer["answer"] not in answer_dict:
                    answer_dict.append(answer["answer"])
                if answer["raw_answer"] not in answer_dict:
                    answer_dict.append(answer["raw_answer"])
            qqa[ann['question_id']] = answer_dict
        return qqa, vqa_test_questions

    def _load_image(self, path):
        return Image.open(path).convert('RGB')  

    def __getitem__(self, index):
        question_metadata = self.vqa_questions[index]
        question = question_metadata['question']
        image_id = str(question_metadata['image_id'])
        image_id_pad = "COCO_val2014_"+str(image_id.zfill(12))+'.jpg'
        
        question_id  = question_metadata['question_id']
        answer_list = self.vqa_answers[question_id]
        
        image_path = os.path.join(self.data_path, self.image_set, image_id_pad)
        image = self._load_image(image_path)
        image = self.processor(images=image,  return_tensors="pt")['pixel_values']

        outputs = (image, 
                    question,
                    answer_list)

        return outputs

    def __len__(self):
        return len(self.vqa_questions)

class AOKVQAEvalDataset(Dataset):
    def __init__(self, processor, ann_file_detailed='aokvqa_v1p0_val.json', image_set='val2014', data_path='../data', transform=None, 
                 n_samples=1000,use_boundingbox=True, 
                 **kwargs):
        
        super(AOKVQAEvalDataset, self).__init__()
        self.data_path = data_path
        self.image_set = image_set
        self.ann_file_detailed = os.path.join(data_path, ann_file_detailed)
        self.vqa_meta_data = self.load_annotations(self.ann_file_detailed)
        self.processor = processor
        self.use_boundingbox = use_boundingbox
        
    def load_annotations(self, ann_file_detailed):
        with open(ann_file_detailed, 'r') as json_file:
            vqa_val_ans = json.load(json_file)
        
        return vqa_val_ans

    def _load_image(self, path):
        return Image.open(path).convert('RGB')  

    def __getitem__(self, index):
        question_metadata = self.vqa_meta_data[index]
        question = question_metadata['question']
        answer_list = question_metadata["direct_answers"]
        rationale_list = question_metadata["rationales"]
        choice_list = question_metadata["choices"]
        image_id = str(question_metadata['image_id'])
        image_id_pad = "COCO_val2014_"+str(image_id.zfill(12))+'.jpg'
        image_path = os.path.join(self.data_path, self.image_set, image_id_pad)
        image = self._load_image(image_path)
        if self.processor != None:
            image = self.processor(images=image,  return_tensors="pt")['pixel_values']

        answer = answer_list[0]
        for choice in choice_list:
            if choice in answer_list:
                answer = choice
        outputs = (image, 
                    question,
                    answer_list, 
                    rationale_list,
                    choice_list,
                    answer)

        return outputs

    def __len__(self):
        return len(self.vqa_meta_data)

class AOKVQATrainDataset(Dataset):
    def __init__(self, processor, ann_file_detailed='aokvqa_v1p0_train.json', image_set='train2014', image_set2='val2014', data_path='../data', transform=None, 
                 n_samples=1000,use_boundingbox=True, 
                 **kwargs):
        
        super(AOKVQATrainDataset, self).__init__()
        self.data_path = data_path
        self.image_set = image_set
        self.image_set2 = image_set2
        self.ann_file_detailed = os.path.join(data_path, ann_file_detailed)
        self.vqa_meta_data = self.load_annotations(self.ann_file_detailed)
        self.processor = processor
        self.use_boundingbox = use_boundingbox
        
    def load_annotations(self, ann_file_detailed):
        with open(ann_file_detailed, 'r') as json_file:
            vqa_val_ans = json.load(json_file)
        
        return vqa_val_ans

    def _load_image(self, path):
        return Image.open(path).convert('RGB')  

    def __getitem__(self, index):
        question_metadata = self.vqa_meta_data[index]
        question = question_metadata['question']
        answer_list = question_metadata["direct_answers"]
        rationale_list = question_metadata["rationales"]
        choice_list = question_metadata["choices"]
        image_id = str(question_metadata['image_id'])
        image_id_pad = "COCO_train2014_"+str(image_id.zfill(12))+'.jpg'
        image_id_pad2 = "COCO_val2014_"+str(image_id.zfill(12))+'.jpg'
        image_path = os.path.join(self.data_path, self.image_set, image_id_pad)
        image_path2 = os.path.join(self.data_path, self.image_set2, image_id_pad2)
        if os.path.exists(image_path):
            image = self._load_image(image_path)
        else:
            image = self._load_image(image_path2)
        if self.processor != None:
            image = self.processor(images=image,  return_tensors="pt")['pixel_values']
        answer = answer_list[0]
        for choice in choice_list:
            if choice in answer_list:
                answer = choice
        outputs = (image, 
                    question,
                    answer_list, 
                    rationale_list,
                    choice_list,
                    answer)

        return outputs

    def __len__(self):
        return len(self.vqa_meta_data)


class VQAXEvalDataset(Dataset):
    def __init__(self, processor, ann_file_detailed='vqaX_test.json', image_set='val2014', data_path='../data', transform=None, 
                 n_samples=1000,use_boundingbox=True, use_imgpath=False, 
                 **kwargs):
        
        super(VQAXEvalDataset, self).__init__()
        self.data_path = data_path
        self.image_set = image_set
        self.ann_file_detailed = os.path.join(data_path, ann_file_detailed)
        self.data = self.load_annotations(self.ann_file_detailed)
        self.indexes = list(self.data.keys())
        self.processor = processor
        self.use_boundingbox = use_boundingbox
        self.use_imgpath = use_imgpath
        
    def load_annotations(self, ann_file_detailed):
        with open(ann_file_detailed, 'r') as json_file:
            vqa_val = json.load(json_file)
        return vqa_val

    def _load_image(self, path):
        return Image.open(path).convert('RGB')  
    
   
    def __getitem__(self, index):
        index_id = self.indexes[index]
        meta_data = self.data[index_id]
        
        question = meta_data['question']
        image_id_pad = meta_data["image_name"]
        rationale = meta_data["explanation"]
        answer_list = []
        answer_dict = meta_data["answers"]
        for item in answer_dict:
            if item["answer"] not in answer_list:
                answer_list.append(item["answer"])

        image_path = os.path.join(self.data_path, self.image_set, image_id_pad)
        image = self._load_image(image_path)

        if self.processor != None:
            image = self.processor(images=image,  return_tensors="pt")['pixel_values']
        best_answer = proc_ans(answer_dict)
        
        outputs = (image, 
                    question,
                    answer_list,
                    rationale,
                    best_answer,
                    image_id_pad,
                    )
        if self.use_imgpath:
            outputs = (image, 
                    question,
                    answer_list,
                    rationale,
                    best_answer,
                    image_id_pad,
                    image_path
                    )
        return outputs

    def __len__(self):
        return len(self.indexes)

class VQAXTrainDataset(Dataset):
    def __init__(self, processor, ann_file_detailed='vqaX_train.json', image_set='train2014', image_set2='val2014', data_path='../data', transform=None, 
                 n_samples=1000,use_boundingbox=True, 
                 **kwargs):
        
        super(VQAXTrainDataset, self).__init__()
        self.data_path = data_path
        self.image_set = image_set
        self.ann_file_detailed = os.path.join(data_path, ann_file_detailed)
        self.data = self.load_annotations(self.ann_file_detailed)
        self.indexes = list(self.data.keys())
        self.processor = processor
        self.use_boundingbox = use_boundingbox
        
    def load_annotations(self, ann_file_detailed):
        with open(ann_file_detailed, 'r') as json_file:
            vqa_val = json.load(json_file)
        return vqa_val

    def _load_image(self, path):
        return Image.open(path).convert('RGB')  

    def __getitem__(self, index):
        index_id = self.indexes[index]
        meta_data = self.data[index_id]
        
        question = meta_data['question']
        image_id_pad = meta_data["image_name"]
        rationale = meta_data["explanation"]
        answer_list = []
        answer_dict = meta_data["answers"]
        for item in answer_dict:
            if item["answer"] not in answer_list:
                answer_list.append(item["answer"])
        if "train" in image_id_pad:
            image_path = os.path.join(self.data_path, self.image_set, image_id_pad)
            image = self._load_image(image_path)
        else:
            image_path = os.path.join(self.data_path, self.image_set2, image_id_pad)
            image = self._load_image(image_path)

        if self.processor != None:
            image = self.processor(images=image,  return_tensors="pt")['pixel_values']

        best_answer = proc_ans(answer_dict)
        
        outputs = (image, 
                    question,
                    answer_list,
                    rationale,
                    best_answer,
                    image_id_pad
                    )

        return outputs

    def __len__(self):
        return len(self.indexes)

if __name__=='__main__':
    VERSION = "Salesforce/blip2-flan-t5-xxl"
    processor = Blip2Processor.from_pretrained(VERSION)
    eval_dataset = VQAXEvalDataset(processor=processor,use_boundingbox=False)

    