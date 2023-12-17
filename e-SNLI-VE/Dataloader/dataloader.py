import torch
from torch.utils.data import Dataset
import os, json
from copy import deepcopy
from PIL import Image
from skimage.draw import polygon
from transformers import Blip2Processor


class eSNLITrainDataset(Dataset):
    def __init__(self, processor, ann_file_detailed='esnlive_train.json', image_set='flickr30k_images', data_path='../data', 
                 use_boundingbox=True, 
                 **kwargs):
        
        super(eSNLITrainDataset, self).__init__()
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
        
        hypothesis = meta_data['hypothesis']
        image_id_pad = meta_data["image_name"]
        rationale = meta_data["explanation"]
        answer = meta_data["answers"]
        if answer == 'contradiction':
            answer = 'No'
        elif answer == 'neutral':
            answer = "It's impossible to say"
        else:
            answer = 'Yes'
        image_path = os.path.join(self.data_path, self.image_set, image_id_pad)
        image = self._load_image(image_path)

        if self.processor != None:
            image = self.processor(images=image,  return_tensors="pt")['pixel_values']
        
        
        outputs = (image, 
                    hypothesis,
                    answer,
                    rationale,
                    image_id_pad
                    )

        return outputs

    def __len__(self):
        return len(self.indexes)
    
class eSNLIEvalDataset(Dataset):
    def __init__(self, processor, ann_file_detailed='esnlive_test.json', image_set='flickr30k_images', data_path='../data', 
                 use_boundingbox=True,use_imgpath=False, 
                 **kwargs):
        
        super(eSNLIEvalDataset, self).__init__()
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
        
        hypothesis = meta_data['hypothesis']
        image_id_pad = meta_data["image_name"]
        rationale = meta_data["explanation"]
        answer = meta_data["answers"]
        if answer == 'contradiction':
            answer = 'No'
        elif answer == 'neutral':
            answer = "It's impossible to say"
        else:
            answer = 'Yes' 
        image_path = os.path.join(self.data_path, self.image_set, image_id_pad)
        image = self._load_image(image_path)

        if self.processor != None:
            image = self.processor(images=image,  return_tensors="pt")['pixel_values']
        
        
        outputs = (image, 
                    hypothesis,
                    answer,
                    rationale,
                    image_id_pad
                    )
        if self.use_imgpath:
           outputs = (image, 
                    hypothesis,
                    answer,
                    rationale,
                    image_id_pad, 
                    image_path
                    ) 
        return outputs

    def __len__(self):
        return len(self.indexes)


if __name__=='__main__':
    VERSION = "Salesforce/blip2-flan-t5-xxl"
    processor = Blip2Processor.from_pretrained(VERSION)
    eval_dataset = eSNLIEvalDataset(processor=processor,use_boundingbox=False)

    