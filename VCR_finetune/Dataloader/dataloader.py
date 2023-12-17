import torch
from torch.utils.data import Dataset
from torchvision import datasets
import os, jsonlines, json
from copy import deepcopy
from PIL import Image, ImageColor, ImageDraw
from skimage.draw import polygon
from transformers import Blip2Processor
# GENDER_NEUTRAL_NAMES = ['Riley', 'Jackie', 'Kerry', 'Kendall',
#                         'Pat', 'Noah', 'James', 'Madison', 'Harper', 'Amy', 'Joey', 'Cindy', 'Tom']
GENDER_NEUTRAL_NAMES = ['person1', 'person2', 'person3', 'person4',
                        'person5', 'person6', 'person7', 'person8', 'person9', 'person10', 'person11', 'person12', 'person13']
# GENDER_NEUTRAL_NAMES = ['Riley', 'Jackie', 'Kerry', 'Kendall', 'Pat', 'Quin', 'Riley', 'Jackie', 'Kerry', 'Kendall', 'Pat', 'Quin', 'Jack']
COLOR_MASKS = ['red', 'yellow', 'blue', 'green', 'purple', 'white', 'black', 'brown', 'orange', 'pink', 'gold', 'silver','gray']
COLOR_TUPLES = [(255, 0, 0, 64),(255, 255, 0, 64),(0, 0, 255, 64),(0, 128, 0, 64),
(128, 0, 128, 64),(255, 255, 255, 64),(0, 0, 0, 64),(165, 42, 42, 64),(255, 165, 0, 64),(255, 192, 203, 64),
(255, 215, 0, 64),(192, 192, 192, 64),(128, 128, 128, 64)]
def generate_instance_mask(seg_polys, box, mask_size=(14, 14), dtype=torch.float32, copy=True):
    """
    Generate instance mask from polygon
    :param seg_poly: torch.Tensor, (N, 2), (x, y) coordinate of N vertices of segmented foreground polygon
    :param box: array-like, (4, ), (xmin, ymin, xmax, ymax), instance bounding box
    :param mask_size: tuple, (mask_height, mask_weight)
    :param dtype: data type of generated mask
    :param copy: whether copy seg_polys to a new tensor first
    :return: torch.Tensor, of mask_size, instance mask
    """
    mask = torch.zeros(mask_size, dtype=dtype)
    w_ratio = float(mask_size[0]) / (box[2] - box[0] + 1)
    h_ratio = float(mask_size[1]) / (box[3] - box[1] + 1)

    # import IPython
    # IPython.embed()

    for seg_poly in seg_polys:
        if copy:
            seg_poly = seg_poly.detach().clone()
        seg_poly = seg_poly.type(torch.float32)
        seg_poly[:, 0] = (seg_poly[:, 0] - box[0]) * w_ratio
        seg_poly[:, 1] = (seg_poly[:, 1] - box[1]) * h_ratio
        rr, cc = polygon(seg_poly[:, 1].clamp(min=0, max=mask_size[1] - 1),
                         seg_poly[:, 0].clamp(min=0, max=mask_size[0] - 1))

        mask[rr, cc] = 1
    return mask

class VCRTrainDataset(Dataset):
    def __init__(self, processor=None, ann_file='train.jsonl', image_set='vcr1images', data_path='../data/vcr', transform=None, task='Q2A', test_mode=False,
                 only_use_relevant_dets=False, add_image_as_a_box=False, mask_size=(14, 14),
                 basic_align=False, 
                 seq_len=64,n_samples=1000,random_choice=True,use_boundingbox=True, use_imagedict=True,
                 **kwargs):
        """
        Visual Commonsense Reasoning Dataset
        :param ann_file: annotation jsonl file
        :param image_set: image folder name, e.g., 'vcr1images'
        :param root_path: root path to cache database loaded from annotation file
        :param data_path: path to vcr dataset
        :param transform: transform
        :param task: 'Q2A' means question to answer, 'QA2R' means question and answer to rationale,
                     'Q2AR' means question to answer and rationale
        :param test_mode: test mode means no labels available
        :param cache_mode: cache whole dataset to RAM first, then __getitem__ read them from RAM
        :param ignore_db_cache: ignore previous cached database, reload it from annotation file
        :param tokenizer: default is BertTokenizer from pytorch_pretrained_bert
        :param only_use_relevant_dets: filter out detections not used in query and response
        :param add_image_as_a_box: add whole image as a box
        :param mask_size: size of instance mask of each object
        :param aspect_grouping: whether to group images via their aspect
        :param basic_align: align to tokens retokenized by basic_tokenizer
        :param qa2r_noq: in QA->R, the query contains only the correct answer, without question
        :param qa2r_aug: in QA->R, whether to augment choices to include those with wrong answer in query
        :param kwargs:
        """
        super(VCRTrainDataset, self).__init__()

        assert task in ['Q2A', 'QA2R', 'Q2AR'] , 'not support task {}'.format(task)
        
        self.seq_len = seq_len

        categories = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                      'trafficlight', 'firehydrant', 'stopsign', 'parkingmeter', 'bench', 'bird', 'cat', 'dog', 'horse',
                      'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                      'suitcase', 'frisbee', 'skis', 'snowboard', 'sportsball', 'kite', 'baseballbat', 'baseballglove',
                      'skateboard', 'surfboard', 'tennisracket', 'bottle', 'wineglass', 'cup', 'fork', 'knife', 'spoon',
                      'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hotdog', 'pizza', 'donut',
                      'cake', 'chair', 'couch', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tv', 'laptop', 'mouse',
                      'remote', 'keyboard', 'cellphone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                      'clock', 'vase', 'scissors', 'teddybear', 'hairdrier', 'toothbrush']
        self.category_to_idx = {c: i for i, c in enumerate(categories)}
        self.data_path = data_path
        self.ann_file = os.path.join(data_path, ann_file)
        self.image_set = image_set
        self.transform = transform
        self.task = task
        self.test_mode = test_mode


        self.add_image_as_a_box = add_image_as_a_box
        self.mask_size = mask_size

        self.database = self.load_annotations(self.ann_file)
        self.person_name_id = 0
        self.random_choice = random_choice
        self.processor = processor
        self.use_boundingbox = use_boundingbox
        self.use_imagedict = use_imagedict

    def load_annotations(self, ann_file):
        database = []
        with jsonlines.open(ann_file) as reader:
            for ann in reader:
                img_fn = os.path.join(self.data_path, self.image_set, ann['img_fn'])
                metadata_fn = os.path.join(self.data_path, self.image_set, ann['metadata_fn'])

                db_i = {
                    'annot_id': ann['annot_id'],
                    'objects': ann['objects'],
                    'img_fn': img_fn,
                    'metadata_fn': metadata_fn,
                    'question': ann['question'],
                    'answer_choices': ann['answer_choices'],
                    'answer_label': ann['answer_label'] if not self.test_mode else None,
                    'rationale_choices': ann['rationale_choices'],
                    'rationale_label': ann['rationale_label'] if not self.test_mode else None,
                }
                database.append(db_i)
        return database

    def get_raw(self, tokens, objects_replace_name, non_obj_tag=-1):
        raw = []
        for mixed_token in tokens:
            if isinstance(mixed_token, list):
                tokens = [objects_replace_name[o] for o in mixed_token]
                raw.append(tokens[0])
                for token in tokens[1:]:
                    raw.extend(['and', token])
            else:
                raw.append(mixed_token)
        raw_string = ""
        for item in raw:
            if item == '?' or item == '.' or item == "'" or item == '"' or item == ',':
                raw_string += item
            else:
                raw_string += " "
                raw_string += item
        return raw_string

    def _load_image(self, path, objects_color_and_box):
        if self.use_boundingbox:
            img = Image.open(path).convert('RGB')
            img1 = ImageDraw.Draw(img)
            draw = ImageDraw.Draw(img, 'RGBA') 
            for name,color,idx,box in objects_color_and_box:
                box = box[:4]
                img1.rectangle(box, outline =color)
                draw.rectangle(box, fill=COLOR_TUPLES[idx])
            
            return img
        else:
            return Image.open(path).convert('RGB')

    def _get_image_dict(self, path, objects_color_and_box):
        img_dict = {}
        if self.use_imagedict:
            img = Image.open(path).convert('RGB')
            for name,color,idx, box in objects_color_and_box:
                box = box[:4]
                img1 = img.crop(box)
                img1 = self.processor(images=img1, return_tensors='pt')['pixel_values']
                img_dict[name] = img1
            return img_dict
        else:
            return img_dict

    def _get_needed_idx(self, question, answers, rationales):
        tot_list = []
        for mixed_token in question:
            if isinstance(mixed_token, list):
                for o in mixed_token:
                    tot_list.append(o)
        for answer in answers:
            for mixed_token in answer:
                if isinstance(mixed_token, list):
                    for o in mixed_token:
                        tot_list.append(o)
        for rationale in rationales:
            for mixed_token in rationale:
                if isinstance(mixed_token, list):
                    for o in mixed_token:
                        tot_list.append(o)
        return tot_list

    def _load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def __getitem__(self, index):
        start = 0
        idb = deepcopy(self.database[index])
        metadata = self._load_json(idb['metadata_fn'])
        idb['boxes'] = metadata['boxes']
        idb['segms'] = metadata['segms']
        objects_replace_name = []
        objects_color_and_box = []
        colored_mask = []
        tot_list = self._get_needed_idx(idb["question"], [answer for answer in idb['answer_choices']], [rationale for rationale in idb['rationale_choices']])
        # print(tot_list)
        for num, (o, box) in enumerate(zip(idb['objects'], idb['boxes'])):
            if o == 'person':
                if not self.use_boundingbox:
                    objects_replace_name.append(GENDER_NEUTRAL_NAMES[start])
                    objects_color_and_box.append((GENDER_NEUTRAL_NAMES[start], "none",box))
                    colored_mask.append(0)
                    start = (start + 1) % len(GENDER_NEUTRAL_NAMES)
                else:
                    if num in tot_list:
                        objects_replace_name.append(GENDER_NEUTRAL_NAMES[start]+ " in "+COLOR_MASKS[start]+" region")
                        objects_color_and_box.append((GENDER_NEUTRAL_NAMES[start]+ " in "+COLOR_MASKS[start]+" region", COLOR_MASKS[start], start, box))
                        colored_mask.append(start)
                        start = (start + 1) % len(GENDER_NEUTRAL_NAMES)
                    else:
                        objects_replace_name.append(o)
                           
            else:
                objects_replace_name.append(o)
         
       
        non_obj_tag = 0 if self.add_image_as_a_box else -1
        idb['question'] = self.get_raw(idb['question'],objects_replace_name=objects_replace_name,
                                        non_obj_tag=non_obj_tag)

        idb['answer_choices'] = [self.get_raw(answer,objects_replace_name=objects_replace_name,
                                            non_obj_tag=non_obj_tag)
                                 for answer in idb['answer_choices']]

        idb['rationale_choices'] = [self.get_raw(rationale,objects_replace_name=objects_replace_name,
                                                non_obj_tag=non_obj_tag)
                                    for rationale in idb['rationale_choices']]

        question = idb['question']
        answer = idb['answer_choices']
        rationale = idb['rationale_choices']
        answer_label = idb['answer_label']
        rationale_label = idb['rationale_label']

        image = self._load_image(idb['img_fn'], objects_color_and_box)
        image_dict = self._get_image_dict(idb['img_fn'], objects_color_and_box)
        w0, h0 = image.size
        objects = idb['objects']

        # extract bounding boxes and instance masks in metadata
        boxes = torch.zeros((len(objects), 6))
        masks = torch.zeros((len(objects), *self.mask_size))
        if len(objects) > 0:
            boxes[:, :5] = torch.tensor(idb['boxes'])
            boxes[:, 5] = torch.tensor([self.category_to_idx[obj] for obj in objects])
            for i in range(len(objects)):
                seg_polys = [torch.as_tensor(seg) for seg in idb['segms'][i]]
                masks[i] = generate_instance_mask(seg_polys, idb['boxes'][i], mask_size=self.mask_size,
                                                  dtype=torch.float32, copy=False)
        if self.add_image_as_a_box:
            image_box = torch.as_tensor([[0, 0, w0 - 1, h0 - 1, 1.0, 0]])
            image_mask = torch.ones((1, *self.mask_size))
            boxes = torch.cat((image_box, boxes), dim=0)
            masks = torch.cat((image_mask, masks), dim=0)


        # transform
        im_info = torch.tensor([w0, h0, 1.0, 1.0, index])
        # clamp boxes
        w = im_info[0].item()
        h = im_info[1].item()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h - 1)

        image = self.processor(images=image,  return_tensors="pt")['pixel_values']
        outputs = (image, boxes, masks,
                    question,
                    answer, answer_label,
                    rationale, rationale_label, colored_mask,image_dict,
                    im_info)
        
        return outputs

    def __len__(self):
        return len(self.database)

class VCREvalDataset(Dataset):
    def __init__(self, processor, ann_file='val.jsonl', image_set='vcr1images', data_path='../data/vcr', transform=None, task='Q2AR', test_mode=False, use_imgpath=False,
                 only_use_relevant_dets=False, add_image_as_a_box=False, mask_size=(14, 14),
                 basic_align=False, 
                 seq_len=64,n_samples=1000,random_choice=True,use_boundingbox=True, use_imagedict=True,
                 **kwargs):
        """
        Visual Commonsense Reasoning Dataset
        :param ann_file: annotation jsonl file
        :param image_set: image folder name, e.g., 'vcr1images'
        :param root_path: root path to cache database loaded from annotation file
        :param data_path: path to vcr dataset
        :param transform: transform
        :param task: 'Q2A' means question to answer, 'QA2R' means question and answer to rationale,
                     'Q2AR' means question to answer and rationale
        :param test_mode: test mode means no labels available
        :param cache_mode: cache whole dataset to RAM first, then __getitem__ read them from RAM
        :param ignore_db_cache: ignore previous cached database, reload it from annotation file
        :param tokenizer: default is BertTokenizer from pytorch_pretrained_bert
        :param only_use_relevant_dets: filter out detections not used in query and response
        :param add_image_as_a_box: add whole image as a box
        :param mask_size: size of instance mask of each object
        :param aspect_grouping: whether to group images via their aspect
        :param basic_align: align to tokens retokenized by basic_tokenizer
        :param qa2r_noq: in QA->R, the query contains only the correct answer, without question
        :param qa2r_aug: in QA->R, whether to augment choices to include those with wrong answer in query
        :param kwargs:
        """
        super(VCREvalDataset, self).__init__()

        assert task in ['Q2A', 'QA2R', 'Q2AR'] , 'not support task {}'.format(task)
        
        self.seq_len = seq_len
        self.tot = 0
        categories = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                      'trafficlight', 'firehydrant', 'stopsign', 'parkingmeter', 'bench', 'bird', 'cat', 'dog', 'horse',
                      'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                      'suitcase', 'frisbee', 'skis', 'snowboard', 'sportsball', 'kite', 'baseballbat', 'baseballglove',
                      'skateboard', 'surfboard', 'tennisracket', 'bottle', 'wineglass', 'cup', 'fork', 'knife', 'spoon',
                      'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hotdog', 'pizza', 'donut',
                      'cake', 'chair', 'couch', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tv', 'laptop', 'mouse',
                      'remote', 'keyboard', 'cellphone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                      'clock', 'vase', 'scissors', 'teddybear', 'hairdrier', 'toothbrush']
        self.category_to_idx = {c: i for i, c in enumerate(categories)}
        self.data_path = data_path
        self.ann_file = os.path.join(data_path, ann_file)
        self.image_set = image_set
        self.transform = transform
        self.task = task
        self.test_mode = test_mode
        self.use_imgpath = use_imgpath

        self.add_image_as_a_box = add_image_as_a_box
        self.mask_size = mask_size

        self.database = self.load_annotations(self.ann_file)
        self.person_name_id = 0
        self.random_choice = random_choice
        self.processor = processor
        self.use_boundingbox = use_boundingbox
        self.use_imagedict = use_imagedict
    def load_annotations(self, ann_file):
        database = []
        with jsonlines.open(ann_file) as reader:
            for ann in reader:
                img_fn = os.path.join(self.data_path, self.image_set, ann['img_fn'])
                metadata_fn = os.path.join(self.data_path, self.image_set, ann['metadata_fn'])

                db_i = {
                    'annot_id': ann['annot_id'],
                    'objects': ann['objects'],
                    'img_fn': img_fn,
                    'metadata_fn': metadata_fn,
                    'question': ann['question'],
                    'answer_choices': ann['answer_choices'],
                    'answer_label': ann['answer_label'] if not self.test_mode else None,
                    'rationale_choices': ann['rationale_choices'],
                    'rationale_label': ann['rationale_label'] if not self.test_mode else None,
                }
                database.append(db_i)
        return database

    def get_raw(self, tokens, objects_replace_name, non_obj_tag=-1):
        raw = []
        for mixed_token in tokens:
            if isinstance(mixed_token, list):
                tokens = [objects_replace_name[o] for o in mixed_token]
                raw.append(tokens[0])
                for token in tokens[1:]:
                    raw.extend(['and', token])
            else:
                raw.append(mixed_token)
        raw_string = ""
        for item in raw:
            if item == '?' or item == '.' or item == "'" or item == '"' or item == ',':
                raw_string += item
            else:
                raw_string += " "
                raw_string += item
        return raw_string

    def _load_image(self, path, objects_color_and_box):
        if self.use_boundingbox:
            img = Image.open(path).convert('RGB')
            img1 = ImageDraw.Draw(img)
            draw = ImageDraw.Draw(img, 'RGBA') 
            for name,color,idx, box in objects_color_and_box:
                box = box[:4]
                img1.rectangle(box, outline =color)
                draw.rectangle(box, fill=COLOR_TUPLES[idx])
            # img.save('./failurecases/'+str(self.tot)+'.png')
            self.tot += 1
            return img
        else:
            # img = Image.open(path).convert('RGB')
            # img.save('../tests/208.png')
            return Image.open(path).convert('RGB')


    def _get_image_dict(self, path, objects_color_and_box):
        img_dict = {}
        if self.use_imagedict:
            img = Image.open(path).convert('RGB')
            for name,color,box in objects_color_and_box:
                box = box[:4]
                img1 = img.crop(box)
                img1 = self.processor(images=img1, return_tensors='pt')['pixel_values']
                img_dict[name] = img1
            return img_dict
        else:
            return img_dict
    
    def _get_needed_idx(self, question, answers, rationales):
        tot_list = []
        for mixed_token in question:
            if isinstance(mixed_token, list):
                for o in mixed_token:
                    tot_list.append(o)
        for answer in answers:
            for mixed_token in answer:
                if isinstance(mixed_token, list):
                    for o in mixed_token:
                        tot_list.append(o)
        for rationale in rationales:
            for mixed_token in rationale:
                if isinstance(mixed_token, list):
                    for o in mixed_token:
                        tot_list.append(o)
        return tot_list

    def _load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def __getitem__(self, index):
        start = 0
        idb = deepcopy(self.database[index])
        metadata = self._load_json(idb['metadata_fn'])
        idb['boxes'] = metadata['boxes']
        idb['segms'] = metadata['segms']
        objects_replace_name = []
        objects_color_and_box = []
        colored_mask = []
        rationale_discription = ""
        tot_list = self._get_needed_idx(idb["question"], [answer for answer in idb['answer_choices']], [rationale for rationale in idb['rationale_choices']])
        # print(tot_list)
        for num, (o, box) in enumerate(zip(idb['objects'], idb['boxes'])):
            if o == 'person':
                if not self.use_boundingbox:
                    objects_replace_name.append(GENDER_NEUTRAL_NAMES[start])
                    objects_color_and_box.append((GENDER_NEUTRAL_NAMES[start], "none",box))
                    colored_mask.append(0)
                    start = (start + 1) % len(GENDER_NEUTRAL_NAMES)
                else:
                    if num in tot_list:
                        objects_replace_name.append(GENDER_NEUTRAL_NAMES[start]+ " in "+COLOR_MASKS[start]+" region")
                        objects_color_and_box.append((GENDER_NEUTRAL_NAMES[start] + " in "+COLOR_MASKS[start] + " region", COLOR_MASKS[start], start, box))
                        colored_mask.append(0)
                        start = (start + 1) % len(GENDER_NEUTRAL_NAMES)
                    # objects_replace_name.append(GENDER_NEUTRAL_NAMES[start]+" in "+COLOR_MASKS[start]+" region")
                    # objects_color_and_box.append((GENDER_NEUTRAL_NAMES[start]+" in "+COLOR_MASKS[start]+" region", COLOR_MASKS[start], start, box))
                    # colored_mask.append(start)
                    # rationale_discription += GENDER_NEUTRAL_NAMES[start] + " is in the " + COLOR_MASKS[start]+" region. "
                    # start = (start + 1) % len(GENDER_NEUTRAL_NAMES)
                    else:
                        objects_replace_name.append(o)              
            else:
                objects_replace_name.append(o)
          
        non_obj_tag = 0 if self.add_image_as_a_box else -1
        idb['question'] = self.get_raw(idb['question'],objects_replace_name=objects_replace_name,
                                        non_obj_tag=non_obj_tag)

        idb['answer_choices'] = [self.get_raw(answer,objects_replace_name=objects_replace_name,
                                            non_obj_tag=non_obj_tag)
                                 for answer in idb['answer_choices']]

        idb['rationale_choices'] = [self.get_raw(rationale,objects_replace_name=objects_replace_name,
                                                non_obj_tag=non_obj_tag)
                                    for rationale in idb['rationale_choices']]

        question = idb['question']
        answer = idb['answer_choices']
        rationale = idb['rationale_choices']
        answer_label = idb['answer_label']
        rationale_label = idb['rationale_label']

        image = self._load_image(idb['img_fn'],  objects_color_and_box)
        # image.save('../tests/'+str(index)+'.png')
        image_dict = self._get_image_dict(idb['img_fn'], objects_color_and_box)
        w0, h0 = image.size
        objects = idb['objects']

        # extract bounding boxes and instance masks in metadata
        boxes = torch.zeros((len(objects), 6))
        masks = torch.zeros((len(objects), *self.mask_size))
        if len(objects) > 0:
            boxes[:, :5] = torch.tensor(idb['boxes'])
            boxes[:, 5] = torch.tensor([self.category_to_idx[obj] for obj in objects])
            for i in range(len(objects)):
                seg_polys = [torch.as_tensor(seg) for seg in idb['segms'][i]]
                masks[i] = generate_instance_mask(seg_polys, idb['boxes'][i], mask_size=self.mask_size,
                                                  dtype=torch.float32, copy=False)
        if self.add_image_as_a_box:
            image_box = torch.as_tensor([[0, 0, w0 - 1, h0 - 1, 1.0, 0]])
            image_mask = torch.ones((1, *self.mask_size))
            boxes = torch.cat((image_box, boxes), dim=0)
            masks = torch.cat((image_mask, masks), dim=0)


        # transform
        im_info = torch.tensor([w0, h0, 1.0, 1.0, index])
        # clamp boxes
        w = im_info[0].item()
        h = im_info[1].item()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h - 1)
        if self.processor != None:
            image_orig = image
            image = self.processor(images=image,  return_tensors="pt")['pixel_values']
        if self.use_imgpath:
            outputs = (image, boxes, masks,
                    question,
                    answer, answer_label,
                    rationale, rationale_label, idb['img_fn'], image_orig
                    )
            return outputs 
        # print(question)
        # print(answer[answer_label])
        # print(rationale[rationale_label])
        outputs = (image, boxes, masks,
                    question,
                    answer, answer_label,
                    rationale, rationale_label, colored_mask,image_dict, rationale_discription, 
                    im_info)

        return outputs

    def __len__(self):
        return len(self.database)



if __name__=='__main__':
    VERSION = "Salesforce/blip2-flan-t5-xxl"
    processor = Blip2Processor.from_pretrained(VERSION)
    train_dataset = VCREvalDataset(processor=processor,use_boundingbox=True, use_imagedict=False)

    