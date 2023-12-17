import json
from Dataloader.dataloader import *
import codecs
def esnli_filter():
   # Enter result file here
    with open('./results/...', 'r') as f:
        data_before1 = json.load(f)
   
    eval_dataset = eSNLIEvalDataset(processor=None, data_path="data")
    indexes_list = list(data_before1.keys())
    data_new = []
    num = 0
    
    for i in indexes_list:
        element = eval_dataset[int(i)]
        image_name = int(i)
        new_dict = {}
        if 'Yes because ' not in data_before1[str(i)] and 'No because ' not in data_before1[str(i)] and "It's impossible to say because " not in data_before1[str(i)]:
            num += 1
            continue
        new_dict["image_id"] = image_name
        if 'Yes because ' in data_before1[str(i)]:
            new_dict['caption'] = bytes(data_before1[str(i)].split('Yes because ')[1], 'utf-8').decode('utf8')
        elif 'No because ' in data_before1[str(i)]:
            new_dict['caption'] = bytes(data_before1[str(i)].split('No because ')[1], 'utf-8').decode('utf8')
        else:
            new_dict['caption'] = bytes(data_before1[str(i)].split("It's impossible to say because ")[1], 'utf-8').decode('utf8')
        prompt1 = data_before1[str(i)].split(' because ')[0]
       
        if data_before1[str(i)].split(' because ')[0] in [element[2]]:
            new_dict['caption'] = bytes(data_before1[str(i)].split(prompt1+' because ')[1], 'utf-8').decode('utf8') 
            data_new.append(new_dict)
        
    with open('./filtered_json/blipv23%', 'w') as f:
        json.dump(data_new, f)

# ReViSE here
def esnli_process():
    # first step result
    with open('./results/step1', 'r') as f:
        data_before1 = json.load(f)
    # second step result
    with open('./results/step2', 'r') as f:
        data_before2 = json.load(f)
    # third step result
    with open('./results/step3', 'r') as f:
        data_before3 = json.load(f)
    # fourth
    with open('./results/step4', 'r') as f:
        data_before4 = json.load(f)
    # fifth
    with open('./results/step5', 'r') as f:
        data_before5 = json.load(f)
    eval_dataset = eSNLIEvalDataset(processor=None, data_path="data")
    indexes_list = list(data_before5.keys())
    data_new = []
    for i in indexes_list:
        element = eval_dataset[int(i)]
        image_name = int(i)
        new_dict = {}
        if 'Yes because ' not in data_before1[str(i)] and 'No because ' not in data_before1[str(i)] and "It's impossible to say because " not in data_before1[str(i)]:
            continue
        new_dict["image_id"] = image_name
        if 'Yes because ' in data_before1[str(i)]:
            new_dict['caption'] = bytes(data_before1[str(i)].split('Yes because ')[1], 'utf-8').decode('utf8')
        elif 'No because ' in data_before1[str(i)]:
            new_dict['caption'] = bytes(data_before1[str(i)].split('No because ')[1], 'utf-8').decode('utf8')
        else:
            new_dict['caption'] = bytes(data_before1[str(i)].split("It's impossible to say because ")[1], 'utf-8').decode('utf8')
        prompt1 = data_before1[str(i)].split(' because ')[0]
        prompt2 = data_before2[str(i)].split(' because ')[0] 
        prompt3 = data_before3[str(i)].split(' because ')[0] 
        prompt4 = data_before4[str(i)].split(' because ')[0]
        prompt5 = data_before5[str(i)].split(' because ')[0] 
        new_dict['caption'] = bytes(data_before2[str(i)].split(prompt2+' because ')[1], 'utf-8').decode('utf8') 
        if data_before1[str(i)].split(' because ')[0] not in [element[2]]:
            if prompt2 == prompt3:
                new_dict['caption'] = bytes(data_before2[str(i)].split(prompt2+' because ')[1], 'utf-8').decode('utf8')
            elif prompt3 == prompt4:
                new_dict['caption'] = bytes(data_before3[str(i)].split(prompt3+' because ')[1], 'utf-8').decode('utf8')
            elif prompt4 == prompt5:
                new_dict['caption'] = bytes(data_before4[str(i)].split(prompt4+' because ')[1], 'utf-8').decode('utf8')
            else:
                new_dict['caption'] = bytes(data_before5[str(i)].split(prompt5+' because ')[1], 'utf-8').decode('utf8') 
            data_new.append(new_dict)
    # Save result here:
    with open('./filtered_json/...', 'w') as f:
        json.dump(data_new, f)
    

if __name__=='__main__':
    esnli_process()
