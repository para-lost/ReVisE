import json
from Dataloader.dataloader import *
import codecs
def vqa_process():
    with open('path_to_step1', 'r') as f:
        data_before1 = json.load(f)
    with open('path_to_step2', 'r') as f:
        data_before2 = json.load(f)
    with open('path_to_step3', 'r') as f:
        data_before3 = json.load(f)
    with open('path_to_step4', 'r') as f:
        data_before4 = json.load(f)
    with open('path_to_step5', 'r') as f:
        data_before5 = json.load(f)
    with open('./data/vqaX_test.json', 'r') as f:
        data_image = json.load(f)
    list_index = []
    for k in data_image.keys():
        list_index.append(k)
    eval_dataset = VQAXEvalDataset(processor=None, data_path="data")
    indexes_list = list(range(0, len(eval_dataset)))
    data_new = []
    num = 0
    for i in indexes_list:
        element = eval_dataset[i]
        image_name = int(list_index[num])
        new_dict = {}
        if ' because ' not in data_before1[str(i)]:
            num += 1
            continue
        new_dict["image_id"] = image_name
        new_dict['caption'] = bytes(data_before1[str(i)].split(' because ')[1], 'utf-8').decode('utf8')
        num += 1
        if data_before1[str(i)].split(' because ')[0] not in [element[4]]:
            if data_before1[str(i)].split(' because ')[0] == data_before2[str(i)].split(' because ')[0]:
                new_dict['caption'] = bytes(data_before1[str(i)].split(' because ')[1], 'utf-8').decode('utf8')
            elif data_before2[str(i)].split(' because ')[0] == data_before3[str(i)].split(' because ')[0]:
                new_dict['caption'] = bytes(data_before2[str(i)].split(' because ')[1], 'utf-8').decode('utf8')
            elif data_before3[str(i)].split(' because ')[0] == data_before4[str(i)].split(' because ')[0]:
                new_dict['caption'] = bytes(data_before3[str(i)].split(' because ')[1], 'utf-8').decode('utf8')
            elif data_before4[str(i)].split(' because ')[0] == data_before5[str(i)].split(' because ')[0]:
                new_dict['caption'] = bytes(data_before4[str(i)].split(' because ')[1], 'utf-8').decode('utf8')
            else:
                new_dict['caption'] = bytes(data_before5[str(i)].split(' because ')[1], 'utf-8').decode('utf8')
            data_new.append(new_dict)
    with open('save_path_here', 'w') as f:
        json.dump(data_new, f)

if __name__=='__main__':
    vqa_process()

