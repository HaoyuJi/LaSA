import torch
import clip
import torch.nn as nn
import numpy as np



class TextCLIP(nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model.float()

    def forward(self,text):
        return self.model.encode_text(text)

def text_prompt_for_class(dataset_name, context_type):
    print("Use text prompt openai pasta pool")

    paste_text_map = []

    with open(f'text/actions/{context_type}/{dataset_name}.txt') as infile:
        lines = infile.readlines()
        for ind, line in enumerate(lines):
            paste_text_map.append(line)

    class_text = torch.cat(
        [clip.tokenize((pasta_list)) for pasta_list in paste_text_map])


    return class_text

def text_prompt_for_joint(dataset_name, context_type):
    print("Use text prompt of joints")

    paste_text_map_joint = []

    with open(f'text/joints/{context_type}/{dataset_name}.txt') as infile:
        lines = infile.readlines()
        for ind, line in enumerate(lines):
            paste_text_map_joint.append(line)

    class_text = torch.cat(
        [clip.tokenize((pasta_list)) for pasta_list in paste_text_map_joint])


    return class_text

def text_prompt_for_clip(gt_split, dataset_name, context_type):
    # print("Use text prompt clip openai pasta pool")

    text_aug_cnts = [f"This clip contains only one action.", f"This clip contains two actions.",
                     f"This clip contains three actions.", f"This clip contains four actions.",
                     f"This clip contains five actions.", f"This clip contains six actions.",
                     f"This clip contains seven actions.", f"This clip contains eight actions."]
    text_aug_acts = [f"Firstly, ", f"Secondly, ", f"Thirdly, ", f"Fourthly, ",
                     f"Fifthly, ", f"Sixthly, ", f"Seventhly, ", f"Eighthly, "]
    text_long_temp = [f"the person is ", f"the character is ",
                      f"the human is ", f"the scene is ",
                      f"the step is ", f"the action is "]

    paste_text_map = []

    with open(f'text/actions/{context_type}/{dataset_name}.txt') as infile:
        lines = infile.readlines()
        for ind, line in enumerate(lines):
            paste_text_map.append(line)

    clip_texts = []
    for gt in gt_split:
        clip_text = text_aug_cnts[len(gt)-1]
        for index, label in enumerate(gt):
            clip_text = (clip_text + " " + text_aug_acts[index]
                         + text_long_temp[np.random.randint(0, len(text_long_temp))]
                         + paste_text_map[label] + ".")
        clip_texts.append(clip_text)

    class_text = torch.cat(
        [clip.tokenize((clip_list)) for clip_list in clip_texts])


    return class_text
