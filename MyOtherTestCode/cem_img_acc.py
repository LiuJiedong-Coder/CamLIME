# This default renderer is used for sphinx docs only. Please delete this cell in IPython.
import plotly.io as pio
pio.renderers.default = "png"

import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import PIL
from omnixai.data.image import Image
import os, json
import numpy as np
from omnixai.explainers.vision import ContrastiveExplainer
import time

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

idx2label, cls2label, cls2idx = [], {}, {}
with open(os.path.abspath('imagenet/imagenet_class_index.json'), 'r') as read_file:
    class_idx = json.load(read_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
    cls2idx = {class_idx[str(k)][0]: k for k in range(len(class_idx))}

trans_A = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

trans_B = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

trans_C = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224)
])

def load_images_from_folder(folder_path):
    images = []
    paths = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            try:
                img = PIL.Image.open(img_path)
                images.append(img)
                paths.append(img_path)
            except:
                print("Unable to load image:", img_path)
    return paths, images


def write_to_file(cor_black, cor_explain, black_explain_same, cor_black_explain_same, sample_num, preds_sample, class_name):
    # 打开文件以写入模式，如果文件不存在则会创建
    with open('results/cem_result.txt', 'a') as file:
        file.write(f'-------------File_Name: {class_name}  label: {cls2idx.get(class_name)}----------------\n')
        file.write(f'preds_sample: \n')
        for i in range(len(preds_sample)):
                file.write(f'{preds_sample[i]}\n')

        file.write(f'sample_num: {sample_num}\n')
        file.write(f'cor_black: {(cor_black / sample_num)*100:.2f}\n')
        file.write(f'cor_explain: {(cor_explain / sample_num)*100:.2f}\n')
        file.write(f'black_explain_same: {(black_explain_same / sample_num)*100:.2f}\n')
        file.write(f'cor_black_explain_same: {(cor_black_explain_same / sample_num)*100:.2f}\n')
        file.write(f'-------------------------------------------' + os.linesep)

def end_write_to_file(cor_black, cor_explain, black_explain_same, cor_black_explain_same, sample_num, time):
    # 打开文件以写入模式，如果文件不存在则会创建
    with open('results/cem_result.txt', 'a') as file:
        file.write(f'--------------------Calculate End---------------------' + os.linesep)
        file.write(f'total_sample_num: {sample_num}\n')
        file.write(f'total_cor_black: {(cor_black / sample_num)*100:.2f}\n')
        file.write(f'total_cor_explain: {(cor_explain / sample_num)*100:.2f}\n')
        file.write(f'total_black_explain_same: {(black_explain_same / sample_num)*100:.2f}\n')
        file.write(f'total_cor_black_explain_same: {(cor_black_explain_same / sample_num)*100:.2f}\n')
        file.write(f'total_time: {time:.2f}  sample_avg_explain_time: {time/sample_num:.2f}\n')
        file.write(f'-------------------------------------------' + os.linesep)

if __name__ == "__main__":

    model = models.resnet50(pretrained=True).eval().to(device)

    # Directory containing the images
    base_dir = 'F:\Databases\ImageNet2012/val'

    preprocess = lambda ims: torch.stack([trans_B(im.to_pil()) for im in ims])

    # Get the list of subdirectories (each representing a class)
    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    n_files = 0
    Time = time.time()
    total_cor_black = 0
    total_cor_explain = 0
    total_black_explain_same = 0
    total_cor_black_explain_same = 0
    total_sample_num = 0

    for class_name in classes:
        label = cls2idx.get(class_name)
        print(f'该file_name为: {class_name} 对应label为: {label}')
        class_dir = os.path.join(base_dir, class_name)

        # Get the list of image files in the class directory
        paths, loaded_images = load_images_from_folder(class_dir)

        cor_black = 0
        cor_explain = 0
        black_explain_same = 0
        cor_black_explain_same = 0
        sample_num = 0
        # Dictionary to store predictions
        preds_sample = []
        i = 0

        for img_path in range(len(loaded_images)):
            predictions = {}
            img_pil = loaded_images[i]
            img_path = paths[i]
            #print(model)

            n_pred = 1  #取前n个预测结果
            input_tensor = trans_A(img_pil).unsqueeze(0).to(device)
            pred_logits = model(input_tensor)
            pred_softmax = F.softmax(pred_logits, dim=1)
            top_n = pred_softmax.topk(n_pred)
            print(f'前{n_pred}个预测：{top_n}')
            black_pre = top_n[1].detach().cpu().numpy().squeeze().item()
            print(f'黑盒 Max predicted labels:{black_pre}')

            if black_pre == label:
                cor_black += 1

            x = np.array(trans_C(img_pil))
            x_test = Image(x)   # 转化为ominixai的数据类型

            explainer = ContrastiveExplainer(
                model=model,
                preprocess_function=preprocess
            )

            explanations = explainer.explain(x_test)
            cem_pre = explanations.explanations[0].get("pp_label")
            print(f'\texp_pp: {cem_pre}')
            predictions[img_path] = label
            predictions['black_pre_label'] = black_pre
            predictions['cem_pre'] = cem_pre
            preds_sample.append(predictions)

            if cem_pre == label:
                cor_explain += 1

            if black_pre == cem_pre:
                black_explain_same += 1
                if black_pre == label:
                    cor_black_explain_same += 1
            sample_num += 1
            i += 1
            if i == 50:  #控制每个files中样本数
                break

        write_to_file(cor_black, cor_explain, black_explain_same, cor_black_explain_same, sample_num, preds_sample, class_name)

        total_sample_num += sample_num
        total_cor_black += cor_black
        total_cor_explain += cor_explain
        total_black_explain_same += black_explain_same
        total_cor_black_explain_same += cor_black_explain_same

        n_files += 1
        if n_files == 2:  # 控制测试的文件夹数
            break

    end_write_to_file(total_cor_black, total_cor_explain, total_black_explain_same, total_cor_black_explain_same, total_sample_num, time.time()-Time)
    print(f'time: {time.time() - Time}')

