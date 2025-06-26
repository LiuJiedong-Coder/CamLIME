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
from omnixai.explainers.vision import IntegratedGradientImage
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


def batch_predict(images):
    batch = torch.stack(tuple(trans_B(i) for i in images), dim=0)
    batch = batch.to(device)

    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

if __name__ == "__main__":

    model = models.resnet50(pretrained=True).eval().to(device)

    img_path = 'test_img/test.JPEG'
    img_pil = PIL.Image.open(img_path)

    n_pred = 1  #取前n个预测结果
    input_tensor = trans_A(img_pil).unsqueeze(0).to(device)
    pred_logits = model(input_tensor)
    pred_softmax = F.softmax(pred_logits, dim=1)
    top_n = pred_softmax.topk(n_pred)
    print(f'前{n_pred}个预测：{top_n}')
    pred_id = top_n[1].detach().cpu().numpy().squeeze().item()
    print(f'黑盒 Max predicted labels:{pred_id}')

    preprocess = lambda ims: torch.stack([trans_B(im.to_pil()) for im in ims])

    x = np.array(trans_C(img_pil))
    x_test = Image(x)   # 转化为ominixai的数据类型

    # Initialize a explainer
    explainer = IntegratedGradientImage(
        model=model,
        preprocess_function=preprocess
    )

    Time = time.time()

    explanations = explainer.explain(x_test)
    #print(f'\texp_pp: {explanations.explanations[0].get("pp_label")}')
    print(f'Explain time: {time.time() - Time}')
    #explanations.plot(index=0, class_names=idx2label)
    explanations.plot()
    plt.show()

