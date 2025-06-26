import torch
import torch.nn.functional as F

from captum.attr import visualization as viz
from captum.attr import Lime, LimeBase
from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso

import os
import json

from torchvision.models import resnet18
from torchvision.datasets import VOCSegmentation
import torchvision.transforms as T
from captum.attr._core.lime import get_exp_kernel_similarity_function

from PIL import Image
import matplotlib.pyplot as plt

if __name__ == '__main__':
    resnet = resnet18(pretrained=True)
    resnet = resnet.eval()

    labels_path = 'imagenet/imagenet_class_index.json'

    with open(labels_path) as json_data:
        idx_to_labels = {idx: label for idx, [_, label] in json.load(json_data).items()}

    voc_ds = VOCSegmentation(
        root=r'F:\Databases',
        year='2012',
        image_set='train',
        download=False,
        transform=T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]),
        target_transform=T.Lambda(
            lambda p: torch.tensor(p.getdata()).view(1, p.size[1], p.size[0])
        )
    )

    print('ok')

