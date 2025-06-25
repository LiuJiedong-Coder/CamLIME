#参考：https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch.ipynb
#参考：https://github.com/frgfm/notebooks/blob/main/torch-cam/quicktour.ipynb

import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json

import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise
)
from skimage.segmentation import mark_boundaries

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

idx2label, cls2label, cls2idx = [], {}, {}
with open(os.path.abspath('imagenet/imagenet_class_index.json'), 'r') as read_file:
    class_idx = json.load(read_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
    cls2idx = {class_idx[str(k)][0]: k for k in range(len(class_idx))}


class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='layercam',
                        choices=[
                            'gradcam', 'hirescam', 'gradcam++',
                            'scorecam', 'xgradcam', 'ablationcam',
                            'eigencam', 'eigengradcam', 'layercam',
                            'fullgrad', 'gradcamelementwise'
                        ],
                        help='CAM method')

    parser.add_argument(
        '--eigen-smooth',
        action='store_true',
        help='Reduce noise by taking the first principle component of cam_weights*activations')

    parser.add_argument('--aug-smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    args = parser.parse_args()

    return args

trans_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

trans_A = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    trans_norm
    ])

trans_B = transforms.Compose([
        transforms.ToTensor(),
        trans_norm
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


if __name__ == '__main__':
    args = get_args()
    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "gradcamelementwise": GradCAMElementWise
    }
    #img_path = 'test_img/ILSVRC2012_val_00030181.JPEG'
    img_path = 'test_img/2aac5ffd72.jpg'  #效果好
    #img_path = 'test_img/lion_tiger.png'

    img_pil = Image.open(img_path)

    #model = models.inception_v3(pretrained=True).eval().to(device)
    model = models.resnet18(pretrained=True).eval().to(device)
    #print(model)

    n_pred = 1  #取前n个预测结果
    input_tensor = trans_A(img_pil).unsqueeze(0).to(device)
    pred_logits = model(input_tensor)
    pred_softmax = F.softmax(pred_logits, dim=1)
    top_n = pred_softmax.topk(n_pred)
    print(f'前{n_pred}个预测：{top_n}')
    pred_id = top_n[1].detach().cpu().numpy().squeeze().item()
    print(f'黑盒 Max predicted labels:{pred_id}')

    targets = [ClassifierOutputTarget(pred_id)]
    cam_algorithm = methods[args.method]

    layers = {
        'S2': [model.layer1, model.layer2],
        'S3': [model.layer1, model.layer2, model.layer3],
        'S4': [model.layer1, model.layer2, model.layer3, model.layer4]
    }

    show_img = []
    for i, layer_list in enumerate(layers):
        with cam_algorithm(model=model, target_layers=layers[layer_list], use_cuda=True) as cam:
            # print(cam.activations_and_grads.activations, cam.activations_and_grads.gradients) #(1, 512, 7, 7)
            # activations = cam.activations_and_grads.activations   #(1, 512, 7, 7), 特征图
            # gradients = cam.activations_and_grads.gradients       #(1, 512, 7, 7), 梯度图

            grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=args.aug_smooth, eigen_smooth=args.eigen_smooth)  # 激活图(1,224,224)
            layer_cam = cam.ever_layer_cam  #layerCAM的各层cam激活图
            fues_cam = grayscale_cam[0, :]  #多层融合后的cam激活图，ndarray(224,224)

            layer_fues_cams = []
            for j in range(len(layer_cam)):
                layer_fues_cams.append(layer_cam[j].squeeze())
            layer_fues_cams.append(fues_cam)

        random_seed = 500  #456  #789
        num_features = 20
        ####################OUrsLIME#####################################
        from lime import lime_image_my
        explainer = lime_image_my.LimeImageExplainer()  #个人修改

        # batch_predict分类预测函数，num_samples是LIME生成的邻域图像个数
        data, labels = explainer.explain_instance_data_label(np.array(trans_C(img_pil)), batch_predict, top_labels=1, hide_color=0, num_samples=8000, random_seed=random_seed)

        figs_our = []
        for j, act_cam in enumerate(layer_fues_cams):
            cam_img = {}
            explanation = explainer.explain_instance(act_cam)
            print(f'解释器预测分类{explanation.top_labels}')
            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=num_features, hide_rest=False)
            img_boundry_our = mark_boundaries(temp / 255.0, mask)
            cam_img[f'layer{j}'] = layer_fues_cams[j]
            cam_img[f'img_our{j}'] = img_boundry_our
            figs_our.append(cam_img)

        show_img.append(figs_our)

    rows = len(show_img)
    cols = len(show_img[len(show_img)-1])*2 + 1

    fig, axes = plt.subplots(rows, cols,  figsize=(cols*1.5, rows*1.5))

    for i in range(len(show_img)):
        axes[i, 0].imshow(img_pil)

        # 遍历每一对图像
        for j, figs in enumerate(show_img[i]):
            axes[i, 2*j+1].imshow(figs[f'layer{j}'])
            axes[i, 2*j+2].imshow(figs[f'img_our{j}'])

    # 关闭画布中所有子图的坐标轴
    for i in range(rows):
        for j in range(cols):
            axes[i, j].axis('off')  # 关闭坐标轴

    plt.tight_layout()  # 调整子图排版
    plt.show()  # 显示绘制完成的图像

