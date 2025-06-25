#参考：https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch.ipynb

import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json

import torch
from torchvision import models, transforms
from torch.autograd import Variable
from skimage.segmentation import mark_boundaries
import torch.nn.functional as F
import argparse
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise
)
import warnings
warnings.filterwarnings("ignore")

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
    # parser.add_argument('--method', type=str, default='hirescam',
    #                     choices=[
    #                         'gradcam', 'hirescam', 'gradcam++',
    #                         'scorecam', 'xgradcam', 'ablationcam',
    #                         'eigencam', 'eigengradcam', 'layercam',
    #                         'fullgrad', 'gradcamelementwise'
    #                     ],
    #                     help='CAM method')

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
        ##"scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        ##"eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        ##"fullgrad": FullGrad,  #论文中未进行该实验
        ##"gradcamelementwise": GradCAMElementWise
    }
    #img_path = 'test_img/cat_dog.jpg'
    #img_path = 'test_img/both.png'
    img_path = 'test_img/ILSVRC2012_val_00030181.JPEG'

    img_pil = Image.open(img_path)

    # 模型根据参数量由小到大
    #model = models.mobilenet_v2(pretrained=True).eval().to(device)
    #model = models.alexnet(pretrained=True).eval().to(device)
    #model= models.googlenet(pretrained=True).eval().to(device)
    model= models.resnet18(pretrained=True).eval().to(device)
    #model = models.inception_v3(pretrained=True).eval().to(device)
    #model= models.densenet121(pretrained=True).eval().to(device)
    ##model = models.vgg16(pretrained=True).eval().to(device)


    #target_layers = [model.features[17]]   #mobilenet
    # target_layers = [model.features[12]]   #alexnet
    # target_layers = [model.inception5b]   #googlenet
    target_layers = [model.layer4[-1]]   #resnet18,50
    # target_layers = [model.Mixed_7c]   #inception_v3
    #target_layers = [model.features[-1]]  # vgg / densenet
    ##target_layers = [model.features[-1]]


    n_pred = 1  #取前n个预测结果
    title_figs = {}
    input_tensor = trans_A(img_pil).unsqueeze(0).to(device)
    pred_logits = model(input_tensor)
    pred_softmax = F.softmax(pred_logits, dim=1)
    top_n = pred_softmax.topk(n_pred)
    print(f'前{n_pred}个预测：{top_n}')
    pred_id = top_n[1].detach().cpu().numpy().squeeze().item()
    print(f'黑盒 Max predicted labels:{pred_id}')

    targets = [ClassifierOutputTarget(pred_id)]
    #cam_algorithm = methods[args.method]


    random_seed = 100
    ################原始LIME######################################
    from lime import lime_image

    explainer_org = lime_image.LimeImageExplainer()
    explanation_org = explainer_org.explain_instance(np.array(trans_C(img_pil)), batch_predict, top_labels=1, hide_color=0, num_samples=8000, random_seed=random_seed)
    # explanation_org = explainer_org.explain_instance(np.array(trans_C(Image.fromarray(grayscale_cam))), batch_predict, top_labels=1, hide_color=0, num_samples=8000, random_seed=random_seed)
    # explanation_org = explainer_org.explain_instance(np.array(trans_C(fig_cam)), batch_predict, top_labels=1, hide_color=0, num_samples=8000, random_seed=random_seed)

    num_features = 20
    temp, mask = explanation_org.get_image_and_mask(explanation_org.top_labels[0], positive_only=False, num_features=num_features, hide_rest=False)
    img_boundry_org = mark_boundaries(temp / 255.0, mask)

    class_act_mode = []
    class_act_mapping = []
    img_boundry_act = []
    for cam_algorithm in methods.values():
        with cam_algorithm(model=model, target_layers=target_layers, use_cuda=True) as cam:
            # print(cam.activations_and_grads.activations, cam.activations_and_grads.gradients) #(1, 512, 7, 7)
            # activations = cam.activations_and_grads.activations   #(1, 512, 7, 7), 特征图
            # gradients = cam.activations_and_grads.gradients       #(1, 512, 7, 7), 梯度图

            grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=args.aug_smooth, eigen_smooth=args.eigen_smooth)  # 激活图(1,7,7)
            grayscale_cam = grayscale_cam[0, :]  #cam激活图，ndarray(7,7)
            #grayscale_cam = grayscale_cam[0]  # cam激活图，ndarray(7,7)
            from torchcam.utils import overlay_mask
            fig_cam = overlay_mask(img_pil, Image.fromarray(grayscale_cam), alpha=0.4)  #alpha越小，原图越淡

        ################OUrsLIME#####################################
        from lime import lime_image_my
        explainer = lime_image_my.LimeImageExplainer()  #个人修改

        # batch_predict分类预测函数，num_samples是LIME生成的邻域图像个数
        data, labels = explainer.explain_instance_data_label(np.array(trans_C(img_pil)), batch_predict, top_labels=1, hide_color=0, num_samples=8000, random_seed=random_seed)

        explanation = explainer.explain_instance(grayscale_cam)
        print(f'解释器预测分类{explanation.top_labels}')
        #print(class_idx.get(str(explanation.top_labels[0]))[1])

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=num_features, hide_rest=False)
        img_boundry_our = mark_boundaries(temp / 255.0, mask)

        class_act_mode.append(grayscale_cam)
        class_act_mapping.append(fig_cam)
        img_boundry_act.append(img_boundry_our)

    title_figs['Original'] = img_pil
    title_figs['LIME'] = img_boundry_org
    title_figs['Activation'] = class_act_mode
    title_figs['Mapping'] = class_act_mapping
    title_figs['Ours'] = img_boundry_act

    fig, axes = plt.subplots(len(methods), 5, figsize=(1.5*5, 1.5*len(methods)))  # 创建 n 行 5 列的子图
    axes = axes.flatten()  # 将子图对象展平成一维数组，方便索引

    # 遍历每行
    for row in range(len(methods)):
        # 前两列：Original 和 LIME
        axes[row * 5 + 0].imshow(title_figs['Original'])  # 第一列
        axes[row * 5 + 0].set_title(" ", fontsize=8)
        axes[row * 5 + 1].imshow(title_figs['LIME'])  # 第二列
        axes[row * 5 + 1].set_title(" ", fontsize=8)

        # 第三列：Activation
        #axes[row * 5 + 2].imshow(title_figs['Activation'][row], cmap='seismic')  # 第三列
        axes[row * 5 + 2].imshow(title_figs['Activation'][row])  # 第三列
        axes[row * 5 + 2].set_title(" ", fontsize=8)

        # 第四列：Mapping
        axes[row * 5 + 3].imshow(title_figs['Mapping'][row], cmap='seismic')  # 第四列
        axes[row * 5 + 3].set_title(" ", fontsize=8)

        # 第五列：Ours
        axes[row * 5 + 4].imshow(title_figs['Ours'][row], cmap='seismic')  # 第五列
        axes[row * 5 + 4].set_title(" ", fontsize=8)

    # 去掉所有子图的坐标轴
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()  # 调整子图间距
    #plt.subplots_adjust(hspace=1.0)  # 顶部和底部边界位置，分别占画布高度的 90% 和 10%
    plt.show()



