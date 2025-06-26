import argparse
import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from torchvision.transforms import Compose, Normalize, ToTensor
from pytorch_grad_cam.base_cam import BaseCAM

from lime import lime_image
import torch.nn.functional as F
import matplotlib.pyplot as plt


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--aug-smooth', action='store_true',

                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen-smooth',
        action='store_true',
        help='Reduce noise by taking the first principle component'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=[
                            'gradcam', 'hirescam', 'gradcam++',
                            'scorecam', 'xgradcam', 'ablationcam',
                            'eigencam', 'eigengradcam', 'layercam',
                            'fullgrad', 'gradcamelementwise'
                        ],
                        help='CAM method')

    args = parser.parse_args()
    return args




# 归一化图像，并转成tensor
def preprocess_image(img: np.ndarray, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> torch.Tensor:
    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy()).unsqueeze(0)

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

def show_neighborhood_img(img, num_cols, save_img=False):
    n_cols = num_cols
    n_rows = (len(img) // n_cols) + 1
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 12))
    axes = axes.flatten()   #将网格展平为1D数组以进行迭代

    for i in range(n_rows*n_cols):
            if i < len(img):
                axes[i].imshow(img[i])
                axes[i].axis('off')
            else:
                fig.delaxes(axes[i])

    plt.tight_layout()  # 调整子图排版
    if save_img:
        plt.savefig(f'output/neighborImg/neighborhood{len(img)}.png')

    plt.show()  # 显示绘制完成的图像



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

    #img_path = 'test_img/cat_dog.jpg'
    img_path = 'test_img/both.png'    #eigengradcam，fullgrad效果改进好
    #img_path = 'test_img/lion_tiger.png'
    #img_path = 'test_img/bird2.png'
    #img_path = 'test_img/ILSVRC2012_val_00011333.JPEG'

    img_pil = Image.open(img_path)

    model = models.resnet18(pretrained=True).eval().to(device)
    #print(model)

    target_layers = [model.layer4]
    #print(target_layers)

    input_tensor = trans_A(img_pil).unsqueeze(0).to(device)
    model_pred = model(input_tensor)
    pred_softmax = F.softmax(model_pred, dim=1)
    top_n = pred_softmax.topk(1)
    pred_id = top_n[1].detach().cpu().numpy().squeeze().item()
    print(f'黑盒 Max predicted labels:{pred_id}')

    targets = [ClassifierOutputTarget(pred_id)]
    cam_algorithm = methods[args.method]

    with cam_algorithm(model=model, target_layers=target_layers, use_cuda=True) as cam:
        #print(cam.activations_and_grads.activations, cam.activations_and_grads.gradients) #(1, 512, 7, 7)
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=args.aug_smooth, eigen_smooth=args.eigen_smooth)  # 激活图(1,224,224)
        grayscale_cam = grayscale_cam[0, :]  #激活图，ndarray(224,224)

        activations = cam.activations_and_grads.activations   #(1, 512, 7, 7), 特征图
        gradients = cam.activations_and_grads.gradients       #(1, 512, 7, 7), 梯度图



    from lime import lime_image
    from lime import lime_image_my
    #explainer = lime_image.LimeImageExplainer()  #源码
    explainer = lime_image_my.LimeImageExplainer()  #个人修改

    #explanation = explainer.explain_instance(rgb_img_org, batch_predict, top_labels=5, hide_color=0, num_samples=8000)  # batch_predict分类预测函数，num_samples是LIME生成的邻域图像个数
    data, labels = explainer.explain_instance_data_label(np.array(trans_C(img_pil)), batch_predict, top_labels=1, hide_color=0, num_samples=50)  # batch_predict分类预测函数，num_samples是LIME生成的邻域图像个数
    # interpolation_cam = bilinear_interpolation(grayscale_cam, data.shape)
    # print(interpolation_cam.shape)

    #经历过超像素分割后的图像
    segments = explainer.segments   #(224,224)ndarray

    #显示邻域样本，同时使用，否则在领域样本太多的时候占用内存
    # nb_samples = explainer.my_nb_samples
    # show_neighborhood_img(nb_samples, num_cols=3, save_img=True)

    explanation = explainer.explain_instance(grayscale_cam)
    print(f'解释器预测分类{explanation.top_labels}')
    print(f'线性模型特征-权重：{explanation.local_exp}')

    n_features = 20
    seg_rank_end = [t[0] for t in explanation.local_exp[explanation.top_labels[0]]]
    seg_rank_topn = seg_rank_end[:n_features]

    seg = explanation.segments
    # Create a mask which is True when segment value equals
    mask = np.isin(explanation.segments, seg_rank_topn)
    masked_image = np.array(trans_C(img_pil)).copy()
    masked_image[~mask] = [0, 0, 0]

    plt.figure(figsize=(4, 2))

    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(masked_image)
    #plt.show()


    from skimage.segmentation import mark_boundaries
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=n_features, hide_rest=True)
    img_boundry = mark_boundaries(temp / 255.0, mask)
    plt.subplot(1, 2, 2)
    plt.title('Boundary')
    plt.imshow(img_boundry)
    plt.show()


