import argparse
import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from skimage.segmentation import mark_boundaries
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
    # parser.add_argument('--method', type=str, default='gradcam',
    #                     choices=[
    #                         'gradcam', 'hirescam', 'gradcam++',
    #                         'scorecam', 'xgradcam', 'ablationcam',
    #                         'eigencam', 'eigengradcam', 'layercam',
    #                         'fullgrad', 'gradcamelementwise'
    #                     ],
    #                     help='CAM method')

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
    #img_path = 'test_img/both.png'    #eigengradcam，fullgrad效果改进好
    #img_path = 'test_img/lion_tiger.png'
    #img_path = 'test_img/bird2.png'
    #img_path = 'test_img/ILSVRC2012_val_00003014.JPEG'
    img_path = 'test_img/ILSVRC2012_val_00002653.JPEG'

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

    random_seed = 100  #保持LIME和OurLIME的随机数种子一致，分割才能一致
    n_features = 20  # 想要可视化的特征数

    # ####################原始LIME######################################
    from lime import lime_image
    explainer_org = lime_image.LimeImageExplainer()
    explanation_org = explainer_org.explain_instance(np.array(trans_C(img_pil)), batch_predict, top_labels=1, hide_color=0, num_samples=8000, random_seed=random_seed)

    temp, mask = explanation_org.get_image_and_mask(explanation_org.top_labels[0], positive_only=False, num_features=n_features, hide_rest=False)
    img_boundry_org = mark_boundaries(temp / 255.0, mask)

    # 权重值
    print(f'exp_org_w: {explanation_org.local_exp.values()}')
    exp_org_w = list(explanation_org.local_exp.values())[0]

    # 经历过超像素分割后的特征块
    seg_org = explanation_org.segments

    # 显示全部特征
    input_image = np.array(trans_C(img_pil)).copy()
    masked_image_org = mark_boundaries(input_image, seg_org)

    show_figs = []
    for cam_algorithm in methods.values():
        title_figs = {}
        ###################OurLIME#######################################
        with cam_algorithm(model=model, target_layers=target_layers, use_cuda=True) as cam:
            #print(cam.activations_and_grads.activations, cam.activations_and_grads.gradients) #(1, 512, 7, 7)
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=args.aug_smooth, eigen_smooth=args.eigen_smooth)  # 激活图(1,224,224)
            grayscale_cam = grayscale_cam[0, :]  #激活图，ndarray(224,224)

            activations = cam.activations_and_grads.activations   #(1, 512, 7, 7), 特征图
            gradients = cam.activations_and_grads.gradients       #(1, 512, 7, 7), 梯度图

        from lime import lime_image_my
        #explainer = lime_image.LimeImageExplainer()  #源码
        explainer = lime_image_my.LimeImageExplainer()  #个人修改
        data, labels = explainer.explain_instance_data_label(np.array(trans_C(img_pil)), batch_predict, top_labels=1, hide_color=0, num_samples=8000, random_seed=random_seed)  # batch_predict分类预测函数，num_samples是LIME生成的邻域图像个数
        explanation = explainer.explain_instance(grayscale_cam)
        print(f'解释器预测分类{explanation.top_labels}')
        #print(f'线性模型特征-权重：{explanation.local_exp}')

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=n_features, hide_rest=False)
        img_boundry_our = mark_boundaries(temp / 255.0, mask)

        #权重值
        #print(f'exp_org_w: {explanation_org.local_exp.values()}')
        print(f'exp_our_w: {explanation.local_exp.values()}')
        #exp_org_w = list(explanation_org.local_exp.values())[0]
        exp_our_w = list(explanation.local_exp.values())[0]

        #经历过超像素分割后的特征块
        #seg_org = explanation_org.segments
        seg_our = explanation.segments  #(224,224)ndarray

        #显示全部特征
        #input_image = np.array(trans_C(img_pil)).copy()
        #masked_image_org = mark_boundaries(input_image, seg_org)
        masked_image_our = mark_boundaries(input_image, seg_our)

        title_figs['OurSeg'] = seg_our
        title_figs['OurLIME'] = img_boundry_our
        title_figs['OurWeight'] = exp_our_w
        show_figs.append(title_figs)


    # 可视化分割图像并添加编号
    fig, ax = plt.subplots(len(show_figs), 4, figsize=(4*2, len(show_figs)*2))

    for i, fig_dict in enumerate(show_figs):
        #加入第1张图片
        ax[i, 0].imshow(masked_image_org)
        unique_segments = np.unique(seg_org)
        for segment_id in unique_segments:
            # 获取当前分割块的像素坐标
            y_coords, x_coords = np.where(seg_org == segment_id)
            # 计算中心点
            center_x, center_y = x_coords.mean(), y_coords.mean()
            # 在中心点标注分割块编号
            ax[i, 0].text(center_x, center_y, str(segment_id), color='blue', fontsize=4, ha='center', va='center')
        #ax[i, 0].set_title('Segmented', fontsize=6)
        ax[i, 0].axis("off")

        #加入第2张图片
        ax[i, 1].imshow(img_boundry_org)
        # 获取分割块编号的中心位置
        unique_segments = np.unique(seg_org)
        for segment_id in unique_segments:
            # 获取当前分割块的像素坐标
            y_coords, x_coords = np.where(seg_org == segment_id)
            # 计算中心点
            center_x, center_y = x_coords.mean(), y_coords.mean()
            # 在中心点标注分割块编号
            ax[i, 1].text(center_x, center_y, str(segment_id), color='blue', fontsize=4, ha='center', va='center')
        #[i, 1].set_title('LIME', fontsize=6)
        ax[i, 1].axis("off")

        #加入第3张图片
        ax[i, 2].imshow(fig_dict['OurLIME'])
        # 获取分割块编号的中心位置
        unique_segments = np.unique(fig_dict['OurSeg'])
        for segment_id in unique_segments:
            # 获取当前分割块的像素坐标
            y_coords, x_coords = np.where(fig_dict['OurSeg'] == segment_id)
            # 计算中心点
            center_x, center_y = x_coords.mean(), y_coords.mean()
            # 在中心点标注分割块编号
            ax[i, 2].text(center_x, center_y, str(segment_id), color='blue', fontsize=4, ha='center', va='center')
        #ax[i, 2].set_title('Ours', fontsize=6)
        ax[i, 2].axis("off")

        #加入第4张图片
        x1, y1 = zip(*(sorted(exp_org_w, key=lambda x: x[0])))
        x2, y2 = zip(*(sorted(title_figs['OurWeight'], key=lambda x: x[0])))
        ax[i, 3].plot(x1, y1, label='LIME', linewidth=0.8)
        ax[i, 3].plot(x2, y2, label='Ours', linewidth=0.8)
        ax[i, 3].set_xlabel('Feature', fontsize=4)
        ax[i, 3].set_ylabel('Weight', fontsize=4)
        # ax[i, 3].set_ylim(-0.10, 0.5)
        # ax[i, 3].set_yticks(np.linspace(-0.10, 0.5, 15))  # Y轴 15 个刻度点
        #ax[i, 3].set_title('Curves Visualization', fontsize=6)
        ax[i, 3].tick_params(labelsize=4)  #刻度大小
        ax[i, 3].legend(fontsize=4) #图例大小
        #ax[i, 3].set_xlim(left=min(x1), right=max(x1) * 1.2)  # 增加横轴范围，拉宽图片

    plt.tight_layout()  #自动调节间距
    #fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, hspace=0.3, wspace=0.3)
    plt.show()



