#参考：https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch.ipynb

import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json
import argparse
import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
import quantus
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise
)




# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

idx2label, cls2label, cls2idx = [], {}, {}
with open(os.path.abspath('imagenet/imagenet_class_index.json'), 'r') as read_file:
    class_idx = json.load(read_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
    cls2idx = {class_idx[str(k)][0]: k for k in range(len(class_idx))}


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


class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='fullgrad',
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

def batch_predict(images):
    batch = torch.stack(tuple(trans_B(i) for i in images), dim=0)
    batch = batch.to(device)

    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

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
    "gradcamelementwise": GradCAMElementWise,
}

if __name__ == '__main__':
    args = get_args()
    #img_path = 'test_img/cat_dog.jpg'
    #img_path = 'examples/both.png'
    img_path = 'test_img/bird2.png'



    img_pil = Image.open(img_path)

    #model = models.inception_v3(pretrained=True).eval().to(device)
    model = models.resnet18(pretrained=True).eval().to(device)
    #print(model)


    input_tensor = trans_A(img_pil).unsqueeze(0).to(device)
    pred_logits = model(input_tensor)
    pred_softmax = F.softmax(pred_logits, dim=1)
    top_n = pred_softmax.topk(1)
    #print(f'Predicted:{top_n}')

    pred_id = top_n[1].detach().cpu().numpy().squeeze().item()
    print(f'黑盒网络预测分类：{pred_id}')
    black_pred_id = np.expand_dims(np.array(pred_id), 0)  #进行该处理是为了满足评价准指标的计算输入维度， ndarray(1,)

    targets = [ClassifierOutputTarget(pred_id)]
    cam_algorithm = methods[args.method]
    target_layers = [model.layer4]

    with cam_algorithm(model=model, target_layers=target_layers, use_cuda=True) as cam:
        # print(cam.activations_and_grads.activations, cam.activations_and_grads.gradients) #(1, 512, 7, 7)
        # activations = cam.activations_and_grads.activations   #(1, 512, 7, 7), 特征图
        # gradients = cam.activations_and_grads.gradients       #(1, 512, 7, 7), 梯度图

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)  # 激活图(1,7,7)
        grayscale_cam = grayscale_cam[0, :]  # cam激活图，ndarray(7,7)
        # grayscale_cam = grayscale_cam[0]  # cam激活图，ndarray(7,7)
        from torchcam.utils import overlay_mask
        fig_cam = overlay_mask(img_pil, Image.fromarray(grayscale_cam), alpha=0.4)  # alpha越小，原图越淡

    random_seed = 500
    num_features = 20
    from skimage.segmentation import mark_boundaries
####################原始LIME######################################
    from lime import lime_image
    explainer_org = lime_image.LimeImageExplainer()
    explanation_org = explainer_org.explain_instance(np.array(trans_C(img_pil)), batch_predict, top_labels=1, hide_color=0, num_samples=8000, random_seed=random_seed)

    temp, mask = explanation_org.get_image_and_mask(explanation_org.top_labels[0], positive_only=False, num_features=num_features, hide_rest=False)
    img_boundry_org = mark_boundaries(temp/255.0, mask)  #ndarray(224, 224, 3), lime结果图，可以通过plt显示

    attributions_org = img_boundry_org.sum(axis=2)  # 计算每个像素的通道值之和，作为归因贡献图 ndarray(224, 224)
    attr_img_org = attributions_org.reshape(1, 1, 224, 224)   #满足评价准指标的计算输入维度

####################OUrsLIME#####################################
    from lime import lime_image_my

    explainer = lime_image_my.LimeImageExplainer()  # 个人修改
    # batch_predict分类预测函数，num_samples是LIME生成的邻域图像个数
    data, labels = explainer.explain_instance_data_label(np.array(trans_C(img_pil)), batch_predict, top_labels=1,hide_color=0, num_samples=8000, random_seed=random_seed)
    explanation = explainer.explain_instance(grayscale_cam)
    print(f'解释器预测分类{explanation.top_labels}')

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=num_features, hide_rest=False)
    img_boundry_our = mark_boundaries(temp / 255.0, mask)

    attributions_our = img_boundry_our.sum(axis=2)  # 计算每个像素的通道值之和，作为归因贡献图 ndarray(224, 224)
    attr_img_our = attributions_our.reshape(1, 1, 224, 224)   #满足评价准指标的计算输入维度

    # 评价指标
    score_lime = quantus.FaithfulnessCorrelation(
        nr_runs=100,
        subset_size=224,
        perturb_baseline="black",
        perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
        similarity_func=quantus.similarity_func.correlation_pearson,
        abs=False,
        return_aggregate=False,
    )(model=model,
      x_batch=input_tensor.cpu().numpy(),
      y_batch=black_pred_id,
      a_batch=attr_img_org,
      device=device)

    score_our = quantus.FaithfulnessCorrelation(
        nr_runs=100,
        subset_size=224,
        perturb_baseline="black",
        perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
        similarity_func=quantus.similarity_func.correlation_pearson,
        abs=False,
        return_aggregate=False,
    )(model=model,
      x_batch=input_tensor.cpu().numpy(),
      y_batch=black_pred_id,
      a_batch=attr_img_our,
      device=device)

    print(score_lime)
    print(score_our)

    figs = [img_pil, fig_cam, img_boundry_org, img_boundry_our,
            attributions_org, img_boundry_org[:, :, 0], img_boundry_org[:, :, 1], img_boundry_org[:, :, 2],
            attributions_our, img_boundry_our[:, :, 0], img_boundry_our[:, :, 1], img_boundry_our[:, :, 2]
            ]

    title_name = ['Original', 'Gradcam', str(explanation_org.top_labels[0])+class_idx.get(str(explanation_org.top_labels[0]))[1]+'-'+'LIME', str(explanation.top_labels[0])+class_idx.get(str(explanation.top_labels[0]))[1]+'-'+'Ours',
                  'Lime_attr', 'Lime_Red', 'Lime_Green', 'Lime_Blue',
                  'Our_attr', 'Our_Red', 'Our_Green', 'Our_Blue']

    # 创建 3 行 4 列的子图布局
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))

    # 在不同的子图中绘制图形
    for i in range(3):
        for j in range(4):
            axes[i, j].imshow(figs[i * 4 + j], cmap="seismic")
            axes[i, j].set_title(title_name[i * 4 + j])

    # 调整布局
    plt.tight_layout()
    plt.show()

    # 创建包含三个子图的图形
    # plt.figure(figsize=(15, 3))
    # plt.subplot(1, 5, 1)
    # plt.title('Lime')
    # plt.imshow(img_boundry)
    #
    # plt.subplot(1, 5, 2)
    # plt.title('Red Channel')
    # plt.imshow(img_boundry[:, :, 0], cmap="seismic")
    #
    # plt.subplot(1, 5, 3)
    # plt.title('Green Channel')
    # plt.imshow(img_boundry[:, :, 1], cmap="seismic")
    #
    # plt.subplot(1, 5, 4)
    # plt.title('Blue Channel')
    # plt.imshow(img_boundry[:, :, 2], cmap="seismic")
    #
    # plt.subplot(1, 5, 5)
    # plt.title('Attributions')
    # plt.imshow(attributions, cmap="seismic")
    # # 调整布局
    # plt.tight_layout()
    # plt.show()
