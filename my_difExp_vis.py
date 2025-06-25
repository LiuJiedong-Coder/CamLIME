#参考：https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch.ipynb

import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json
import time
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

from captum.attr import IntegratedGradients, NoiseTunnel, Occlusion, GradientShap, LRP
from matplotlib.colors import LinearSegmentedColormap
from captum.attr import visualization as viz

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
    parser.add_argument('--method', type=str, default='gradcam',
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

trans_D = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])


def batch_predict(images):
    batch = torch.stack(tuple(trans_B(i) for i in images), dim=0)
    batch = batch.to(device)

    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

import tensorflow as tf
from keras.preprocessing import image
from alibi.explainers import AnchorImage
def anchorExp(img_path):
    model = tf.keras.applications.MobileNet(weights='imagenet', input_shape=(224, 224, 3))
    img = image.load_img(img_path, target_size=(224, 224))
    # 将图像转换为 NumPy 数组
    img_array = image.img_to_array(img)
    # 添加额外的维度，因为 preprocess_input 需要输入一个 batch
    img_array_ex = tf.expand_dims(img_array, axis=0)
    processed_img = tf.keras.applications.resnet.preprocess_input(img_array_ex)
    pre_model = model.predict(processed_img)
    pre_class = tf.keras.applications.resnet.decode_predictions(pre_model, top=1)
    black_pre = cls2idx.get(pre_class[0][0][0])
    print(f'tf_black_pre: {black_pre}')

    predict_fn = lambda x: model.predict(x)

    #Time = time.time()
    segmentation_fn = 'slic'  # slic, quickshift, felzenszwalb
    slic_kwargs = {'n_segments': 15, 'compactness': 20, 'sigma': .5, 'start_label': 0}
    explainer = AnchorImage(predict_fn, (224, 224, 3), segmentation_fn=segmentation_fn, segmentation_kwargs=slic_kwargs, images_background=None)
    explanation = explainer.explain(img_array, threshold=.95, p_sample=0.5, tau=0.25)
    Anchor_pre = explanation.raw.get('prediction')
    #print(f'Anchor_pre: {Anchor_pre}')
    #print(f'time: {time.time() - Time}')
    # plt.imshow(explanation.anchor)
    # plt.show()
    # plt.imshow(explanation.segments)
    # plt.show()
    return explanation

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
    #img_path = 'test_img/both.png'
    img_path = 'test_img/cat_dog.jpg'

    img_pil = Image.open(img_path)

    #model = models.inception_v3(pretrained=True).eval().to(device)
    #model = models.resnet18(pretrained=True).eval().to(device)
    model = models.mobilenet_v2(pretrained=True).eval().to(device)
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
    #target_layers = [model.layer4]  #resnet18/50
    #target_layers = [model.Mixed_7c]  #inception_v3
    target_layers = [model.features[17]]  #mobilenet_v2

    with cam_algorithm(model=model, target_layers=target_layers, use_cuda=True) as cam:
        # print(cam.activations_and_grads.activations, cam.activations_and_grads.gradients) #(1, 512, 7, 7)
        # activations = cam.activations_and_grads.activations   #(1, 512, 7, 7), 特征图
        # gradients = cam.activations_and_grads.gradients       #(1, 512, 7, 7), 梯度图

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=args.aug_smooth, eigen_smooth=args.eigen_smooth)  # 激活图(1,7,7)
        grayscale_cam = grayscale_cam[0, :]  #cam激活图，ndarray(7,7)
        #grayscale_cam = grayscale_cam[0]  # cam激活图，ndarray(7,7)
        from torchcam.utils import overlay_mask
        fig_cam = overlay_mask(img_pil, Image.fromarray(grayscale_cam), alpha=0.4)  #alpha越小，原图越淡

    # test_data = trans_B(np.array(trans_C(img_pil))) #ndarry(224,224,3)
    # print(test_data.shape)

    random_seed = 500
####################原始LIME######################################
    from lime import lime_image
    explainer_org = lime_image.LimeImageExplainer()
    explanation_org = explainer_org.explain_instance(np.array(trans_C(img_pil)), batch_predict, top_labels=1, hide_color=0, num_samples=8000, random_seed=random_seed)
    #explanation_org = explainer_org.explain_instance(np.array(trans_C(Image.fromarray(grayscale_cam))), batch_predict, top_labels=1, hide_color=0, num_samples=8000, random_seed=random_seed)
    #explanation_org = explainer_org.explain_instance(np.array(trans_C(fig_cam)), batch_predict, top_labels=1, hide_color=0, num_samples=8000, random_seed=random_seed)

####################OUrsLIME#####################################
    from lime import lime_image_my
    explainer = lime_image_my.LimeImageExplainer()  #个人修改

    # batch_predict分类预测函数，num_samples是LIME生成的邻域图像个数
    data, labels = explainer.explain_instance_data_label(np.array(trans_C(img_pil)), batch_predict, top_labels=1, hide_color=0, num_samples=8000, random_seed=random_seed)

    explanation = explainer.explain_instance(grayscale_cam)
    print(f'解释器预测分类{explanation.top_labels}')
    #print(class_idx.get(str(explanation.top_labels[0]))[1])


#############IntegratedGradients########################
    targets2 = torch.tensor(pred_id)
    transformed_img = trans_D(img_pil)
    integrated_gradients = IntegratedGradients(model)
    attributions_ig = integrated_gradients.attribute(input_tensor, target=targets2, n_steps=200)

    default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                     [(0, '#ffffff'),
                                                      (0.25, '#000000'),
                                                      (1, '#000000')], N=256)

    vis, ax = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                 np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                 method='heat_map',
                                 cmap=default_cmap,
                                 show_colorbar=True,
                                 sign='positive',   # positive、all、absolute_value、negative
                                 outlier_perc=1,
                                 use_pyplot=False)

    IG_Img = ax.get_images()[0].get_array()
    # plt.imshow(IG_Img)
    # plt.show()
#############GradientShap########################
    gradient_shap = GradientShap(model)

    # Defining baseline distribution of images
    rand_img_dist = torch.cat([input_tensor * 0, input_tensor * 1])

    attributions_gs = gradient_shap.attribute(input_tensor,
                                              n_samples=50,
                                              stdevs=0.0001,
                                              baselines=rand_img_dist,
                                              target=targets2)
    vis, ax = viz.visualize_image_attr(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                          np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                          method='heat_map',
                                          sign='positive',
                                          cmap=default_cmap,
                                          outlier_perc=1,
                                          show_colorbar=True,
                                          use_pyplot=False)

    GS_Img = ax.get_images()[0].get_array()

#############Occlusion########################
    occlusion = Occlusion(model)

    attributions_occ = occlusion.attribute(input_tensor,
                                           strides=(3, 8, 8),
                                           target=targets2,
                                           sliding_window_shapes=(3, 15, 15),
                                           baselines=0)

    vis, ax= viz.visualize_image_attr(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                          np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                          method='heat_map',
                                          sign='positive',
                                          show_colorbar=True,
                                          outlier_perc=2,
                                          use_pyplot=False)

    OCC_Img = ax.get_images()[0].get_array()

#########################Anchor#############
    anchor_exp = anchorExp(img_path)
    Anchor_Img = anchor_exp.anchor


    from skimage.segmentation import mark_boundaries

    plt.figure(figsize=(18, 3))

    plt.subplot(1, 8, 1)
    plt.title('Original')
    plt.tick_params(labelsize=6)
    plt.imshow(img_pil)

    plt.subplot(1, 8, 2)
    plt.title('IG_Img')
    plt.tick_params(labelsize=6)
    plt.imshow(IG_Img)

    plt.subplot(1, 8, 3)
    plt.title('GS_Img')
    plt.tick_params(labelsize=6)
    plt.imshow(GS_Img)

    plt.subplot(1, 8, 4)
    plt.title('OCC_Img')
    plt.tick_params(labelsize=6)
    plt.imshow(OCC_Img)

    plt.subplot(1, 8, 5)
    plt.title('Anchor_Img')
    plt.tick_params(labelsize=6)
    plt.imshow(Anchor_Img)

    plt.subplot(1, 8, 6)
    plt.title('fig_Gradcam')
    plt.tick_params(labelsize=6)
    plt.imshow(fig_cam)

    num_features = 20
    plt.subplot(1, 8, 7)
    temp, mask = explanation_org.get_image_and_mask(explanation_org.top_labels[0], positive_only=False, num_features=num_features, hide_rest=False)
    img_boundry = mark_boundaries(temp / 255.0, mask)
    plt.title(str(explanation_org.top_labels[0])+class_idx.get(str(explanation_org.top_labels[0]))[1]+'-'+'LIME', fontsize=8)
    plt.tick_params(labelsize=6)  # 设置坐标轴刻度大小
    plt.imshow(img_boundry)

    plt.subplot(1, 8, 8)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=num_features, hide_rest=False)
    img_boundry = mark_boundaries(temp / 255.0, mask)
    plt.title(str(explanation.top_labels[0])+class_idx.get(str(explanation.top_labels[0]))[1]+'-'+'Ours', fontsize=8)
    plt.tick_params(labelsize=6)  # 设置坐标轴刻度大小
    plt.imshow(img_boundry)

    plt.show()




