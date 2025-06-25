#参考：https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch.ipynb

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
import time
import PIL
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


def batch_predict(images):
    batch = torch.stack(tuple(trans_B(i) for i in images), dim=0)
    batch = batch.to(device)

    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


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

def write_to_file(sample_info, sample_num, class_name, sample_info_count):
    with open('results/difmodel.txt', 'a') as file:
        file.write(f'-------------File_Name: {class_name}  label: {cls2idx.get(class_name)}----------------\n')
        file.write(f'sample_info: \n')
        for i in range(len(sample_info)):
                file.write(f'{sample_info[i]}\n')

        file.write(f'sample_info_count: {sample_info_count}\n')
        file.write(f'sample_num: {sample_num}\n')
        file.write(f'-------------------------------------------' + os.linesep)

def end_write_to_file(sample_num, total_cor_black,time):
    # 打开文件以写入模式，如果文件不存在则会创建
    with open('results/difmodel.txt', 'a') as file:
        file.write(f'--------------------Calculate End---------------------' + os.linesep)
        file.write(f'total_sample_num: {sample_num}\n')
        file.write(f'total_correct_black: {total_cor_black}\n')
        file.write(f'total_time: {time:.2f}  sample_avg_explain_time: {time/sample_num:.2f}\n')
        file.write(f'-------------------------------------------' + os.linesep)

def count_matching_values(sample_info):
    # 初始化计数器
    counters = {}

    # 遍历列表中的每个元素（字典）
    for sample in sample_info:
        keys = list(sample.keys())
        img_value = sample[keys[0]]

        # 遍历除第一个键（img_value）之外的其他键
        for key in keys[1:]:
            # 如果键未在计数器中，则添加到计数器中并从零开始
            if key not in counters:
                counters[key] = 0

            # 比较当前键对应的值是否等于 img_value，如果相等则增加计数器
            if sample[key] == img_value:
                counters[key] += 1

    return counters

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
    #img_path = 'examples/both.png'
    #img_path = 'test_img/lion_tiger.png'

    base_dir = 'F:\Databases\ImageNet2012/val'
    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    # 模型根据参数量由小到大
    model_mob = models.mobilenet_v2(pretrained=True).eval().to(device)
    model_alex = models.alexnet(pretrained=True).eval().to(device)
    model_google = models.googlenet(pretrained=True).eval().to(device)
    model_res18 = models.resnet18(pretrained=True).eval().to(device)
    model1_inc = models.inception_v3(pretrained=True).eval().to(device)
    model_dense = models.densenet121(pretrained=True).eval().to(device)
    #model_vgg = models.vgg16(pretrained=True).eval().to(device)




    target_layers_mob = [model_mob.features[17]]   #mobilenet
    target_layers_alex = [model_alex.features[11]]   #alexnet
    target_layers_google = [model_google.inception5b]   #googlenet
    target_layers_res18 = [model_res18.layer4[-1]]   #resnet18,50
    target_layers_inc = [model1_inc.Mixed_7c]   #inception_v3
    target_layers_dense = [model_dense.features[-1]]  # vgg / densenet
    #target_layers_vgg = [model_vgg.features[-1]]


    model_and_target_layers = [(model_mob, target_layers_mob),
                               (model_alex, target_layers_alex),
                               (model_google, target_layers_google),
                               (model_res18, target_layers_res18),
                               (model1_inc, target_layers_inc),
                               (model_dense, target_layers_dense)
                               #(model_vgg, target_layers_vgg)
                               ]

    # model_and_target_layers = [
    #                            (model_res18, target_layers_res18),
    #                            (model_dense, target_layers_dense)
    #                            ]

    n_files = 0
    total_sample_num = 0
    total_cor_black = {}
    Time = time.time()

    for class_name in classes:
        label = cls2idx.get(class_name)
        print(f'-----------------------该file_name为: {class_name} 对应label为: {label}------------------------')
        class_dir = os.path.join(base_dir, class_name)
        paths, loaded_images = load_images_from_folder(class_dir)
        sample_num = 0
        cor_black = 0
        sample_info = []
        i = 0

        for img_path in range(len(loaded_images)):
            info = {}
            img_pil = loaded_images[i]

            model_pre_dict = {}
            model_figs = {}
            for model, target_layers in model_and_target_layers:
                n_pred = 1  #取前n个预测结果
                title_figs = {}
                input_tensor = trans_A(img_pil).unsqueeze(0).to(device)
                pred_logits = model(input_tensor)
                pred_softmax = F.softmax(pred_logits, dim=1)
                top_n = pred_softmax.topk(n_pred)
                print(f'前{n_pred}个预测：{top_n}')
                pred_id = top_n[1].detach().cpu().numpy().squeeze().item()
                print(f'黑盒{model.__class__.__name__} Max predicted labels:{pred_id}')

                if pred_id == label:
                    cor_black += 1

                targets = [ClassifierOutputTarget(pred_id)]
                cam_algorithm = methods[args.method]

                with cam_algorithm(model=model, target_layers=target_layers, use_cuda=True) as cam:
                    # print(cam.activations_and_grads.activations, cam.activations_and_grads.gradients) #(1, 512, 7, 7)
                    # activations = cam.activations_and_grads.activations   #(1, 512, 7, 7), 特征图
                    # gradients = cam.activations_and_grads.gradients       #(1, 512, 7, 7), 梯度图

                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=args.aug_smooth, eigen_smooth=args.eigen_smooth)  # 激活图(1,7,7)
                    grayscale_cam = grayscale_cam[0, :]  #cam激活图，ndarray(7,7)
                    #grayscale_cam = grayscale_cam[0]  # cam激活图，ndarray(7,7)
                    from torchcam.utils import overlay_mask
                    fig_cam = overlay_mask(img_pil, Image.fromarray(grayscale_cam), alpha=0.4)  #alpha越小，原图越淡


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


                model_pre_dict[model.__class__.__name__] = pred_id

                from skimage.segmentation import mark_boundaries
                num_features = 20
                temp, mask = explanation_org.get_image_and_mask(explanation_org.top_labels[0], positive_only=False, num_features=num_features, hide_rest=False)
                img_boundry_org = mark_boundaries(temp / 255.0, mask)

                temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=num_features, hide_rest=False)
                img_boundry_our = mark_boundaries(temp / 255.0, mask)

                title_figs['Original'] = img_pil
                title_figs['Activation'] = grayscale_cam
                #title_figs['fig_Gradcam'] = fig_cam
                title_figs['LIME'] = img_boundry_org
                title_figs['Our'] = img_boundry_our
                model_figs[model.__class__.__name__] = title_figs

                print()
            # 绘图

            fig, axes = plt.subplots(len(model_figs), 4, figsize=(10, 2*len(model_figs)))
            # 在不同的子图中绘制图形
            for ix, (model, images) in enumerate(model_figs.items()):
                for jx, (title,  fig) in enumerate(images.items()):
                    axes[ix, jx].imshow(fig)
                    if jx == 0:
                        axes[ix, jx].set_title(f'{title}:{label}-{model}:{pred_id}', fontsize=8)
                    else:
                        axes[ix, jx].set_title(title, fontsize=8)
                    axes[ix, jx].set_xticks([])
                    axes[ix, jx].set_yticks([])

            plt.tight_layout()
            img_name = os.path.splitext(os.path.basename(paths[i]))[0].split('_')[-1]
            plt.savefig(f'output/test/{img_name}.png')
            #plt.show()

            info[paths[i]] = label
            info.update(model_pre_dict)
            sample_info.append(info)

            sample_num += 1
            i += 1
            if i == 1:
                break
        #print(sample_info)
        sample_info_count = count_matching_values(sample_info)    # 返回字典，计算不同黑盒预测正确数 {'ResNet': num1, 'DenseNet': num2,...}
        write_to_file(sample_info, sample_num, class_name, sample_info_count)
        total_sample_num += sample_num

        for key, value in sample_info_count.items():
            if key not in total_cor_black:
                total_cor_black[key] = 0
            total_cor_black[key] += value

        n_files += 1
        if n_files == 2:   #控制计算的样本数
            break

    endTime = time.time() - Time
    print('total_time:', endTime)
    print('total_sample_num:', total_sample_num)
    end_write_to_file(total_sample_num, total_cor_black,endTime)
    print('total_cor_black:', total_cor_black)

