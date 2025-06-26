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
import quantus



# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

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



#img_path = 'test_img/cat_dog.jpg'
#img_path = 'examples/both.png'
img_path = 'test_img/bird2.png'



img_pil = Image.open(img_path)

#model = models.inception_v3(pretrained=True).eval().to(device)
model = models.resnet18(pretrained=True).eval().to(device)
#print(model)

idx2label, cls2label, cls2idx = [], {}, {}
with open(os.path.abspath('imagenet/imagenet_class_index.json'), 'r') as read_file:
    class_idx = json.load(read_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
    cls2idx = {class_idx[str(k)][0]: k for k in range(len(class_idx))}


input_tensor = trans_A(img_pil).unsqueeze(0).to(device)
pred_logits = model(input_tensor)
pred_softmax = F.softmax(pred_logits, dim=1)
top_n = pred_softmax.topk(1)
#print(f'Predicted:{top_n}')

pred_id = top_n[1].detach().cpu().numpy().squeeze().item()
print(f'黑盒网络预测分类：{pred_id}')
black_pred_id = np.expand_dims(np.array(pred_id), 0)  #进行该处理是为了满足评价准指标的计算输入维度， ndarray(1,)


# test_pred = batch_predict([trans_C(img_pil)])
# black_pred = np.array(test_pred.argmax())
# print(f'黑盒网络预测分类：{black_pred}')
# black_pred_id = np.expand_dims(black_pred, 0)


from lime import lime_image
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(np.array(trans_C(img_pil)), batch_predict, top_labels=1, hide_color=0, num_samples=8000) # LIME生成的邻域图像个数

print(f'解释器预测分类：{explanation.top_labels[0]}')

from skimage.segmentation import mark_boundaries

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=20, hide_rest=False)
img_boundry = mark_boundaries(temp/255.0, mask)  #ndarray(224, 224, 3)
# plt.imshow(img_boundry)
# plt.show()

# 创建包含三个子图的图形
attributions = img_boundry.sum(axis=2)  # 计算每个像素的通道值之和，作为归因贡献图 ndarray(224, 224)
attr_img = attributions.reshape(1, 1, 224, 224)   #满足评价准指标的计算输入维度

# 评价指标
score = quantus.FaithfulnessCorrelation(
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
  a_batch=attr_img,
  device=device)

print(score)

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
