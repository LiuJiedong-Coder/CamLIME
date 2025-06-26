#https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20Faces%20and%20GradBoost.ipynb

import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
import numpy as np
from skimage.color import gray2rgb, rgb2gray # since the code wants color images
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import label2rgb
# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)


class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]

def test_model(model, test_data):
    # switch the model to evaluation mode
    model.eval()

    correct = 0
    total = 0

    # we don't need gradients for validation, so wrap in
    # no_grad to save memory
    with torch.no_grad():
        for inputs, labels in test_data:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    #print('Test Accuracy: %d %%' % (100 * correct / total))
    return 100 * correct / total

def train_model(model, train_data, test_data, epoch=1):

    model = model.to(device)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)



    for epoch in range(epoch):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        for inputs, labels in train_data:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # backward + optimize
            loss.backward()
            optimizer.step()
        #exp_lr_scheduler.step()
        test_acc = test_model(model, test_data)
        print(f'Epoch{epoch+1} Test Accuracy: {test_acc}')

    print('Finished Training')

trans_B = transforms.Compose([
        transforms.ToTensor()
    ])

def batch_predict(images):
    batch = torch.stack(tuple(trans_B(i) for i in images), dim=0)
    batch = batch.to(device)

    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()



if __name__ == '__main__':
    # 加载数据集
    faces = fetch_olivetti_faces()
    # make each image color so lime_image works correctly
    X_vec = np.stack([gray2rgb(iimg) for iimg in faces.data.reshape((-1, 64, 64))], 0)
    X_vec = np.transpose(X_vec, (0, 3, 1, 2))  # change to NCHW
    y_vec = faces.target.astype(np.uint8)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y_vec, train_size=0.70)

    # 准备训练数据
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()

    train_data = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)

    # 准备测试数据
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()

    test_data = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=True)

    # 载入resnet18预训练模型
    model = models.resnet18(pretrained=True)

    # 修改最后一层以适应你的分类任务，这里假设你有40个类别
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 40)

    startTime = time.time()
    train_model(model, train_loader, test_loader, epoch=7)
    print("训练时间：", time.time() - startTime)

    to_image = transforms.ToPILImage()
    img_pil = X_test[0]
    Img = to_image((img_pil * 255).byte()).convert("RGB")

    input_tensor = X_test[0].unsqueeze(0).to(device)

    n_pred = 1
    pred_logits = model(input_tensor)
    pred_softmax = F.softmax(pred_logits, dim=1)
    top_n = pred_softmax.topk(n_pred)  # 取最可能的结果
    print(f'前{n_pred}个预测：{top_n}')
    pred_id = top_n[1].detach().cpu().numpy()
    if n_pred != 1:
        pred_id = pred_id.squeeze()
    print(f'黑盒 Max predicted labels:{pred_id}')

    targets = [ClassifierOutputTarget(pred_id)]
    target_layers = [model.layer4[-1]]
    with GradCAM(model=model, target_layers=target_layers, use_cuda=True) as cam:

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)  # cam激活图(1,7,7)
        # activations = cam.activations_and_grads.activations   #(1, 512, 7, 7), 特征图
        # gradients = cam.activations_and_grads.gradients       #(1, 512, 7, 7), 梯度图
        grayscale_cam = grayscale_cam[0, :]  # cam激活图，ndarray(7,7)
        from torchcam.utils import overlay_mask
        cam_image = overlay_mask(Img, Image.fromarray(grayscale_cam), alpha=0.4)  # alpha越小，原图越淡
    # plt.imshow(cam_image)
    # plt.show()

    from lime import lime_image_my
    from lime.wrappers.scikit_image import SegmentationAlgorithm
    segmenter = SegmentationAlgorithm('slic', n_segments=100, compactness=1, sigma=1)

    #explainer = lime_image.LimeImageExplainer()  #源码
    explainer = lime_image_my.LimeImageExplainer()  #个人修改

    #explanation = explainer.explain_instance(rgb_img_org, batch_predict, top_labels=5, hide_color=0, num_samples=8000)  # batch_predict分类预测函数，num_samples是LIME生成的邻域图像个数
    data, labels = explainer.explain_instance_data_label(np.array(Img), batch_predict, top_labels=5, hide_color=0, num_samples=10000, segmentation_fn=segmenter)  # batch_predict分类预测函数，num_samples是LIME生成的邻域图像个数


    explanation = explainer.explain_instance(grayscale_cam)
    print(f'解释器预测分类{explanation.top_labels}')

    from skimage.segmentation import mark_boundaries
    # temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=20, hide_rest=False)
    # img_boundry = mark_boundaries(temp / 255.0, mask)
    # plt.imshow(img_boundry)
    # plt.show()

    fig, m_axs = plt.subplots(2, 5, figsize=(10, 4))
    for i, (c_ax, gt_ax) in zip(explanation.top_labels, m_axs.T):
        temp, mask = explanation.get_image_and_mask(i, positive_only=True, num_features=4, hide_rest=False)
        c_ax.imshow(label2rgb(mask, temp, bg_label=0), interpolation='nearest')
        c_ax.set_title('Test positive for {}'.format(i))
        c_ax.axis('off')
        face_id = np.random.choice(np.where(y_train == i)[0])
        gt_ax.imshow(np.array(X_train[face_id].permute(1, 2, 0)))
        gt_ax.set_title('Train example of {}'.format(i))
        gt_ax.axis('off')
    plt.tight_layout()  # 调整子图排版
    plt.show()