#参考：https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch.ipynb

import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import cv2
import torch
from torchvision import models, transforms
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise, GuidedBackpropReLUModel
)
from pytorch_grad_cam.utils.image import show_cam_on_image

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

coco_names = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', \
              'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
              'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
              'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
              'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
              'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
              'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
              'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
              'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
              'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
              'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# This will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

# Gradient Free Methods to the rescue
def predict(input_tensor, model, device, detection_threshold):
    outputs = model(input_tensor)
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    pred_labels = outputs[0]['labels'].cpu().numpy()
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()

    boxes, classes, labels, indices = [], [], [], []
    for index in range(len(pred_scores)):
        if pred_scores[index] >= detection_threshold:
            boxes.append(pred_bboxes[index].astype(np.int32))
            classes.append(pred_classes[index])
            labels.append(pred_labels[index])
            indices.append(index)
    boxes = np.int32(boxes)
    return boxes, classes, labels, indices


def draw_boxes(boxes, labels, classes, image):
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return image



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
    parser.add_argument('--method', type=str, default='eigencam',
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

trans_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

trans_B = transforms.Compose([
        transforms.ToTensor()
    ])

trans_C = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224)
])

def fasterrcnn_reshape_transform(x):
    target_size = x['pool'].size()[-2 : ]
    activations = []
    for key, value in x.items():
        activations.append(torch.nn.functional.interpolate(torch.abs(value), target_size, mode='bilinear'))
    activations = torch.cat(activations, axis=1)
    return activations

class FasterRCNNBoxScoreTarget:

    # For every original detected bounding box specified in "bounding boxes",
	# 	assign a score on how the current bounding boxes match it,
	# 		1. In IOU
	# 		2. In the classification score.
	# 	If there is not a large enough overlap, or the category changed,
	# 	assign a score of 0.
	# 	The total score is the sum of all the box scores.

    def __init__(self, labels, bounding_boxes, iou_threshold=0.5):
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.iou_threshold = iou_threshold

    def __call__(self, model_outputs):
        output = torch.Tensor([0])
        if torch.cuda.is_available():
            output = output.cuda()

        if len(model_outputs["boxes"]) == 0:
            return output

        for box, label in zip(self.bounding_boxes, self.labels):
            box = torch.Tensor(box[None, :])
            if torch.cuda.is_available():
                box = box.cuda()

            ious = torchvision.ops.box_iou(box, model_outputs["boxes"])
            index = ious.argmax()
            if ious[0, index] > self.iou_threshold and model_outputs["labels"][index] == label:
                score = ious[0, index] + model_outputs["scores"][index]
                output = output + score
        return output


def batch_predict(images):
    batch = torch.stack(tuple(trans_B(i) for i in images), dim=0)
    batch = batch.to(device)

    pred_n = 1
    logits = model(batch)
    preds = []
    for i in range(len(logits)):
        pred_scores = logits[i]['scores']
        top_n = pred_scores.topk(pred_n)
        pred = top_n[0].detach().cpu().numpy()
        if pred_n != 1:
            pred = pred.squeeze()
        preds.append(pred)
    return preds


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
    #img_path = 'test_img/both.png'    #eigengradcam，fullgrad效果改进好
    img_path = 'odb_img/bear.jpg'
    #img_path = 'test_img/bird2.png'
    #img_path = 'test_img/ILSVRC2012_val_00011333.JPEG'

    img_pil = Image.open(img_path)
    image = np.array(img_pil)
    image_float_np = np.float32(image) / 255

    #model = models.resnet18(pretrained=True).eval().to(device)
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval().to(device)
    #print(model)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Run the model and display the detections
    boxes, classes, pre_labels, indices = predict(input_tensor, model, device, 0.9)
    image_box = draw_boxes(boxes, pre_labels, classes, image)
    # plt.imshow(image)
    # plt.show()

    target_layers = [model.backbone]
    targets = [FasterRCNNBoxScoreTarget(labels=pre_labels, bounding_boxes=boxes)]
    cam = EigenCAM(model,
                   target_layers,
                   use_cuda=torch.cuda.is_available(),
                   reshape_transform=fasterrcnn_reshape_transform)

    grayscale_cam = cam(input_tensor, targets=targets)
    # Take the first image in the batch:
    grayscale_cam = grayscale_cam[0, :]

    ####################OUrsLIME#####################################
    from lime import lime_image_my

    explainer = lime_image_my.LimeImageExplainer()  # 个人修改

    # batch_predict分类预测函数，num_samples是LIME生成的邻域图像个数
    data, labels = explainer.explain_instance_data_label(image, batch_predict, top_labels=1, hide_color=0, num_samples=100, random_seed=1000, batch_size=5, detect_task=True)

    explanation = explainer.explain_instance(grayscale_cam)
    from skimage.segmentation import mark_boundaries

    num_features = 120
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=num_features, hide_rest=False)
    img_boundry = mark_boundaries(temp / 255.0, mask)

    image_with_bounding_boxes = draw_boxes(boxes, pre_labels, classes, (img_boundry*255).astype(np.uint8))

    plt.imshow(image_with_bounding_boxes)
    plt.show()

