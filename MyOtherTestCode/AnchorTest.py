#参考：https://docs.seldon.io/projects/alibi/en/stable/examples/anchor_image_imagenet.html


import time
import tensorflow as tf

from alibi.explainers import AnchorImage
import matplotlib.pyplot as plt
import os
import json

idx2label, cls2label, cls2idx = [], {}, {}
with open(os.path.abspath('imagenet/imagenet_class_index.json'), 'r') as read_file:
    class_idx = json.load(read_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
    cls2idx = {class_idx[str(k)][0]: k for k in range(len(class_idx))}

if __name__ == '__main__':
    # 创建 ResNet50 模型
    model = tf.keras.applications.ResNet50(weights='imagenet', input_shape=(224, 224, 3))
    print(model)

    # 读取图像并将其调整为模型所需的大小
    img_path = 'test_img/lion_tiger.png'
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))

    # 将图像转换为 NumPy 数组
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    # 添加额外的维度，因为 preprocess_input 需要输入一个 batch
    img_array_ex = tf.expand_dims(img_array, axis=0)
    processed_img = tf.keras.applications.resnet.preprocess_input(img_array_ex)
    print(processed_img.shape)

    pre_model = model.predict(processed_img)
    print(pre_model.shape)

    pre_class = tf.keras.applications.resnet.decode_predictions(pre_model, top=1)
    # test = pre_class[0][0][0]
    black_pre = cls2idx.get(pre_class[0][0][0])
    print(f'black_pre: {black_pre}')

    predict_fn = lambda x: model.predict(x)

    Time = time.time()
    segmentation_fn = 'slic'   # slic, quickshift, felzenszwalb
    slic_kwargs = {'n_segments': 15, 'compactness': 20, 'sigma': .5, 'start_label': 0}
    #quickshift_kwargs = {'ratio': 1, 'kernel_size': 5, 'max_dist': 300, 'return_tree': False, 'sigma': 0.5}
    explainer = AnchorImage(predict_fn, (224, 224, 3), segmentation_fn=segmentation_fn, segmentation_kwargs=slic_kwargs, images_background=None)
    explanation = explainer.explain(img_array, threshold=.95, p_sample=0.5, tau=0.25)
    Anchor_pre = explanation.raw.get('prediction')
    print(f'Anchor_pre: {Anchor_pre}')
    print(f'time: {time.time() - Time}')
    # plt.imshow(img)
    # plt.show()



    plt.imshow(explanation.anchor)
    plt.show()
    #
    # plt.imshow(explanation.segments)
    # plt.show()