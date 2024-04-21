#代码参考：https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20Faces%20and%20GradBoost.ipynb
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import gray2rgb, rgb2gray # since the code wants color images
from skimage.util import montage as montage2d

from sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces()
# make each image color so lime_image works correctly
X_vec = np.stack([gray2rgb(iimg) for iimg in faces.data.reshape((-1, 64, 64))],0)
y_vec = faces.target.astype(np.uint8)


#show image
# fig, ax1 = plt.subplots(1,1, figsize = (8,8))
# ax1.imshow(montage2d(X_vec[:, :, :, 0]), cmap='gray', interpolation = 'none')
# ax1.set_title('All Faces')
# ax1.axis('off')
# plt.show()

#Setup a Pipeline
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA

class PipeStep(object):
    """
    Wrapper for turning functions into pipeline transforms (no-fitting)
    """
    def __init__(self, step_func):
        self._step_func=step_func
    def fit(self,*args):
        return self
    def transform(self,X):
        return self._step_func(X)

makegray_step = PipeStep(lambda img_list: [rgb2gray(img) for img in img_list])
flatten_step = PipeStep(lambda img_list: [img.ravel() for img in img_list])

simple_rf_pipeline = Pipeline([
    ('Make Gray', makegray_step),
    ('Flatten Image', flatten_step),
    ('Normalize', Normalizer()),
    ('PCA', PCA(25)),
    ('XGBoost', GradientBoostingClassifier())
                              ])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y_vec,
                                                    train_size=0.70)

simple_rf_pipeline.fit(X_train, y_train)

# Scoring the Model
# compute on remaining test data
pipe_pred_test = simple_rf_pipeline.predict(X_test)
pipe_pred_prop = simple_rf_pipeline.predict_proba(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_true=y_test, y_pred = pipe_pred_test))

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
explainer = lime_image.LimeImageExplainer(verbose = False)
segmenter = SegmentationAlgorithm('slic', n_segments=100, compactness=1, sigma=1)

x = X_test[0]

explanation = explainer.explain_instance(X_test[0],
                                         classifier_fn = simple_rf_pipeline.predict_proba,
                                         top_labels=6, hide_color=0, num_samples=10000, segmentation_fn=segmenter)

from skimage.color import label2rgb
temp, mask = explanation.get_image_and_mask(y_test[0], positive_only=True, num_features=5, hide_rest=False)
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (8, 4))
ax1.imshow(label2rgb(mask,temp, bg_label = 0), interpolation = 'nearest')
ax1.set_title('Positive Regions for {}'.format(y_test[0]))
temp, mask = explanation.get_image_and_mask(y_test[0], positive_only=False, num_features=10, hide_rest=False)
ax2.imshow(label2rgb(3-mask,temp, bg_label = 0), interpolation = 'nearest')
ax2.set_title('Positive/Negative Regions for {}'.format(y_test[0]))
plt.show()

# now show them for each class
# fig, m_axs = plt.subplots(2,6, figsize = (12,4))
# for i, (c_ax, gt_ax) in zip(explanation.top_labels, m_axs.T):
#     temp, mask = explanation.get_image_and_mask(i, positive_only=True, num_features=5, hide_rest=False, min_weight=0.01)
#     c_ax.imshow(label2rgb(mask,temp, bg_label = 0), interpolation = 'nearest')
#     c_ax.set_title('Positive for {}\nScore:{:2.2f}%'.format(i, 100*pipe_pred_prop[0, i]))
#     c_ax.axis('off')
#     face_id = np.random.choice(np.where(y_train==i)[0])
#     gt_ax.imshow(X_train[face_id])
#     gt_ax.set_title('Example of {}'.format(i))
#     gt_ax.axis('off')
#
# plt.show()

#Gaining Insight
#Can we find an explanation for a classification the algorithm got wrong

# wrong_idx = np.random.choice(np.where(pipe_pred_test!=y_test)[0])
#
# print('Using #{} where the label was {} and the pipeline predicted {}'.format(wrong_idx, y_test[wrong_idx], pipe_pred_test[wrong_idx]))
# explanation = explainer.explain_instance(X_test[wrong_idx],
#                                          classifier_fn = simple_rf_pipeline.predict_proba,
#                                          top_labels=6, hide_color=0, num_samples=10000, segmentation_fn=segmenter)
#
# # now show them for each class
# fig, m_axs = plt.subplots(2,6, figsize = (12,4))
# for i, (c_ax, gt_ax) in zip(explanation.top_labels, m_axs.T):
#     temp, mask = explanation.get_image_and_mask(i, positive_only=True, num_features=5, hide_rest=False, min_weight=0.01)
#     c_ax.imshow(label2rgb(mask,temp, bg_label = 0), interpolation = 'nearest')
#     c_ax.set_title('Positive for {}\nScore:{:2.2f}%'.format(i, 100*pipe_pred_prop[wrong_idx, i]))
#     c_ax.axis('off')
#     face_id = np.random.choice(np.where(y_train==i)[0])
#     gt_ax.imshow(X_train[face_id])
#     gt_ax.set_title('Example of {}'.format(i))
#     gt_ax.axis('off')
#
# plt.show()