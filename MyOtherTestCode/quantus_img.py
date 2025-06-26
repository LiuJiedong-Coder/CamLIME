
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import models, transforms

import torch.nn.functional as F
import quantus
from typing import Union
import gc
from captum.attr import *


# Enable GPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def normalize_image(arr: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    if isinstance(arr, torch.Tensor):
        arr_copy = arr.clone().cpu().numpy()
    else:
        arr_copy = arr.copy()

    arr_copy = quantus.normalise_func.denormalise(arr_copy, mean=mean, std=std)
    arr_copy  = np.moveaxis(arr_copy, 0, -1)
    arr_copy = (arr_copy * 255.).astype(np.uint8)
    return arr_copy


if __name__ == "__main__":

    # Adjust this path.
    path_to_files = "tutorials/assets/imagenet_samples"

    # Load test data and make loaders.
    x_batch = torch.load(f'{path_to_files}/x_batch.pt')  #tensor of shape (17, 3, 224, 224)
    y_batch = torch.load(f'{path_to_files}/y_batch.pt')
    s_batch = torch.load(f'{path_to_files}/s_batch.pt')  #tensor of shape (17, 224, 224)
    s_batch = torch.unsqueeze(s_batch, dim=1)  #扩展为与x_batch相同形状的tensor
    x_batch, s_batch, y_batch = x_batch.to(device), s_batch.to(device), y_batch.to(device)
    print(f"{len(x_batch)} matches found.")

    # Plot some inputs!
    # nr_images = 5
    # fig, axes = plt.subplots(nrows=1, ncols=nr_images, figsize=(nr_images * 3, int(nr_images * 2 / 3)))
    # for i in range(nr_images):
    #     axes[i].imshow(normalize_image(x_batch[i]), vmin=0.0, vmax=1.0, cmap="gray")
    #     axes[i].title.set_text(f"ImageNet class - {y_batch[i].item()}")
    #     axes[i].axis("off")
    # plt.show()

    model = models.resnet18(pretrained=True).eval().to(device)

    a_batch = quantus.explain(model, x_batch, y_batch, method="IntegratedGradients") #ndarray(17, 1, 224, 224)


    #index = random.randint(0, len(x_batch) - 1)
    # index = 1
    # # Plot examplary explanations!
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))
    # axes[0].imshow(normalize_image(x_batch[index]), vmin=0.0, vmax=1.0)
    # axes[0].title.set_text(f"ImageNet class {y_batch[index].item()}")
    # exp = axes[1].imshow(a_batch[index].reshape(224, 224), cmap="seismic")
    # fig.colorbar(exp, fraction=0.03, pad=0.05)
    # axes[0].axis("off")
    # axes[1].axis("off")
    # plt.show()

    # Return faithfulness correlation scores in an one-liner - by calling the metric instance.
    score = quantus.FaithfulnessCorrelation(
        nr_runs=100,
        subset_size=224,
        perturb_baseline="black",
        perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
        similarity_func=quantus.similarity_func.correlation_pearson,
        abs=False,
        return_aggregate=False,
    )(model=model,
      x_batch=x_batch.cpu().numpy(),
      y_batch=y_batch.cpu().numpy(),
      a_batch=a_batch,
      device=device)

    print(score)