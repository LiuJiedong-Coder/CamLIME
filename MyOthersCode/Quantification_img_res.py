#参考：https://github.com/understandable-machine-intelligence-lab/Quantus/blob/main/tutorials/Tutorial_ImageNet_Quantification_with_Quantus.ipynb
import pathlib
import random
import copy
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchvision import models, transforms
from captum.attr import *
import quantus
# print(quantus.available_methods_captum())
# print(quantus.available_methods_tf_explain())
import warnings
warnings.filterwarnings("ignore")

# Plotting specifics.
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

#设置print不省略
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

sns.set()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def saliency_explainer(model, inputs, targets, abs=False, normalise=False, *args, **kwargs) -> np.array:
    """Wrapper aorund captum's Saliency implementation."""

    gc.collect()
    torch.cuda.empty_cache()

    # Set model in evaluate mode.
    model.to(kwargs.get("device", None))
    model.eval()

    if not isinstance(inputs, torch.Tensor):
        inputs = (
            torch.Tensor(inputs)
            .reshape(
                -1,
                kwargs.get("nr_channels", 3),
                kwargs.get("img_size", 224),
                kwargs.get("img_size", 224),
            )
            .to(kwargs.get("device", None))
        )
    if not isinstance(targets, torch.Tensor):
        targets = (
            torch.as_tensor(targets).long().to(kwargs.get("device", None))
        )  # inputs = inputs.reshape(-1, 3, 224, 224)

    assert (
        len(np.shape(inputs)) == 4
    ), "Inputs should be shaped (nr_samples, nr_channels, img_size, img_size) e.g., (1, 3, 224, 224)."

    explanation = (
        Saliency(model)
        .attribute(inputs, targets, abs=abs)
        .sum(axis=1)
        .reshape(-1, kwargs.get("img_size", 224), kwargs.get("img_size", 224))
        .cpu()
        .data
    )

    gc.collect()
    torch.cuda.empty_cache()

    if normalise:
        explanation = quantus.normalise_func.normalise_by_negative(explanation)

    if isinstance(explanation, torch.Tensor):
        if explanation.requires_grad:
            return explanation.cpu().detach().numpy()
        return explanation.cpu().numpy()

    return explanation

def intgrad_explainer(
    model, inputs, targets, abs=False, normalise=False, *args, **kwargs
) -> np.array:
    """Wrapper aorund captum's Integrated Gradients implementation."""

    gc.collect()
    torch.cuda.empty_cache()

    # Set model in evaluate mode.
    model.to(kwargs.get("device", None))
    model.eval()

    if not isinstance(inputs, torch.Tensor):
        inputs = (
            torch.Tensor(inputs)
            .reshape(
                -1,
                kwargs.get("nr_channels", 3),
                kwargs.get("img_size", 224),
                kwargs.get("img_size", 224),
            )
            .to(kwargs.get("device", None))
        )
    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets).long().to(kwargs.get("device", None))

    assert (
        len(np.shape(inputs)) == 4
    ), "Inputs should be shaped (nr_samples, nr_channels, img_size, img_size) e.g., (1, 3, 224, 224)."

    explanation = (
        IntegratedGradients(model)
        .attribute(
            inputs=inputs,
            target=targets,
            baselines=torch.zeros_like(inputs),
            n_steps=10,
            method="riemann_trapezoid",
        )
        .sum(axis=1)
        .reshape(-1, kwargs.get("img_size", 224), kwargs.get("img_size", 224))
        .cpu()
        .data
    )

    gc.collect()
    torch.cuda.empty_cache()

    if normalise:
        explanation = quantus.normalise_func.normalise_by_negative(explanation)

    if isinstance(explanation, torch.Tensor):
        if explanation.requires_grad:
            return explanation.cpu().detach().numpy()
        return explanation.cpu().numpy()

    return explanation


def gradshap_explainer(
    model, inputs, targets, abs=False, normalise=False, *args, **kwargs
) -> np.array:
    """Wrapper aorund captum's GradShap implementation."""

    gc.collect()
    torch.cuda.empty_cache()

    # Set model in evaluate mode.
    model.to(kwargs.get("device", None))
    model.eval()

    if not isinstance(inputs, torch.Tensor):
        inputs = (
            torch.Tensor(inputs)
            .reshape(
                -1,
                kwargs.get("nr_channels", 3),
                kwargs.get("img_size", 224),
                kwargs.get("img_size", 224),
            )
            .to(kwargs.get("device", None))
        )

    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets).long().to(kwargs.get("device", None))

    assert (
        len(np.shape(inputs)) == 4
    ), "Inputs should be shaped (nr_samples, nr_channels, img_size, img_size) e.g., (1, 3, 224, 224)."

    baselines = torch.zeros_like(inputs).to(kwargs.get("device", None))
    explanation = (
        GradientShap(model)
        .attribute(inputs=inputs, target=targets, baselines=baselines)
        .sum(axis=1)
        .reshape(-1, kwargs.get("img_size", 224), kwargs.get("img_size", 224))
        .cpu()
        .data
    )

    gc.collect()
    torch.cuda.empty_cache()

    if normalise:
        explanation = quantus.normalise_func.normalise_by_negative(explanation)

    if isinstance(explanation, torch.Tensor):
        if explanation.requires_grad:
            return explanation.cpu().detach().numpy()
        return explanation.cpu().numpy()

    return explanation

def fusiongrad_explainer(
    model, inputs, targets, abs=False, normalise=False, *args, **kwargs
) -> np.array:
    """Wrapper aorund captum's FusionGrad implementation."""

    std = kwargs.get("std", 0.5)
    mean = kwargs.get("mean", 1.0)
    n = kwargs.get("n", 10)
    m = kwargs.get("m", 10)
    sg_std = kwargs.get("sg_std", 0.5)
    sg_mean = kwargs.get("sg_mean", 0.0)
    posterior_mean = kwargs.get("posterior_mean", None)
    noise_type = kwargs.get("noise_type", "multiplicative")
    clip = kwargs.get("clip", False)

    def _sample(model, posterior_mean, std, distribution=None, noise_type="multiplicative"):
        """Implmentation to sample a model."""

        # Load model params.
        model.load_state_dict(posterior_mean)

        # If std is not zero, loop over each layer and add Gaussian noise.
        if not std == 0.0:
            with torch.no_grad():
                for layer in model.parameters():
                    if noise_type == "additive":
                        layer.add_(distribution.sample(layer.size()).to(layer.device))
                    elif noise_type == "multiplicative":
                        layer.mul_(distribution.sample(layer.size()).to(layer.device))
                    else:
                        print(
                            "Set NoiseGrad attribute 'noise_type' to either 'additive' or 'multiplicative' (str)."
                        )

        return model


    # Creates a normal (also called Gaussian) distribution.
    distribution = torch.distributions.normal.Normal(
        loc=torch.as_tensor(mean, dtype=torch.float),
        scale=torch.as_tensor(std, dtype=torch.float),
    )

    # Set model in evaluate mode.
    model.to(kwargs.get("device", None))
    model.eval()

    if not isinstance(inputs, torch.Tensor):
        inputs = (
            torch.Tensor(inputs)
            .reshape(
                -1,
                kwargs.get("nr_channels", 3),
                kwargs.get("img_size", 224),
                kwargs.get("img_size", 224),
            )
            .to(kwargs.get("device", None))
        )
    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets).long().to(kwargs.get("device", None))

    assert (
        len(np.shape(inputs)) == 4
    ), "Inputs should be shaped (nr_samples, nr_channels, img_size, img_size) e.g., (1, 3, 224, 224)."

    if inputs.shape[0] > 1:
        explanation = torch.zeros(
            (
                n,
                m,
                inputs.shape[0],
                kwargs.get("img_size", 224),
                kwargs.get("img_size", 224),
            )
        )
    else:
        explanation = torch.zeros(
            (n, m, kwargs.get("img_size", 224), kwargs.get("img_size", 224))
        )

    for i in range(n):
        model = _sample(
            model=model,
            posterior_mean=posterior_mean,
            std=std,
            distribution=distribution,
            noise_type=noise_type,
        )
        for j in range(m):
            inputs_noisy = inputs + torch.randn_like(inputs) * sg_std + sg_mean
            if clip:
                inputs_noisy = torch.clip(inputs_noisy, min=0.0, max=1.0)

            explanation[i][j] = (
                Saliency(model)
                .attribute(inputs_noisy, targets, abs=abs)
                .sum(axis=1)
                .reshape(-1, kwargs.get("img_size", 224), kwargs.get("img_size", 224))
                .cpu()
                .data
            )

    explanation = explanation.mean(axis=(0, 1))

    gc.collect()
    torch.cuda.empty_cache()

    if normalise:
        explanation = quantus.normalise_func.normalise_by_negative(explanation)

    if isinstance(explanation, torch.Tensor):
        if explanation.requires_grad:
            return explanation.cpu().detach().numpy()
        return explanation.cpu().numpy()

    return explanation

def explainer_wrapper(**kwargs):
    """Wrapper for explainer functions."""
    if kwargs["method"] == "Saliency":
        return saliency_explainer(**kwargs)
    elif kwargs["method"] == "IntegratedGradients":
        return intgrad_explainer(**kwargs)
    elif kwargs["method"] == "FusionGrad":
        return fusiongrad_explainer(**kwargs)
    elif kwargs["method"] == "GradientShap":
        return gradshap_explainer(**kwargs)
    else:
        raise ValueError("Pick an explaination function that exists.")

# 雷达图 Source code: https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html#sphx-glr-gallery-specialty-plots-radar-chart-py
def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.
    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default."""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default."""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels, angles=None):
            self.set_thetagrids(angles=np.degrees(theta), labels=labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped."""
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)


        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)

                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta



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
    # fig, axes = plt.subplots(nrows=1, ncols=nr_images, figsize=(nr_images*3, int(nr_images*2/3)))
    # for i in range(nr_images):
    #     axes[i].imshow((np.moveaxis(quantus.normalise_func.denormalise(x_batch[i].cpu().numpy(), mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225])), 0, -1) * 255).astype(np.uint8), vmin=0.0, vmax=1.0, cmap="gray")
    #     axes[i].title.set_text(f"ImageNet class - {y_batch[i].item()}")
    #     axes[i].axis("off")
    # plt.show()

    #load model
    #model = models.mobilenet_v3_small(pretrained=True).to(device)
    #model = models.resnet50(pretrained=True).eval().to(device)  会满存
    model = models.resnet18(pretrained=True).eval().to(device)

    # Produce explanations and empty cache to to survive memory-wise.
    #Generate explanations
    # Saliency.
    gc.collect()
    torch.cuda.empty_cache()
    a_batch_saliency = saliency_explainer(model=model.cpu(),
                                          inputs=x_batch,
                                          targets=y_batch,
                                          **{"device": device},
                                          )
    # GradShap.
    gc.collect()
    torch.cuda.empty_cache()
    a_batch_gradshap = gradshap_explainer(model=model.cpu(),
                                          inputs=x_batch,
                                          targets=y_batch,
                                          **{"device": device},
                                          )
    # Integrated Gradients.
    gc.collect()
    torch.cuda.empty_cache()
    a_batch_intgrad = intgrad_explainer(model=model.cpu(),
                                        inputs=x_batch,
                                        targets=y_batch,
                                        **{"device": device},
                                        )

    # FusionGrad
    # gc.collect()
    # torch.cuda.empty_cache()
    # posterior_mean = copy.deepcopy(model.state_dict())
    # a_batch_fusiongrad = fusiongrad_explainer(model=model,
    #                                           inputs=x_batch,
    #                                           targets=y_batch,
    #                                           **{"posterior_mean": posterior_mean, "mean": 1.0, "std": 0.5,
    #                                              "sg_mean": 0.0, "sg_std": 0.5, "n": 25, "m": 25,
    #                                              "noise_type": "multiplicative", "device": device})

    # Save explanations to file.
    explanations = {
        "Saliency": a_batch_saliency,
        "GradientShap": a_batch_gradshap,
        "IntegratedGradients": a_batch_intgrad
        #"FusionGrad": a_batch_fusiongrad
    }

    #print(explanations)

    #Visulise attributions given model and pairs of input-output.
    # index = 10  # random.randint(0, len(x_batch)-1)
    # fig, axes = plt.subplots(nrows=1, ncols=1 + len(explanations), figsize=(15, 8))
    # axes[0].imshow(np.moveaxis(
    #     quantus.normalise_func.denormalise(x_batch[index].cpu().numpy(), mean=np.array([0.485, 0.456, 0.406]),
    #                                        std=np.array([0.229, 0.224, 0.225])), 0, -1), vmin=0.0, vmax=1.0)
    # axes[0].title.set_text(f"ImageNet class {y_batch[index].item()}")
    # axes[0].axis("off");
    # for i, (k, v) in enumerate(explanations.items()):
    #     axes[i + 1].imshow(quantus.normalise_func.normalise_by_negative(explanations[k][index].reshape(224, 224)),
    #                        cmap="seismic", vmin=-1.0, vmax=1.0)
    #     axes[i + 1].title.set_text(f"{k}")
    #     axes[i + 1].axis("off")
    #
    # plt.show()

    ### Quantification with Quantus
    ##analyse the set of explanations under different perspectives:

    # Plotting configs.
    #colours_order = ["#008080", "#FFA500", "#124E78", "#d62728"]
    colours_order = ["#008080", "#FFA500", "#124E78"]
    #methods_order = ["Saliency (SA)", "Integrated\nGradients (IG)", "GradientShap (GS)", "FusionGrad (FG)"]
    methods_order = ["Saliency (SA)", "Integrated\nGradients (IG)", "GradientShap (GS)"]

    plt.rcParams['ytick.left'] = False
    plt.rcParams['ytick.labelleft'] = False
    plt.rcParams['xtick.bottom'] = False
    plt.rcParams['xtick.labelbottom'] = False
    include_titles = True

########################################################
    ## Qualitative analysis
    # first, Plot explanations!
    # index = 10
    # ncols = 1 + len(explanations)
    #
    # fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(15, int(ncols) * 3))
    #
    # for i in range(ncols):
    #
    #     if i == 0:
    #         axes[0].imshow(np.moveaxis(
    #             quantus.normalise_func.denormalise(x_batch[index].cpu().numpy(), mean=np.array([0.485, 0.456, 0.406]),
    #                                                std=np.array([0.229, 0.224, 0.225])), 0, -1), vmin=0.0, vmax=1.0)
    #         if include_titles:
    #             axes[0].set_title(f"ImageNet class {y_batch[index].item()}", fontsize=14)
    #             axes[0].axis("off")
    #
    #     else:
    #
    #         xai = methods_order[i - 1].split("(")[0].replace(" ", "").replace("\n", "")
    #
    #         axes[i].imshow(quantus.normalise_func.normalise_by_negative(explanations[xai][index].reshape(224, 224)),
    #                        cmap="seismic", vmin=-1.0, vmax=1.0)
    #         if include_titles:
    #             axes[i].set_title(f"{methods_order[i - 1]}", fontsize=14)
    #
    #         # Frame configs.
    #         axes[i].xaxis.set_visible([])
    #         axes[i].yaxis.set_visible([])
    #         axes[i].spines["top"].set_color(colours_order[i - 1])
    #         axes[i].spines["bottom"].set_color(colours_order[i - 1])
    #         axes[i].spines["left"].set_color(colours_order[i - 1])
    #         axes[i].spines["right"].set_color(colours_order[i - 1])
    #         axes[i].spines["top"].set_linewidth(5)
    #         axes[i].spines["bottom"].set_linewidth(5)
    #         axes[i].spines["left"].set_linewidth(5)
    #         axes[i].spines["right"].set_linewidth(5)
    #
    # plt.show()
#################################

    #Second, we use Quantus to be able to quantiatively assess the different explanation methods on various evaluation criteria
    # Define XAI methods and metrics.
    xai_methods = list(explanations.keys())
    metrics = {
        "Robustness": quantus.AvgSensitivity(
            nr_samples=10,
            lower_bound=0.2,
            norm_numerator=quantus.norm_func.fro_norm,
            norm_denominator=quantus.norm_func.fro_norm,
            perturb_func=quantus.perturb_func.uniform_noise,
            similarity_func=quantus.similarity_func.difference,
            abs=False,
            normalise=False,
            aggregate_func=np.mean,
            return_aggregate=True,
            disable_warnings=True,
        ),
        "Faithfulness": quantus.FaithfulnessCorrelation(
            nr_runs=10,
            subset_size=224,
            perturb_baseline="black",
            perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
            similarity_func=quantus.similarity_func.correlation_pearson,
            abs=False,
            normalise=False,
            aggregate_func=np.mean,
            return_aggregate=True,
            disable_warnings=True,
        ),
        "Localisation": quantus.RelevanceRankAccuracy(
            abs=False,
            normalise=False,
            aggregate_func=np.mean,
            return_aggregate=True,
            disable_warnings=True,
        ),
        "Complexity": quantus.Sparseness(
            abs=True,
            normalise=False,
            aggregate_func=np.mean,
            return_aggregate=True,
            disable_warnings=True,
        ),
        "Randomisation": quantus.RandomLogit(
            num_classes=1000,
            similarity_func=quantus.similarity_func.ssim,
            abs=True,
            normalise=False,
            aggregate_func=np.mean,
            return_aggregate=True,
            disable_warnings=True,
        ),
    }

    # run quantification analysis!
    results = {method: {} for method in xai_methods}

    for method in xai_methods:
        for metric, metric_func in metrics.items():
            print(f"Evaluating {metric} of {method} method.")
            gc.collect()
            torch.cuda.empty_cache()

            # Get scores and append results.
            scores = metric_func(
                model=model,
                x_batch=x_batch.cpu().numpy(),   # 此处一定要转为narray类型
                y_batch=y_batch.cpu().numpy(),
                a_batch=None,
                s_batch=s_batch.cpu().numpy(),
                device=device,
                explain_func=explainer_wrapper,
                explain_func_kwargs={
                    "method": method,
                    "posterior_mean": copy.deepcopy(
                        model.state_dict()
                    ),
                    "mean": 1.0,
                    "std": 0.5,
                    "sg_mean": 0.0,
                    "sg_std": 0.5,
                    "n": 25,
                    "m": 25,
                    "noise_type": "multiplicative",
                    "device": device,
                },
            )
            results[method][metric] = scores

            # Empty cache.
            gc.collect()
            torch.cuda.empty_cache()

    # Postprocessing of scores: to get how the different explanation methods rank across criteria.
    results_agg = {}
    for method in xai_methods:
        results_agg[method] = {}
        for metric, metric_func in metrics.items():
            results_agg[method][metric] = np.mean(results[method][metric])

    df = pd.DataFrame.from_dict(results_agg)
    #print(df)

    df = df.T.abs()
    #print(df)

    # Take inverse ranking for Robustness, since lower is better.
    df_normalised = df.loc[:, df.columns != 'Robustness'].apply(lambda x: x / x.max())
    df_normalised["Robustness"] = df["Robustness"].min() / df["Robustness"].values
    df_normalised_rank = df_normalised.rank()
    print(df_normalised_rank)

    # Plotting configs.
    sns.set(font_scale=1.8)
    plt.style.use('seaborn-white')
    plt.rcParams['ytick.labelleft'] = True
    plt.rcParams['xtick.labelbottom'] = True

    include_titles = True
    include_legend = True

    # Make spyder graph!
    # data = [df_normalised_rank.columns.values, (df_normalised_rank.to_numpy())]
    # theta = radar_factory(len(data[0]), frame='polygon')
    # spoke_labels = data.pop(0)
    #
    # fig, ax = plt.subplots(figsize=(11, 11), subplot_kw=dict(projection='radar'))
    # fig.subplots_adjust(top=0.85, bottom=0.05)
    # for i, (d, method) in enumerate(zip(data[0], xai_methods)):
    #     line = ax.plot(theta, d, label=method, color=colours_order[i], linewidth=5.0)
    #     ax.fill(theta, d, alpha=0.15)
    #
    # # Set lables.
    # if include_titles:
    #     ax.set_varlabels(labels=['Faithfulness', 'Localisation', '\nComplexity', '\nRandomisation', 'Robustness'])
    # else:
    #     ax.set_varlabels(labels=[])
    #
    # ax.set_rgrids(np.arange(0, df_normalised_rank.values.max() + 0.5), labels=[])
    #
    # # Set a title.
    # ax.set_title("Quantus: Summary of Explainer Quantification", position=(0.5, 1.1), ha='center', fontsize=15)
    #
    # # Put a legend to the right of the current axis.
    # if include_legend:
    #     ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
    #
    # plt.tight_layout()
    # plt.show()

    #### Sensitivity analysis
    #Third, we will investigate how much different parameters influence the evaluation outcome, i.e., how different explanations methods rank.
    #We use Faithfulness Correlation by Bhatt et al., 2020 for this example

    # Define some parameter settings to evaluate.
    baseline_strategies = ["mean", "uniform"]
    subset_sizes = np.array([2, 52, 102])
    sim_funcs = {"pearson": quantus.similarity_func.correlation_pearson,
                 "spearman": quantus.similarity_func.correlation_spearman}

    result = {
        "Faithfulness score": [],
        "Method": [],
        "Similarity function": [],
        "Baseline strategy": [],
        "Subset size": [],
    }

    # Score explanations!
    for b in baseline_strategies:
        for s in subset_sizes:
            for method, attr in explanations.items():
                for sim, sim_func in sim_funcs.items():
                    metric = quantus.FaithfulnessCorrelation(abs=True,
                                                             normalise=True,
                                                             return_aggregate=True,
                                                             disable_warnings=True,
                                                             aggregate_func=np.mean,
                                                             normalise_func=quantus.normalise_func.normalise_by_negative,
                                                             nr_runs=10,
                                                             perturb_baseline=b,
                                                             perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
                                                             similarity_func=sim_func,
                                                             subset_size=s)
                    score = metric(model=model.cuda(), x_batch=x_batch.cpu().numpy(), y_batch=y_batch.cpu().numpy(), a_batch=attr, device=device)
                    result["Method"].append(method)
                    result["Baseline strategy"].append(b.capitalize())
                    result["Subset size"].append(s)
                    result["Faithfulness score"].append(score[0])
                    result["Similarity function"].append(sim)

    df = pd.DataFrame(result)
    #print(df.head())
    #print(df)

    # Group by the ranking.
    df["Rank"] = df.groupby(['Baseline strategy', 'Subset size', 'Similarity function'])["Faithfulness score"].rank()

    # Smaller adjustments.
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = map(lambda x: str(x).capitalize(), df.columns)
    #print(df.head())
    print(df)

    # Group by rank.
    df_view = df.groupby(["Method"])["Rank"].value_counts(normalize=True).mul(100).reset_index(name='Percentage').round(2)

    # Reorder the methods for plotting purporses.
    df_view_ordered = pd.DataFrame(columns=["Method", "Rank", "Percentage"])
    df_view_ordered = pd.concat([
        df_view_ordered, df_view.loc[df_view["Method"] == 'Saliency'],
        df_view.loc[df_view["Method"] == 'GradientShap'],
        df_view.loc[df_view["Method"] == 'IntegratedGradients']], ignore_index=True)

    #pandas库版本过高，DataFrame.append()方法已弃用，改为concat()方法
    # df_view_ordered = df_view_ordered.append([df_view.loc[df_view["Method"] == 'Saliency']], ignore_index=True)
    # df_view_ordered = df_view_ordered.append([df_view.loc[df_view["Method"] == 'GradientShap']], ignore_index=True)
    # df_view_ordered = df_view_ordered.append([df_view.loc[df_view["Method"] == 'IntegratedGradients']], ignore_index=True)
    # df_view_ordered = df_view_ordered.append([df_view.loc[df_view["Method"] == 'FusionGrad']], ignore_index=True)
    print(df_view_ordered)

    # Plot results!
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax = sns.histplot(x='Method', hue='Rank', weights='Percentage', multiple='stack', data=df_view_ordered, shrink=0.6,
                      palette="colorblind", legend=False)
    ax.spines["right"].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylabel('Frequency of rank', fontsize=15)
    ax.set_xlabel('')
    ax.set_xticklabels(["SAL", "GS", "IG"])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True, shadow=False,
               labels=['1st', "2nd", "3rd"])
    plt.axvline(x=3.5, ymax=0.95, color='black', linestyle='-')
    plt.tight_layout()
    plt.show()
