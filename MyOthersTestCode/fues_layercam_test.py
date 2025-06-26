import matplotlib.pyplot as plt
import torch
from torch.nn.functional import softmax, interpolate
from torchvision.io.image import read_image
from torchvision.models import resnet18
from torchvision.transforms.functional import normalize, resize, to_pil_image

from torchcam.methods import SmoothGradCAMpp, LayerCAM
from torchcam.utils import overlay_mask

if __name__ == '__main__':
    # Instantiate your model here
    model = resnet18(pretrained=True).eval()

    img = read_image("test_img/ILSVRC2012_val_00030181.JPEG")

    # Preprocess it for your chosen model
    input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Preprocess your data and feed it to the model
    out = model(input_tensor.unsqueeze(0))

    # Retrieve the CAM from several layers at the same time
    cam_extractor = LayerCAM(model, ["layer2", "layer3", "layer4"])

    # Preprocess your data and feed it to the model
    out = model(input_tensor.unsqueeze(0))
    print(softmax(out, dim=1).max())

    cams = cam_extractor(out.squeeze(0).argmax().item(), out)

    # The raw

    _, axes = plt.subplots(1, len(cam_extractor.target_names))
    for idx, name, cam in zip(range(len(cam_extractor.target_names)), cam_extractor.target_names, cams):
        axes[idx].imshow(cam.squeeze(0).numpy())
        axes[idx].axis('off')
        axes[idx].set_title(name)
    plt.show()

    # Let's fuse them
    fused_cam = cam_extractor.fuse_cams(cams)
    # Plot the raw version
    plt.imshow(fused_cam.squeeze(0).numpy())
    plt.axis('off')
    plt.title(" + ".join(cam_extractor.target_names))
    plt.show()
    # Plot the overlayed version
    result = overlay_mask(to_pil_image(img), to_pil_image(fused_cam, mode='F'), alpha=0.5)
    plt.imshow(result)
    plt.axis('off')
    plt.title(" + ".join(cam_extractor.target_names))
    plt.show()

    # Once you're finished, clear the hooks on your model
    cam_extractor.remove_hooks()



