import quantus
from quantus.helpers.model.models import LeNet
import torch
import torchvision
from torchvision import transforms
import numpy as np
import captum
from captum.attr import Saliency, IntegratedGradients



# Enable GPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load a pre-trained LeNet classification model (architecture at quantus/helpers/models).
model = LeNet()
if device.type == "cpu":
    model.load_state_dict(torch.load("tests/assets/mnist", map_location=torch.device('cpu')))
else:
    model.load_state_dict(torch.load("tests/assets/mnist"))

# Load datasets and make loaders.
test_set = torchvision.datasets.MNIST(root='./data', download=True,
                                      transform=transforms.Compose([transforms.ToTensor()]))
test_loader = torch.utils.data.DataLoader(test_set, batch_size=24)

# Load a batch of inputs and outputs to use for XAI evaluation.
x_batch, y_batch = iter(test_loader).__next__()
x_batch, y_batch = torch.from_numpy(x_batch.cpu().numpy()),torch.from_numpy(y_batch.cpu().numpy())
print(x_batch.shape)



# Generate Integrated Gradients attributions of the first batch of the test set.
a_batch_saliency = Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1).cpu().numpy()
a_batch_intgrad = IntegratedGradients(model).attribute(inputs=x_batch, target=y_batch, baselines=torch.zeros_like(x_batch)).sum(axis=1).cpu().numpy()

# Save x_batch and y_batch as numpy arrays that will be used to call metric instances.
x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

# Quick assert.
assert [isinstance(obj, np.ndarray) for obj in [x_batch, y_batch, a_batch_saliency, a_batch_intgrad]]

metric = quantus.MaxSensitivity()

scores = metric(
    model=model,
    x_batch=x_batch,
    y_batch=y_batch,
    device=device,
    explain_func=quantus.explain,
    explain_func_kwargs={"method": "Saliency"}
)

print(len(scores))
print(scores)