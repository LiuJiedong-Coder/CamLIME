# 代码参考：https://opensource.salesforce.com/OmniXAI/latest/tutorials/vision/cem_torch.html

# This default renderer is used for sphinx docs only. Please delete this cell in IPython.
import plotly.io as pio
pio.renderers.default = "png"

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from omnixai.data.image import Image
from omnixai.explainers.vision import ContrastiveExplainer

class InputData(Dataset):

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


class MNISTNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x


if  __name__ == "__main__":
    # Load the training and test datasets
    train_data = torchvision.datasets.MNIST(root='../data', train=True, download=True)
    test_data = torchvision.datasets.MNIST(root='../data', train=False, download=True)
    train_data.data = train_data.data.numpy()
    test_data.data = test_data.data.numpy()

    class_names = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    # Use `Image` objects to represent the training and test datasets
    x_train, y_train = Image(train_data.data, batched=True), train_data.targets
    x_test, y_test = Image(test_data.data, batched=True), test_data.targets

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Build the CNN model
    model = MNISTNet().to(device)
    # The preprocessing function
    transform = transforms.Compose([transforms.ToTensor()])
    preprocess = lambda ims: torch.stack([transform(im.to_pil()) for im in ims])

    learning_rate = 1e-3
    batch_size = 128
    num_epochs = 10

    train_loader = DataLoader(
        dataset=InputData(preprocess(x_train), y_train),
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=InputData(preprocess(x_test), y_test),
        batch_size=batch_size,
        shuffle=False
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            loss = loss_func(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    correct_pred = {name: 0 for name in class_names}
    total_pred = {name: 0 for name in class_names}

    model.eval()
    for x, y in test_loader:
        images, labels = x.to(device), y.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[class_names[label]] += 1
            total_pred[class_names[label]] += 1

    for name, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[name]
        print("Accuracy for class {} is: {:.1f} %".format(name, accuracy))


    test = preprocess(x_test[4])

    explainer = ContrastiveExplainer(
        model=model,
        preprocess_function=preprocess
    )

    explanations = explainer.explain(x_test[4])
    explanations.plot()
    plt.show()