import torch
import torch.nn as nn


class HomeworkCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            32, 64,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=False,
        )
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(
            64, 128,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(
            128, 256,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(
            256, 256,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv2d(
            256, 512,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.relu6 = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512, 256)
        self.head_relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 100)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool(x)

        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.head_relu(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = HomeworkCNN().eval()
    with torch.inference_mode():
        output = model(torch.randn(2, 3, 224, 224))
    assert output.shape == (2, 100)
    print(output.shape)
