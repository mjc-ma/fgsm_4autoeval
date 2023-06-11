import torch.nn as nn


class LeNetSimCLR(nn.Module):
    def __init__(self, num_classes, out_dim=128):
        super(LeNetSimCLR, self).__init__()
        # cnn backbone
        self.conv1 = nn.Conv2d(3, 6, 5)  # ‘RGB’ three channels
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()

        # <fix>, classifier
        self.fc3 = nn.Linear(84, num_classes)
        self.relu5 = nn.ReLU()
        self.classify_head = nn.Sequential(
            self.fc3,
            self.relu5
        )

        # <fix>, contrastive learning head
        self.contrastive_head = nn.Sequential(
            nn.Linear(84, 84),  # projector
            nn.ReLU(),
            nn.Linear(84, out_dim),
        )

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        cla_out = self.classify_head(y)
        con_out = self.contrastive_head(y)

        return cla_out, con_out


# <fix>, refer to Meta-set-main,
class MetaLeNetSimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        self.fc_params = nn.Sequential(nn.Linear(50 * 4 * 4, 500), nn.ReLU(), nn.Dropout(p=0.5))
        self.classifier = nn.Linear(500, 10)
        self.contraster = nn.Sequential(
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 128)
        )
        self.__in_features = 500

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        return self.classifier(x), self.contraster(x)
        # y = self.classifier(x)
        # return x, y

    def output_num(self):
        return self.__in_features