import math
import torch
from torch import nn

# Models
## Original Model
class BasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, down: bool = False) -> None:
        super().__init__()
        self.down = down
        self.conv1 = (
            nn.Conv2d(inplanes, planes, 3, stride=2, padding=1, bias=False) if down
            else nn.Conv2d(inplanes, planes, 3, padding='same', bias=False)
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        if self.down:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride=2, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.down:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class BottleNeck(nn.Module):
    # unused, not confirm the correctness yet
    def __init__(self, inplanes: int, planes: int, outplanes: int, down: bool = False) -> None:
        super().__init__()
        self.down = down
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = (
            nn.Conv2d(planes, planes, 3, stride=2, padding=1, bias=False) if down
            else nn.Conv2d(planes, planes, 3, padding='same', bias=False)
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, outplanes, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        if self.down:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, 1, stride=2, bias=False),
                nn.BatchNorm2d(outplanes)
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.down:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class CIFAR_ResNet(nn.Module):
    def __init__(self, n: int = 3, num_classes: int = 100, p: float = 0.2) -> None:
        super().__init__()
        self.n = n
        self.inplanes = 16
        self.planes = self.inplanes
        def consruct_layers(self, down: bool = False):
            layers = []
            for i in range(self.n):
                if i == 0 and down == True:
                    self.inplanes = self.planes
                    self.planes *= 2
                    layers.append(BasicBlock(self.inplanes, self.planes, down=True))
                else:
                    layers.append(BasicBlock(self.planes, self.planes))
            return nn.Sequential(*layers)
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True)
        )
        self.layer1 = consruct_layers(self)
        self.layer2 = consruct_layers(self, down=True)
        self.layer3 = consruct_layers(self, down=True)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=p, inplace=True),
            nn.Linear(self.planes, num_classes)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)
    def adjust_dropout(self, p: float = 0.2):
        self.classifier[2] = nn.Dropout(p=p, inplace=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.classifier(x)
        return x