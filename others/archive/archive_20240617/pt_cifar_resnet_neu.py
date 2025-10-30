import torch
from torchvision import datasets, transforms
# should be modified

# Data
def load_data(
    root: str,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int = 2,
    num_ops: int = 2,
    magnitude: int = 9,
    PIN_MEMORY_FLAG: bool = True,
    PERSISTENT_WORKERS_FLAG: bool = True,
):
    '''
    # Conventional Transform
    ## For cifar-10
    mean = torch.tensor([125.3, 123.0, 113.9]) / 255
    std = torch.tensor([63.0, 62.1, 66.7]) / 255
    ## For cifar-100
    mean = torch.tensor([129.3, 124.1, 112.4]) / 255
    std = torch.tensor([68.2, 65.4, 70.4]) / 255
    ## Transform Step
    transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='constant'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    '''

    transform = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=num_ops, magnitude=magnitude),
            transforms.ToTensor(),
        ]),
        'eval': transforms.Compose([
            transforms.ToTensor(),
        ])
    }

    dataset = {
        'train': datasets.CIFAR100(
            root=root, train=True, download=True, transform=transform['train']
        ),
        'test': datasets.CIFAR100(
            root=root, train=False, download=True, transform=transform['eval']
        )
    }

    dataloader = {
        'train': torch.utils.data.DataLoader(
            dataset['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=PIN_MEMORY_FLAG,
            persistent_workers=PERSISTENT_WORKERS_FLAG
        ),
        'test': torch.utils.data.DataLoader(
            dataset['test'],
            batch_size=eval_batch_size,
            num_workers=num_workers,
            pin_memory=PIN_MEMORY_FLAG,
            persistent_workers=PERSISTENT_WORKERS_FLAG
        )
    }
    
    return dataset, dataloader


# Models
## Original Model
class BasicBlock(torch.nn.Module):
    def __init__(self, inplanes: int, planes: int, down: bool = False) -> None:
        super().__init__()
        self.down = down
        self.conv1 = (
            torch.nn.Conv2d(inplanes, planes, 3, stride=2, padding=1, bias=False) if down
            else torch.nn.Conv2d(inplanes, planes, 3, padding='same', bias=False)
        )
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, 3, padding='same', bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        if self.down:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(inplanes, planes, 1, stride=2, bias=False),
                torch.nn.BatchNorm2d(planes)
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

class BottleNeck(torch.nn.Module):
    # unused, not confirm the correctness yet
    def __init__(self, inplanes: int, planes: int, outplanes: int, down: bool = False) -> None:
        super().__init__()
        self.down = down
        self.conv1 = torch.nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = (
            torch.nn.Conv2d(planes, planes, 3, stride=2, padding=1, bias=False) if down
            else torch.nn.Conv2d(planes, planes, 3, padding='same', bias=False)
        )
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv3 = torch.nn.Conv2d(planes, outplanes, 1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(outplanes)
        self.relu = torch.nn.ReLU(inplace=True)
        if self.down:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(inplanes, outplanes, 1, stride=2, bias=False),
                torch.nn.BatchNorm2d(outplanes)
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

class CIFAR_ResNet(torch.nn.Module):
    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        self.inplanes = 16
        self.planes = self.inplanes
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(3, self.inplanes, kernel_size=3, padding='same', bias=False),
            torch.nn.BatchNorm2d(self.inplanes),
            torch.nn.ReLU(inplace=True)
        )
        self.layer1 = torch.nn.Sequential(
            BasicBlock(self.inplanes, self.planes),
            BasicBlock(self.planes, self.planes),
            BasicBlock(self.planes, self.planes),
        )
        self.inplanes = self.planes
        self.planes *= 2
        self.layer2 = torch.nn.Sequential(
            BasicBlock(self.inplanes, self.planes, down=True),
            BasicBlock(self.planes, self.planes),
            BasicBlock(self.planes, self.planes),
        )
        self.inplanes = self.planes
        self.planes *= 2
        self.layer3 = torch.nn.Sequential(
            BasicBlock(self.inplanes, self.planes, down=True),
            BasicBlock(self.planes, self.planes),
            BasicBlock(self.planes, self.planes),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(self.planes, num_classes)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.classifier(x)
        return x

## Lazy Model
class LazyBasicBlock(torch.nn.Module):
    def __init__(self, planes: int, down: bool = False) -> None:
        super().__init__()
        self.down = down
        self.conv1 = (
            torch.nn.LazyConv2d(planes, 3, stride=2, padding=1, bias=False) if down
            else torch.nn.LazyConv2d(planes, 3, padding='same', bias=False)
        )
        self.bn1 = torch.nn.LazyBatchNorm2d()
        self.conv2 = torch.nn.LazyConv2d(planes, 3, padding='same', bias=False)
        self.bn2 = torch.nn.LazyBatchNorm2d()
        self.relu = torch.nn.ReLU(inplace=True)
        if self.down:
            self.downsample = torch.nn.Sequential(
                torch.nn.LazyConv2d(planes, 1, stride=2, bias=False),
                torch.nn.LazyBatchNorm2d()
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

class LazyBottleNeck(torch.nn.Module):
    def __init__(self, planes: int, outplanes: int, down: bool = False) -> None:
        super().__init__()
        self.down = down
        self.conv1 = torch.nn.LazyConv2d(planes, 1, bias=False)
        self.bn1 = torch.nn.LazyBatchNorm2d()
        self.conv2 = (
            torch.nn.LazyConv2d(planes, 3, stride=2, padding=1, bias=False) if down
            else torch.nn.LazyConv2d(planes, 3, padding='same', bias=False)
        )
        self.bn2 = torch.nn.LazyBatchNorm2d()
        self.conv3 = torch.nn.LazyConv2d(outplanes, 1, bias=False)
        self.bn3 = torch.nn.LazyBatchNorm2d()
        self.relu = torch.nn.ReLU(inplace=True)
        if self.down:
            self.downsample = torch.nn.Sequential(
                torch.nn.LazyConv2d(outplanes, 1, stride=2, bias=False),
                torch.nn.LazyBatchNorm2d()
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

class Lazy_CIFAR_ResNet(torch.nn.Module):
    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        self.planes = 16
        self.stem = torch.nn.Sequential(
            torch.nn.LazyConv2d(self.planes, kernel_size=3, padding='same', bias=False),
            torch.nn.LazyBatchNorm2d(),
            torch.nn.ReLU(inplace=True)
        )
        self.layer1 = torch.nn.Sequential(
            LazyBasicBlock(self.planes),
            LazyBasicBlock(self.planes),
            LazyBasicBlock(self.planes),
        )
        self.planes *= 2
        self.layer2 = torch.nn.Sequential(
            LazyBasicBlock(self.planes, down=True),
            LazyBasicBlock(self.planes),
            LazyBasicBlock(self.planes),
        )
        self.planes *= 2
        self.layer3 = torch.nn.Sequential(
            LazyBasicBlock(self.planes, down=True),
            LazyBasicBlock(self.planes),
            LazyBasicBlock(self.planes),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.LazyLinear(num_classes)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.classifier(x)
        return x
