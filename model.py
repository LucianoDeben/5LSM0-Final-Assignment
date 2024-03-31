from torch import nn
from torchvision.models.segmentation import (DeepLabV3_ResNet101_Weights,
                                             deeplabv3_resnet101)
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead


class Model(nn.Module):
    def __init__(self, num_classes=20):
        super(Model, self).__init__()
        self.weights = DeepLabV3_ResNet101_Weights.DEFAULT
        self.model = deeplabv3_resnet101(weights=self.weights)
        self.model.classifier = DeepLabHead(2048, num_classes)
        self.model.aux_classifier = FCNHead(1024, num_classes)
        self.model.train()

    def forward(self, x):
        return self.model(x)['out']