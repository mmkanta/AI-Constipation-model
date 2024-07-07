import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from .cbam import CBAM
from torchvision.models import efficientnet_v2_m

class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        model = torchvision.models.efficientnet_v2_m(weights='IMAGENET1K_V1')
        model.classifier = nn.Sequential(nn.Flatten(),
                                         nn.Linear(1280, 2),
                                         nn.Softmax(dim=1))
        self.features = model.features
        self.cbam = CBAM(gate_channels=1280)
        self.avgpool = model.avgpool #nn.AdaptiveMaxPool2d(output_size=1)
        self.classifier = model.classifier
        
    def forward(self, x):
        x = self.features(x)
        x_cbam = self.cbam(x)
        x = self.avgpool(x_cbam)
        output = self.classifier(x)
        return output
