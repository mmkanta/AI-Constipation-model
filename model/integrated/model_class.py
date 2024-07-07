import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from ..image.cbam import CBAM

class Fully_Connected_Model(nn.Module):
    def __init__(self):
        super(Fully_Connected_Model, self).__init__()
        self.Dense_tabular = nn.Linear(15, 30)
        self.Auxiliary_tabular = nn.Linear(30, 1)
        self.Auxiliary_images = nn.Linear(30, 1)
        self.MHA = nn.MultiheadAttention(embed_dim=30, num_heads=3)
        self.linear1 = nn.Linear(120, 60)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(60, 1)

    def forward(self, image, tabular):
        ## Tabular
        tabular = F.relu(self.Dense_tabular(tabular))
        outputs_tabular = F.sigmoid(self.Auxiliary_tabular(tabular))
        ## Images
        outputs_images = F.sigmoid(self.Auxiliary_images(image))
        ## Fusion
        attn_output_1, _ = self.MHA(image, tabular, tabular)
        attn_output_2, _ = self.MHA(tabular, image, image)
        attn_output_1 = torch.add(tabular, attn_output_1)
        attn_output_2 = torch.add(image, attn_output_2)
        x = torch.cat((tabular, attn_output_1, attn_output_2, image), dim=1)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        outputs_fusion = F.sigmoid(x)

        return outputs_fusion
    
model = torchvision.models.efficientnet_v2_m(weights='IMAGENET1K_V1')
model.classifier = nn.Sequential(nn.Flatten(),
                                 nn.Linear(1280, 1280),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(1280, 30),
                                 nn.ReLU(inplace=True))

class Image_Model(nn.Module):
    def __init__(self, model):
        super(Image_Model, self).__init__()
        self.features = model.features
        self.cbam = CBAM(gate_channels=1280)
        self.avgpool = model.avgpool #nn.AdaptiveMaxPool2d(output_size=1)
        self.classifier = model.classifier
        
    def forward(self, x):
        x = self.features(x)
        x_cbam = self.cbam(x)
        #x = torch.add(x, x_cbam)
        x = self.avgpool(x_cbam)
        output = self.classifier(x)
        #output = self.Sigmoid(x)
    
        return output

class IntegratedModel(nn.Module):
    def __init__(self):
        super(IntegratedModel, self).__init__()
        self.image_model = Image_Model(model)
        self.fully_model = Fully_Connected_Model()
        
    def forward(self, image, tabular):
        x = self.image_model(image)
        outputs_fusion = self.fully_model(x, tabular)
    
        return outputs_fusion