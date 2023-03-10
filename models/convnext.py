from torchvision.models import convnext_small
from torchvision.models.convnext import LayerNorm2d
import torch.nn as nn
from models.attentionblocks import AttnCABfc

class ConvNextSmallAB(nn.Module):
    def __init__(self, in_planes=768, classes=5, k=5, modo='original'):
        super(ConvNextSmallAB, self).__init__()

        self.backbone = nn.Sequential(
            *list(convnext_small(pretrained=True).children())[:-2])
        
        self.attnblocks = AttnCABfc(in_planes, classes, k, modo)

    def forward(self, x):
        x = self.backbone(x)
        x = self.attnblocks(x)

        return x


def convNextSmallCustom(n_class):

    model = convnext_small(pretrained=True, progress=True)
    
    sequential_layers = nn.Sequential(
        LayerNorm2d((768,), eps=1e-06, elementwise_affine=True),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(768, 2048, bias=True),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(2048, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        nn.Linear(2048, n_class),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = sequential_layers

    return model