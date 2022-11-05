from torch import nn
import torchvision
import torch

def get_backbone_net(backbone, out_dim, is_pretrained):
    if backbone == 'resnet18':
        backbone_net = torchvision.models.resnet18(num_classes = out_dim, pretrained=is_pretrained)
        backbone_net.conv1.weight.data = backbone_net.conv1.weight.data.sum(1).unsqueeze(1)
        backbone_net.conv1.in_channels = 1
        setattr(backbone_net, 'out_dim', out_dim)
    elif backbone == 'resnet34':
        backbone_net = torchvision.models.resnet34(num_classes = out_dim, pretrained=is_pretrained)
        backbone_net.conv1.weight.data = backbone_net.conv1.weight.data.sum(1).unsqueeze(1)
        backbone_net.conv1.in_channels = 1
        setattr(backbone_net, 'out_dim', out_dim)
    elif backbone == 'resnet50':
        backbone_net = torchvision.models.resnet50(num_classes = out_dim, pretrained=is_pretrained)
        backbone_net.conv1.weight.data = backbone_net.conv1.weight.data.sum(1).unsqueeze(1)
        backbone_net.conv1.in_channels = 1
        setattr(backbone_net, 'out_dim', out_dim)
    else:
        raise NotImplementedError
    return backbone_net



class FourSingleDimOutNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.feature_extractor = kwargs['backbone_net']
    def forward(self, batch):
        out_L_CC = self.feature_extractor(batch['L_CC'])
        out_R_CC = self.feature_extractor(batch['R_CC'])
        out_L_MLO = self.feature_extractor(batch['L_MLO'])
        out_R_MLO = self.feature_extractor(batch['R_MLO'])
        return out_L_CC, out_R_CC, out_L_MLO, out_R_MLO


class FourViewModuleSingleDim(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        self.feature_extractor = kwargs['backbone_net']
        self.classifier = nn.Linear(4 * self.feature_extractor.out_dim, num_classes) # TODO: remove hard coding for num of classes
    def forward(self, batch):
        out_L_CC = self.feature_extractor(batch['L_CC'])
        out_R_CC = self.feature_extractor(batch['R_CC'])
        out_L_MLO = self.feature_extractor(batch['L_MLO'])
        out_R_MLO = self.feature_extractor(batch['R_MLO'])
        out_all = torch.cat([out_L_CC, out_R_CC, out_L_MLO, out_R_MLO], dim=1)
        out = self.classifier(out_all)
        return out


# TODO: add batch norm
class FourViewModuleConv(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=4*64, out_channels=4*64, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=4*64, out_channels=4*64, kernel_size=3),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            nn.Dropout(0.5),
            nn.Linear(4*64, num_classes)
        )
    def forward(self, batch):
        out_L_CC = self.feature_extractor(batch['L_CC'])
        out_R_CC = self.feature_extractor(batch['R_CC'])
        out_L_MLO = self.feature_extractor(batch['L_MLO'])
        out_R_MLO = self.feature_extractor(batch['R_MLO'])
        out_all = torch.cat([out_L_CC, out_R_CC, out_L_MLO, out_R_MLO], dim=1)
        out = self.classifier(out_all)
        return out