import torch
import torch.nn as nn
from torchvision.models import resnet50

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


model_urls = dict(
    auc93='https://github.com/tobysuwindra/birdsimilarity/releases/download/BirdSimilarityModel/model_auc93.pth'
)


def load_state(arch, progress=True):
    state = load_state_dict_from_url(model_urls.get(arch), progress=progress)
    return state


def model_auc93(pretrained=True, progress=True):
    model = BirdNetModel()
    if pretrained:
        state = load_state('auc93', progress)
        model.load_state_dict(state['state_dict'])
    return model



class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class BirdNetModel(nn.Module):
    def __init__(self, pretrained=False):
        super(BirdNetModel, self).__init__()

        self.model = resnet50(pretrained)
        embedding_size = 128
        num_classes = 500
        self.cnn = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4)

        self.model.fc = nn.Sequential(
            Flatten(),
            nn.Linear(100352, embedding_size))

        self.model.classifier = nn.Linear(embedding_size, num_classes)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def freeze_fc(self):
        for param in self.model.fc.parameters():
            param.requires_grad = False

    def unfreeze_fc(self):
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def freeze_only(self, freeze):
        for name, child in self.model.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    def unfreeze_only(self, unfreeze):
        for name, child in self.model.named_children():
            if name in unfreeze:
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.cnn(x)
        x = self.model.fc(x)

        features = self.l2_norm(x)
        alpha = 10
        features = features * alpha
        return features

    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.model.classifier(features)
        return res