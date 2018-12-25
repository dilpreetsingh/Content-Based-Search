import torch.nn as nn
import torchvision.models as tvm
import torch.nn.functional as F

from torchsummary import summary


import utils


def get_model(name):
    if name is 'vgg16':
        return VGG16()
    elif name is 'resnet152':
        return Resnet152()
    elif name is 'squeezenet':
        return Squeezenet()
    elif name is 'densenet':
        return Densenet()
    else:
        raise SystemExit('Please specify a model name')


class Model:
    def verbose(self):
        summary(self.model, (3, 224, 224))

    def forward_pass(self, batch):
        return self.features.forward(batch).view(batch.size(0), -1)

class VGG16(Model):
    def __init__(self):
        super(VGG16, self).__init__()
        self.model = tvm.vgg16(pretrained=True)
        self.features = self.model.features.to(utils.torch_device())
        self.classifier = self.model.classifier[:4].to(utils.torch_device())

    def forward_pass(self, batch):

        feature_output = self.features.forward(batch)
        feature_output = feature_output.view(feature_output.size(0), -1)

        return self.classifier.forward(feature_output)


class Resnet152(Model):
    def __init__(self):
        super(Resnet152, self).__init__()

        self.model = tvm.resnet152(pretrained=True)

        self.features = nn.Sequential(*list(self.model.children())[:-1]).to(utils.torch_device())


class Densenet(Model):
    def __init__(self):
        super(Densenet, self).__init__()

        self.model = tvm.densenet201(pretrained=True)
        self.features = self.model.features.to(utils.torch_device())

    def forward_pass(self, batch):
        features = self.features.forward(batch)
        out = F.relu(features, inplace=True)
        return F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)


class Squeezenet(Model):
    def __init__(self):
        super(Squeezenet, self).__init__()

        self.model = tvm.squeezenet1_1(pretrained=True)

        self.features = nn.Sequential(*list(self.model.children())[:-3]).to(utils.torch_device())


