import torch.nn as nn
import torchvision.models as tvm

from torchsummary import summary


def get_model(name):
    if name is 'vgg16':
        return VGG16()
    elif name is 'resnet152':
        return Resnet152()
    else:
        raise SystemExit('Please specify a model name')


class Model:
    def verbose(self):
        summary(self.model, (3, 224, 224))

    def forward_pass(self, batch):
        raise SystemExit('No Implementation')


class VGG16(Model):
    def __init__(self):
        super(VGG, self).__init__()
        self.model = tvm.vgg16(pretrained=True)

    def forward_pass(self, batch):
        layer_relu_36 = self.model.classifier[:4]

        feature_output = self.model.features.forward(batch)
        feature_output = feature_output.view(feature_output.size(0), -1)

        return layer_relu_36.forward(feature_output)


class Resnet152(Model):
    def __init__(self):
        super(Resnet152, self).__init__()

        self.model = tvm.resnet152(pretrained=True)

        self.features = nn.Sequential(*list(self.model.children())[:-1])

    def forward_pass(self, batch):
        return self.features.forward(batch).view(batch.size(0), -1)
