import torch.nn as nn
import torchvision.models as tvm
import torch.nn.functional as F
import torch

from torchsummary import summary


import torchx
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

        utils.disable_gradients(self.features.parameters())
        utils.disable_gradients(self.classifier.parameters())

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


class VAE(nn.Module):
    def __init__(self, latent_dims=2, verbose=False):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            torchx.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(True),
            nn.Linear(120, 2 * latent_dims),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, 120),
            nn.ReLU(True),
            nn.Linear(120, 16 * 4 * 4),
            nn.ReLU(True),
            torchx.Reshape([16, 4, 4]),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(16, 6, 5),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(6, 1, 5),
            nn.Sigmoid()
        )

        self.loss = torchx.vae_loss

    def forward(self, x):
        h = self.encoder(x)
        z_mu, z_logvar = torch.chunk(h, 2, dim=1)
        z_mu = z_mu.view(z_mu.size(0), -1)
        z_logvar = z_logvar.view(z_logvar.size(0), -1)

        z_logvar = nn.Softplus()(z_logvar)

        z = self.sampling(z_mu, z_logvar)

        x_hat = self.decoder(z)
        return x_hat, z, z_mu, z_logvar

    def sampling(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

