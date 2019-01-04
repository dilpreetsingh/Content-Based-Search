import models
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

import torch.nn as nn
from functools import reduce
import torch.optim as optim
import utils


from datetime import datetime

from PIL import Image


import torchvision.transforms as transforms

import argparse
import os
import glob

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--train_dir', type=str, default='./data-train', metavar='N',
                    help='train-data')
parser.add_argument('--test_dir', type=str, default='./data-test', metavar='N',
                    help='test-data')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if args.cuda else "cpu")

base_transform = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()])
noisy_transform = transforms.RandomChoice([
    transforms.ColorJitter(brightness=0.5),
    transforms.ColorJitter(contrast=0.5),
    transforms.ColorJitter(saturation=0.2),
])


class NoisyImageDataset(object):
    def __init__(self, root_dir, transform=None, noisy_transform=None):
        self.image_paths = sorted(glob.glob(root_dir + '/*.jpeg'))
        assert self.image_paths != 0, "No images found in {}".format(root_dir)

        self.image_names = [os.path.basename(path) for path in self.image_paths]
        self.transform = transform

        self.noisy_transofrm = transforms.Compose([noisy_transform, self.transform])
        self.total_data = len(self.image_paths)

    def __len__(self):
        return self.total_data

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image, x = self._load_and_transform(index)
        pos_x = self.noisy_transofrm(image)
        _, neg_x = self._load_and_transform(np.random.choice(self.total_data))

        return x, pos_x, neg_x

    def _load_and_transform(self, index):
        image_path = self.image_paths[index]

        # Returns image in RGB format; each pixel ranges between 0.0 and 1.0
        image = Image.open(image_path).convert('RGB')
        x = self.transform(image)
        return image, x

    def get(self, index):
        return self.__getitem__(index)


noisy_dataset = NoisyImageDataset(args.train_dir, transform=base_transform, noisy_transform=noisy_transform)
trainloader = torch.utils.data.DataLoader(noisy_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

test_noisy_dataset = NoisyImageDataset(args.test_dir, transform=base_transform, noisy_transform=noisy_transform)
testloader = torch.utils.data.DataLoader(test_noisy_dataset, batch_size=4, shuffle=False, num_workers=1)


class Model(nn.Module):
    def __init__(self, verbose=False):
        super(Model, self).__init__()
        self.vgg = models.VGG16()
        self.encoder = nn.Sequential(
            nn.Linear(4096, 1024)
        )

    def forward(self, x):
        x = self.vgg.forward_pass(x)
        z = self.encoder(x)

        norm = z.norm(p=2, dim=1, keepdim=True)
        normed_z = z.div(norm)

        return normed_z

    def parameters(self):
        return list(self.encoder.parameters())


running_loss = []

model = Model().to(device)
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(args.epochs):
    for i, data in enumerate(trainloader, 0):

        optimizer.zero_grad()

        z = list(map(lambda x: model(x.to(device)), data))

        loss = triplet_loss(*z)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss.append(loss.item())
        if i % 10 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, np.mean(running_loss)))
            running_loss = []

print('Computing features for testing set')

embeddings = np.zeros((len(testloader.dataset), 1024))
test_loss = []
with torch.no_grad():
    for i, data in enumerate(testloader):
        x = data[0]
        z = model(x.to(device))

        embeddings[i*4:(i+1)*4] = z


sim_matrix = embeddings.dot(embeddings.T)
nearest_neighbors = np.argsort(-sim_matrix, axis=1)


utils.get_stats(testloader.dataset.image_paths, nearest_neighbors, k=3)

# todo: save to file

# compute feature
