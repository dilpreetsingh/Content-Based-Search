import models
import torch
import torch.optim as optim

from datetime import datetime

##
import torchvision
import torchvision.transforms as transforms

import argparse
import os
import time

## https://github.com/pytorch/examples/blob/master/vae/main.py
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--train_dir', type=str, default='./data-train', metavar='N',
                    help='train-data')
parser.add_argument('--test_dir', type=str, default='./data-test', metavar='N',
                    help='test-data')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")


base_transform = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()])
noisy_transform = transforms.RandomChoice([
    transforms.ColorJitter(brightness=0.5),
    transforms.ColorJitter(contrast=0.5),
    transforms.ColorJitter(saturation=0.2),
])


class NoisyImageDataset(object):
    def __init__(self, root_dir, transform=None, noisy_transform=None):
        # Look for jpegs in the directory
        self.image_paths = sorted(glob.glob(root_dir + '*.jpeg'))
        assert self.image_paths != 0, "No images found in {}".format(root_dir)

        self.image_names = [os.path.basename(path) for path in self.image_paths]
        self.transform = transform

        self.noisy_transofrm = transforms.Compose([noisy_transform, self.transform])
        self.total_data = len(self.image_paths)

    def __len__(self):
        return self.total_data

    def __getitem__(self, index):
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
trainloader = torch.utils.data.DataLoader(noisy_dataset, batch_size=32, shuffle=True, num_workers=2)

test_noisy_dataset = NoisyImageDataset(args.test_dir, transform=base_transform, noisy_transform=noisy_transform)
testloader = torch.utils.data.DataLoader(test_noisy_dataset, batch_size=4, shuffle=False, num_workers=2)

ooo
## Prepare model
experiment_name = 'model-%s' % datetime.now().strftime('%Y-%m-%d--%H-%M')




optimizer = optim.Adam(vae.parameters(), lr=args.lr)

for epoch in range(2):
    for i, data in enumerate(trainloader, 0):
        x, labels = data

        x = x.to(device)

        optimizer.zero_grad()

        x_hat, z, z_mu, z_logvar = vae(x)

        loss = vae.loss(x_hat, x, z_mu, z_logvar)\

        loss.backward()

        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 150 == 149:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 149))
            running_loss = 0.0

# todo: artifact save

output_dir = './model-results/%s' % experiment_name
os.mkdir(output_dir)

prefix_dir = lambda x: '%s/%s' % (output_dir, x)

torch.save(vae.state_dict(), prefix_dir('model.pth'))
