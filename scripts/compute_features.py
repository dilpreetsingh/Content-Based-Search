import time
import logging
import fire
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

import models
import utils
from dataset import ImageDataset

logging.getLogger().setLevel(logging.INFO)


def run(model_name, output_dir, dataname, data_dir='./data', batch_size=16, test_run=-1):
    data_path = '%s/%s' % (data_dir, dataname)
    logging.info('Load data from %s' % data_path)
    logging.info('Using model=%s' % model_name)

    ds = ImageDataset(data_path)
    model = models.get_model(model_name)

    data_loader = DataLoader(ds, batch_size=batch_size)

    features_list = []

    count = 0
    iterator = tqdm(data_loader)
    for batch in iterator:

        output = model.forward_pass(batch.to(utils.torch_device()))

        features_list.append(output.cpu().detach().numpy())

        if test_run != -1 and count > test_run:
            iterator.close()
            break

        count = count + 1

    features = np.vstack(features_list)
    logging.info(features.shape)

    output_path = '%s/%s-%s--%s' % (output_dir, model_name, dataname, time.strftime('%Y-%m-%d-%H-%M-%S'))
    np.save(output_path, features)
    logging.info('save data at %s' % output_path)

if __name__ == "__main__":
    fire.Fire(run)
