import fire
import json

from multiprocessing import Pool
from functools import partial

import numpy as np
from urllib import  request
from tqdm import tqdm
import urllib.request


SEED = 71


def download(artwork, output_dir='./data/moma-artworks'):
    request.urlretrieve(artwork['ThumbnailURL'], '%s/%d.jpeg' % (output_dir, artwork['ObjectID']))


def run(artworks_json, num_artworks=100, parallel=5, output_dir='./data/moma-artworks'):
    with open(artworks_json) as f:
        data = json.load(f)

    artwork_with_thumbnails = list(filter(lambda x: x['ThumbnailURL'], data))
    total_artworks = len(artwork_with_thumbnails)

    print('total artworks %d (having %d with thumbnails)' % (len(data), total_artworks))
    print('We take %d artworks' % num_artworks)

    np.random.seed(SEED)

    selected_artworks = map(lambda idx: artwork_with_thumbnails[idx], np.random.choice(total_artworks, num_artworks))
    selected_artworks = list(selected_artworks)

    download_func = partial(download, output_dir=output_dir)

    with Pool(parallel) as pool:
        _ = list(tqdm(pool.imap(download_func, selected_artworks), total=len(selected_artworks)))


if __name__ == '__main__':
    fire.Fire(run)
