import fire
import json

from multiprocessing import Pool


import numpy as np
from urllib import  request
import urllib.request


SEED = 71

def download(artwork, output_dir='./data/moma-artworks'):
    request.urlretrieve(artwork['ThumbnailURL'], '%s/%d.jpg' % (output_dir, artwork['ObjectID']))

def run(artworks_json, num_artworks=100, parallel=5):
    # select ... outout num_artworks

    with open(artworks_json) as f:
        data = json.load(f)

    artwork_with_thumbnails = list(filter(lambda x: x['ThumbnailURL'], data))
    total_artworks = len(artwork_with_thumbnails)

    print('total artworks %d (having %d with thumbnails)' % (len(data), total_artworks))
    print('We take %d artworks' % num_artworks)

    np.random.seed(SEED)

    selected_artworks = map(lambda idx: artwork_with_thumbnails[idx], np.random.choice(total_artworks, num_artworks))

    # for artwork in selected_artworks:
    #     print(artwork)
    #     download(artwork)

    with Pool(parallel) as pool:
        pool.map(download, selected_artworks)


if __name__ == '__main__':
    fire.Fire(run)
