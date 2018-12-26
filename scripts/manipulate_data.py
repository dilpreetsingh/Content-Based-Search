import glob
import fire
import numpy as np

from PIL import Image, ImageFilter, ImageEnhance

from tqdm import tqdm

from multiprocessing import Pool


def manipulate(img_path):
    img = Image.open(img_path)
    dims = img.size

    path = img_path.split('.jpeg')

    manipulated_imgs = [
        img.convert("L"),
        img.resize((np.array(dims)*0.8).astype(int)),
        img.filter(ImageFilter.SHARPEN),
        img.filter(ImageFilter.EDGE_ENHANCE_MORE),
        ImageEnhance.Color(img).enhance(0.5)
    ]

    for idx, mimg in enumerate(manipulated_imgs):
        mimg.save('%s--m%s.jpeg' % (path[0], idx+1))


def run(data_dir, no_parallel=5):
    images = sorted(glob.glob(data_dir + '/*.jpeg'))
    images = list(filter(lambda f: '--m' not in f, images))

    print('We have %d images' % (len(images)))

    with Pool(no_parallel) as pool:
        _ = list(tqdm(pool.imap(manipulate, images), total=len(images)))


if __name__ == '__main__':
    fire.Fire(run)