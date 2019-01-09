import glob
import os


from fastai import vision
from torch.utils.data import Dataset

import utils


TRANSFORMATION_PARAMS = dict(
    size=224,
    padding_mode="zeros"
)


class ImageDataset(Dataset):
    def __init__(self, root_dir):
        # Look for jpegs in the directory
        self.image_paths = sorted(glob.glob(root_dir + '/*.jpeg'))
        assert self.image_paths != 0, "No images found in {}".format(root_dir)

        self.image_names = [os.path.basename(path) for path in self.image_paths]
        _, self.tfms = vision.get_transforms()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index, with_transformation=True):
        image_path = self.image_paths[index]

        # Returns image in RGB format; each pixel ranges between 0.0 and 1.0
        image = vision.open_image(image_path)

        if not with_transformation:
            return image.px

        # Apply transforms to the image
        return self.transform(self.tfms, image).px

    def transform(self, tfms, img):
        # return vision.apply_tfms(tfms, img, **TRANSFORMATION_PARAMS)
        return img.apply_tfms(tfms, **TRANSFORMATION_PARAMS)

    def get(self, index, **kwargs):
        return self.__getitem__(index, **kwargs)