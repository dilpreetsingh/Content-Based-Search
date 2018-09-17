from annoy import AnnoyIndex
from fastai import conv_learner as learner
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import cv2
import glob
import numpy as np
import os

# --- Image Dataset ---

class ImageDataset(Dataset):
    
    def __init__(self, root_dir, tfms):
        # Look for jpegs in the directory
        self.image_paths = glob.glob(root_dir + '*.jpg')
        assert self.image_paths != 0, "No images found in {}".format(root_dir)

        self.image_names = [os.path.basename(path) for path in self.image_paths]
        self.tfms = tfms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        
        # Returns image in RGB format; each pixel ranges between 0.0 and 1.0
        image = learner.open_image(image_path)

        # Apply transforms to the image
        image = self.tfms(image)
        return image

# --- Forward hook for storing layer features ---

class LayerHook():

    def __init__(self, m):
        self.features = None
        self.hook = m.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        # Save the computed features
        self.features = output

    def close(self):
        self.hook.remove()

# --- Model setup ---

def setup_inceptionv4_model():
    model = learner.inceptionv4(True)
    # fast.ai intelligently detects if cuda is available, so  works for both cpu and gpu
    model = learner.to_gpu(model).eval()
    # Explore model that you choose to figure out what layer you want
    # print(learner.children(model))
    # summary(model, input_size=(3,224,224))
    
    feature_layer = learner.children(model)[-2][-1] # this is the `AdaptiveAvgPool2d-634` layer
    feature_hook = LayerHook(feature_layer)
    return model, feature_hook

# --- Compute Features ---

def compute_features(model, feature_hook, dataset):
    features_list = []
    data_loader = DataLoader(dataset, batch_size=16)
    
    # Compute features for all images
    for batch in tqdm(data_loader):
        model(learner.VV(batch)) # `VV` creates a pytorch variable (volatile=True)
        batch_features = feature_hook.features.clone().data.cpu().numpy()
        batch_features = batch_features.squeeze()
        features_list.append(batch_features)
    
    features = np.vstack(features_list)
    return features

# --- Construct approximate nearest neighbour index ---

def construct_ann_index(metric, num_trees, features):
    feature_dims = features[0].shape[0]
    ann = AnnoyIndex(feature_dims, metric=metric)
    for index, feature in enumerate(tqdm(features)):
        ann.add_item(index, feature)

    ann.build(num_trees)
    return ann

# --- Visualise transforms ---

def visualise_transforms(tfms, img_path):
    img = learner.open_image(img_path)
    cv2.imshow('original', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Last transform switches the channel order, so we omit that (using [:-1])
    for tfm in tfms.tfms[:-1]:
        img, y = tfm(img, None)
        print(tfm)
        cv2.imshow('tfm', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)

# Different types of cropping choices
CENTER_CROP = learner.CropType.CENTER
NO_CROP = learner.CropType.NO

# Setup model and image transforms
model, feature_hook = setup_inceptionv4_model()
_, tfms = learner.tfms_from_model(model.__class__, 224, crop_type=CENTER_CROP)

# visualise_transforms(tfms)

dataset = ImageDataset('data/art-collection/', tfms)
features = compute_features(model, feature_hook, dataset)
print(features.shape)

# Save the computed features
model = 'inceptionv4'
np.save('{}_features.npy'.format(model), np.array(features))

# Build the index
ann = construct_ann_index("angular", 25, features)
ann.save("{}_angular_25.ann".format(model))

# Perform a search
query_index = np.random.randint(0, ann.get_n_items())
closest_items = ann.get_nns_by_item(query_index, 5)
# Exclude the first item as it's simply the query image itself
closest_items = closest_items[1:]

print("Closest images for {}:".format(dataset.image_names[query_index]))
for item in closest_items:
    print(dataset.image_names[item])