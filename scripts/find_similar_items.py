import json
import fire

import numpy as np

from annoy import AnnoyIndex
from dataset import ImageDataset

import config

PUBLIC_URL = 'https://s3.amazonaws.com/pchormai-public/artwork-similarity'


def remove_suffix(name):
    return name.split('--m')[0]


def construct_ann_index(metric, num_trees, features):
    feature_dims = features[0].shape[0]
    ann = AnnoyIndex(feature_dims, metric=metric)
    for index, feature in enumerate(features):
        ann.add_item(index, feature)

    ann.build(num_trees)
    return ann


def run(file, sim_matrix, data_path, annoy_distrance='angular', annoy_num_tree=50, output_path="./results"):
    lot_dict = dict()

    with open(file, 'r') as f:
        data = json.load(f)

    for row in data:
        lot_dict[str(row['ObjectID'])] = {
            'title': row['Title'],
            'dimensions': row['Dimensions'],
            'date': row['Date'],
            'artist_name': ','.join(row['Artist']),
            'medium': row['Medium'],
            'moma_image': row['ThumbnailURL']
        }

    ds = ImageDataset(data_path)
    ann_indices = list(range(len(ds.image_paths)))
    data_indices = list(map(lambda x: x.split('/')[-1].split('.')[0], ds.image_paths))

    ann_to_data = dict(zip(ann_indices, data_indices))
    data_to_ann = dict(zip(data_indices, ann_indices))

    features = np.load(sim_matrix)
    ann = construct_ann_index(annoy_distrance, annoy_num_tree, features)

    public_image_url = lambda k: '%s/%s/%s.jpeg' % (PUBLIC_URL, data_path, k)

    arr_results = []
    for k in list(data_indices):
        if k in data_indices and '--m' not in k:
            ann_idx = data_to_ann[k]

            similar_images, distances = ann.get_nns_by_item(ann_idx, 11, include_distances=True)
            similar_images, distances = similar_images[1:], distances[1:]
            # try:
            item = dict(
                artwork=dict(lot_dict[remove_suffix(k)])
            )

            item['artwork']['image'] = public_image_url(k)

            sims = list(map(lambda s: lot_dict[remove_suffix(ann_to_data[s])].copy(), similar_images))
            for idx in range(len(sims)):
                sims[idx]['score'] = 1 - float(distances[idx])
                cur_sim = sims[idx]
                cur_sim['score'] = 1 - float(distances[idx])

                data_idx = ann_to_data[similar_images[idx]]
                cur_sim['image'] = public_image_url(data_idx)

                if '--m' in data_idx:
                    cur_sim['manipulation_profile'] = config.MANIPULATE_MAPPING[data_idx.split('--')[-1]]

            item['sims'] = list(sorted(sims, key=lambda x: x['score'], reverse=True))
            arr_results.append(item)
            # except Exception as e:
            #     print('this one fails %d' % ann_idx)
            #     print(e)
    sim_name = sim_matrix.split('/')[-1]

    with open('%s/nn-from-%s.json' % (output_path, sim_name), 'w') as outfile:
        json.dump(arr_results, outfile)


if __name__ == '__main__':
    fire.Fire(run)
