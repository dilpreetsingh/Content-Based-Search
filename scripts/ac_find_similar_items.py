import json
import fire

import numpy as np

from annoy import AnnoyIndex
from dataset import ImageDataset
# load file
# load sim matrix
# output

def construct_ann_index(metric, num_trees, features):
    feature_dims = features[0].shape[0]
    ann = AnnoyIndex(feature_dims, metric=metric)
    for index, feature in enumerate(features):
        ann.add_item(index, feature)

    ann.build(num_trees)
    return ann


def run(file, sim_matrix, data_path, annoy_distrance='angular', annoy_num_tree=50, output_path="./results"):
    lot_dict = dict()

    import csv
    reader = list(csv.reader(open(file, 'r')))
    cols = reader[0]
    for row in reader[1:]:
        lot = dict(zip(cols, row))
        lot_dict[lot['id']] = lot

    ds = ImageDataset(data_path)
    ann_indices = list(range(len(ds.image_paths)))
    data_indices = list(map(lambda x: x.split('/')[-1].split('.')[0], ds.image_paths))

    ann_to_data = dict(zip(ann_indices, data_indices))
    data_to_ann = dict(zip(data_indices, ann_indices))

    features = np.load(sim_matrix)
    ann = construct_ann_index(annoy_distrance, annoy_num_tree, features)

    arr_results = []
    for k in list(data_indices):
        if k in data_indices:
            ann_idx = data_to_ann[k]

            similar_images, distances = ann.get_nns_by_item(ann_idx, 11, include_distances=True)
            similar_images, distances = similar_images[1:], distances[1:]
            try:
                item = dict(
                    artwork=dict(lot_dict[k])
                )

                sims = list(map(lambda s: lot_dict[ann_to_data[s]].copy(), similar_images))
                for idx in range(len(sims)):
                    sims[idx]['score'] = 1 - float(distances[idx])
                item['sims'] = list(sorted(sims, key=lambda x: x['score'], reverse=True))
                arr_results.append(item)
            except:
                print('this one fails %d' % ann_idx)
    sim_name = sim_matrix.split('/')[-1]

    with open('%s/nn-from-%s.json' % (output_path, sim_name), 'w') as outfile:
        json.dump(arr_results, outfile)


if __name__ == '__main__':
    fire.Fire(run)
