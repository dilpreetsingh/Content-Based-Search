import torch
import numpy as np

def torch_device():
    if torch.cuda.device_count() > 0:
        return 'cuda:0'
    else:
        return 'cpu'


def get_id(path):
    return path.split('/')[-1].split('.')[0]


def get_stats(file_paths, nearest_neighbors, k=3, max_rel_items=5, epsilon=np.finfo(float).eps):
    num_relevant_items = []
    for i, path in enumerate(file_paths):
        if '--m' in path:
            continue

        id_prefix = '%s--' % get_id(path)
        sims_path = map(lambda x: file_paths[x], nearest_neighbors[i, 1:(k + 1)])
        sim_ids = list(filter(lambda s: id_prefix in get_id(s), sims_path))

        num_relevant_items.append(float(len(sim_ids)))
    num_relevant_items = np.array(num_relevant_items)
    print('computing stats from %d items' % num_relevant_items.shape[0])
    precision = num_relevant_items / k
    recall = num_relevant_items / max_rel_items
    f1 = 2 * ((precision * recall) / (precision + recall + epsilon))

    print('Precision %.4f +/- %.4f' % (np.mean(precision), np.std(precision)))
    print('Recall    %.4f +/- %.4f' % (np.mean(recall), np.std(recall)))
    print('f1    %.4f +/- %.4f' % (np.mean(f1), np.std(f1)))


def disable_gradients(parameters):
    for p in parameters:
        p.require_grads = False
