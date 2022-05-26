import copy
import logging

import numpy as np
import torch
from scipy.spatial import distance


def fake_prune( each_prune_ratio, param_dict, mask):
    '''
    This function derives the new pruning mask, it put 0 for the weights under the given percentile

    :param percent: pruning percent
    :param model: a pytorch model
    :param mask: the pruning mask

    :return mask: updated pruning mask
    '''
    # Calculate percentile value
    new_masks = copy.deepcopy(mask)
    for name in param_dict:
        if "weight" in name and "bn" not in name:
            tensor = param_dict[name]
            alive = tensor[np.nonzero((tensor * mask[name]).numpy())]  # flattened array of nonzero values
            # logging.info("alive num: {} name:{}".format(len(alive),name))
            percentile_value = np.percentile(abs(alive), each_prune_ratio*100)
            new_mask = np.where(abs(tensor) < percentile_value, 0, mask[name])
            # Apply mask
            new_masks[name] = torch.from_numpy(new_mask)
    return new_masks


def real_prune(param_dict, mask):
    '''
    This function applies the derived mask. It zeros the weights needed to be pruned based on the updated mask

    :param model: a pytorch model
    :param mask: pruning mask

    :return state_dict: updated (pruned) model state_dict
    '''
    new_param_dict = copy.deepcopy(param_dict)
    for name in param_dict:
        param = param_dict[name]
        tensor = param.data.cpu()
        weight_dev = param.device
        # Apply new weight and mask
        new_param_dict[name] = tensor * mask[name].to(weight_dev)
    return new_param_dict


def dist_masks(m1, m2):
    '''
    Calculates hamming distance of two pruning masks. It averages the hamming distance of all layers and returns it

    :param m1: pruning mask 1
    :param m2: pruning mask 2

    :return average hamming distance of two pruning masks:
    '''
    temp_dist = []
    for name in m1:
        # 1 - float(m1[step].reshape([-1]) == m2[step].reshape([-1])) / len(m2[step].reshape([-1]))
        temp_dist.append(distance.hamming(m1[name].reshape([-1]), m2[name].reshape([-1])))
    dist = np.mean(temp_dist)
    return dist


def print_pruning(param_dict, is_print=False):
    '''
    This function prints the pruning percentage and status of a given model

    :param model: a pytorch model

    :return pruning percentage, number of remaining weights:
    '''
    nonzero = 0
    total = 0
    for name in param_dict:
        tensor = param_dict[name]
        nz_count = torch.count_nonzero(tensor)
        total_params = torch.numel(tensor)
        nonzero += nz_count
        total += total_params
        # print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total},'
        #       f'Compression rate : {total / nonzero:10.2f}x  ({100 * (total - nonzero) / total:3.2f}% pruned)')
    return  nonzero/ total, nonzero
