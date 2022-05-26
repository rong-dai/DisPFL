import copy
import logging
import numpy as np
import torch
import pdb

def cosine_annealing(args, round):
    return args.anneal_factor / 2 * (1 + np.cos((round * np.pi) / args.comm_round))

def model_difference(model_a, model_b):
    a = sum([torch.sum(torch.square(model_a[name] - model_b[name])) for name in model_a])
    return a

def hamming_distance(mask_a, mask_b):
    dis = 0; total = 0
    for key in mask_a:
        dis += torch.sum(mask_a[key].int() ^ mask_b[key].int())
        total += mask_a[key].numel()
    return dis, total
