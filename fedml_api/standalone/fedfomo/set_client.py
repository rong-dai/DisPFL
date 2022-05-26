import copy
import logging
import math

import numpy as np
import torch

class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                 model_trainer):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        logging.info("self.local_sample_number = " + str(self.local_sample_number))

        self.args = args
        self.device = device
        self.model_trainer = model_trainer

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, w_per, w_global, masks,round):
        logging.info(sum([torch.count_nonzero(w_per[name]) for name in w_per if "meta" in name]))
        gradient = self.model_trainer.screen_gradients(self.local_training_data, w_per,  self.device)
        masks, num_remove = self.fire_mask(masks, w_per, round)
        masks = self.regrow_mask(masks,gradient, num_remove)
        # apply global model weights according to the new masks
        for name in masks:
            w_per[name] = w_global[name]*masks[name]
        self.model_trainer.set_model_params(w_per)
        self.model_trainer.set_masks(masks)
        self.model_trainer.set_id(self.client_idx)
        self.model_trainer.train(self.local_training_data, self.device, self.args, round)
        weights = self.model_trainer.get_model_params()
        # masks, weights = self.iterate_prune(masks,weights,round)
        update= {}
        for name in weights:
            update[name] = w_global[name]*masks[name] - weights[name].data
        # logging.info("masks{}".format(sum([ sum([ torch.sum(mask)  for mask in layer_masks]) for layer_masks in next_masks]) ))
        logging.info(sum([torch.count_nonzero(weights[name]) for name in weights if "meta" in name]))
        return masks, weights, update


    def fire_mask(self, masks, weights, round):
        drop_ratio = self.args.anneal_factor / 2 * (1 + np.cos((round * np.pi) / self.args.comm_round))
        new_masks = {}
        new_weights = {}
        num_remove = {}
        for name in masks:
            num_total = torch.numel(masks[name])
            num_zeros = num_total - torch.sum(masks[name])
            num_non_zeros= torch.sum(masks[name])
            num_remove[name] = math.ceil(drop_ratio * num_non_zeros)
            x, idx = torch.sort(torch.abs(weights[name].data.view(-1)))
            k = math.ceil(num_zeros + num_remove[name])
            threshold = x[k - 1].item()
            new_masks[name] = (torch.abs(weights[name].data) > threshold).float()
            new_weights[name] = new_masks[name] * weights[name]
        return new_masks, num_remove

    def regrow_mask(self,masks, gradient, num_remove):
        new_masks = copy.deepcopy(masks)
        for name in masks:
            temp = gradient[name] * (1 - new_masks[name])
            sort_temp, idx = torch.sort(torch.abs(temp.data.view(-1)), descending=True)
            new_masks[name].view(-1)[idx[:num_remove[name]]] = 1
        return new_masks

    def cosine_annealing(self, args, round):
        return args.anneal_factor / 2 * (1 + np.cos((round * np.pi) / args.comm_round))


    def local_test(self, w_per, mask_per, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        self.model_trainer.set_model_params(w_per)
        self.model_trainer.set_masks(mask_per)
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
