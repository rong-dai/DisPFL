import copy
import logging
import math
import pdb
import numpy as np
import torch

from fedml_api.standalone.subavg.prune_func import dist_masks, real_prune, print_pruning


class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                 model_trainer,logger):
        self.logger = logger
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.logger.info("self.local_sample_number = " + str(self.local_sample_number))
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



    def train(self, w_global, masks, round_idx):
        num_comm_params = self.model_trainer.count_communication_params(w_global)
        w_per  = real_prune(w_global,masks)
        # self.logger.info("before train{}".format(sum([torch.count_nonzero(w_per[name]) for name in w_per])))
        self.model_trainer.set_model_params(w_per)
        self.model_trainer.set_masks(masks)
        self.model_trainer.set_id(self.client_idx)
        self.dense, _ = print_pruning(w_per)
        m1, m2 = self.model_trainer.train(self.local_training_data, self.device, self.args, round_idx)
        dist = dist_masks(m1, m2)

        state_dict = copy.deepcopy(self.model_trainer.get_model_params())
        final_mask = copy.deepcopy(masks)

        if dist > self.args.dist_thresh and self.dense > self.args.dense_ratio:
            # self.logger.info("test prune{}".format(sum([torch.count_nonzero(m2[name]) for name in m2])))
            metrics = self.local_test(state_dict, m2,False)
            acc =  metrics['test_correct'] / metrics['test_total']
            # print(f'acc after pruning: {acc}')
            if acc > self.args.acc_thresh:
                # print(f'Pruned! acc after pruning {acc}')
                state_dict = real_prune(state_dict, m2)
                final_mask = copy.deepcopy(m2)
        self.logger.info("after train{}".format(sum([torch.count_nonzero(state_dict[name]) for name in state_dict])))
        self.logger.info("-----------------------------------")
        training_flops = self.args.epochs * self.local_sample_number * self.model_trainer.count_training_flops_per_sample()
        num_comm_params += self.model_trainer.count_communication_params(state_dict)
        return final_mask, state_dict,training_flops, num_comm_params




    def local_test(self, w, mask_per, b_use_test_dataset):
        w_prune = real_prune(w, mask_per)
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        self.model_trainer.set_model_params(w_prune)
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
