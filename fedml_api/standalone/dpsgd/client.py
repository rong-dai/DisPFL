import copy
import logging
import math
import time
import pdb
import numpy as np
import torch

class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                 model_trainer, logger):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.logger = logger
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

    def train(self, w_global,round):
        # self.logger.info(sum([torch.sum(w_per[name]) for name in w_per]))
        num_comm_params = self.model_trainer.count_communication_params(w_global)
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.set_id(self.client_idx)
        if self.args.type == "epoch":
            self.model_trainer.train(self.local_training_data, self.device, self.args, round, self.logger)
        elif self.args.type == "step":
            self.model_trainer.step_train(self.local_training_data, self.device, self.args, round)
        weights = self.model_trainer.get_model_params()
        # self.logger.info( "training_flops{}".format( self.model_trainer.count_training_flops_per_sample()))
        # self.logger.info("full{}".format(self.model_trainer.count_full_flops_per_sample()))
        training_flops = self.args.epochs * self.local_sample_number * self.model_trainer.count_training_flops_per_sample()
        num_comm_params += self.model_trainer.count_communication_params(weights)
        return weights,training_flops,num_comm_params

    def local_test(self, w, b_use_test_dataset = True):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        self.model_trainer.set_model_params(w)
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
