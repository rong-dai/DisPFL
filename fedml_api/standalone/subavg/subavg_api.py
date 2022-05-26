import copy
import logging
import math
import pickle
import random

import numpy as np
import torch
# import wandb

from fedml_api.standalone.DisPFL.slim_util import model_difference
from fedml_api.standalone.subavg.client import Client


class SubAvgAPI(object):
    def __init__(self, dataset, device, args, model_trainer,logger):
        self.logger = logger
        self.device = device
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_counts] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)
        self.init_stat_info()

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        self.logger.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_in_total):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer,self.logger)
            self.client_list.append(c)
        self.logger.info("############setup_clients (END)#############")

    def train(self):
        # for first initialization, all the weights and the masks are the same
        masks = self.model_trainer.init_masks()
        mask_pers = [copy.deepcopy(masks) for i in range(self.args.client_num_in_total)]
        w_global = self.model_trainer.get_model_params()

        for round_idx in range(self.args.comm_round):
            self.logger.info("################Communication round : {}".format(round_idx))
            self.record_mask_diffrence(mask_pers)
            w_locals = []
            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                   self.args.client_num_per_round)
            self.logger.info("client_indexes = " + str(client_indexes))
            next_masks = []
            for idx in client_indexes:
                # update dataset
                client_idx = idx
                client = self.client_list[client_idx]
                # w_per = client.train(copy.deepcopy(w_global), copy.deepcopy(mask_pers[client_idx]), round_idx)
                new_mask, w_client,training_flops, num_comm_params = client.train(copy.deepcopy(w_global),
                                                                     copy.deepcopy(mask_pers[client_idx]), round_idx)
                # self.logger.info("local weights = " + str(w))
                w_locals.append((copy.deepcopy(mask_pers[client_idx]), copy.deepcopy(w_client)))
                next_masks.append((client_idx, copy.deepcopy(new_mask)))
                self.stat_info["sum_training_flops"] += training_flops
                self.stat_info["sum_comm_params"] += num_comm_params
                # w_pers[client_idx] = copy.deepcopy(virtual_information[0])

            w_global = self._aggregate(w_global, w_locals)
            # test results at last round
            if round_idx == self.args.comm_round - 1:
                self._local_test_on_all_clients(w_global, mask_pers, round_idx)
            elif round_idx % self.args.frequency_of_the_test == 0:
                self._local_test_on_all_clients(w_global, mask_pers, round_idx)

            # update public components for each client's full mask
            for client_idx, next_mask in next_masks:
                mask_pers[client_idx] = copy.deepcopy(next_mask)

        # saved_masks = [{} for index in range(len(mask_pers))]
        # for index, mask in enumerate(mask_pers):
        #     for name in mask:
        #             saved_masks[index][name] = mask[name].data.bool()
        # self.stat_info["final_masks"] = saved_masks
        self.record_avg_inference_flops(w_global, mask_pers)
        self.record_information()

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        self.logger.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    # def _aggregate(self, w_server,w_locals):
    #     # training_num = 0
    #     # for idx in range(len(w_locals)):
    #     #     (sample_num, _) = w_locals[idx]
    #     #     training_num += sample_num
    #     w_global = {}
    #     masks = [w_local[0] for w_local in w_locals]
    #     w_clients = [w_local[1] for w_local in w_locals]
    #     for k in masks[0]:
    #         for i in range(0, len(w_locals)):
    #             # local_sample_number, local_model_params = w_locals[i]
    #             # w = local_sample_number / training_num
    #             w = 1 / len(w_locals)
    #             if i == 0:
    #                 w_global[k] = w_clients[i][k] * w
    #             else:
    #                 w_global[k] += w_clients[i][k] * w
    #     return w_global

    def _aggregate(self, w_server, w_locals):
        masks =[w_local[0] for w_local in  w_locals]
        w_clients =[w_local[1] for w_local in  w_locals]
        for name in w_server:
            count = torch.zeros_like(w_clients[0][name])
            avg = torch.zeros_like(w_clients[0][name])
            for i in range(len(w_clients)):
                count += masks[i][name]
            for i in range(len(w_clients)):
                avg += 1/count*w_clients[i][name]
            ind = np.isfinite(avg)
            w_server[name][ind]= avg[ind]

            # print(f'Name: {name}, NAN: {np.mean(np.isnan(temp_server))}, INF: {np.mean(np.isinf(temp_server))}')
            # shape = w_server[name].data.cpu().numpy().shape
            # w_server[name].data = torch.from_numpy(temp_server.reshape(shape))
        return w_server

    def _local_test_on_all_clients(self, w_global, mask_pers, round_idx):

        self.logger.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }



        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """

            client = self.client_list[client_idx]

            # # # train data
            # train_local_metrics = client.local_test(w_global, mask_pers[client_idx],False)
            # train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
            # train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
            # train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

            # test data
            test_local_metrics = client.local_test(w_global, mask_pers[client_idx], True)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        # test on training dataset
        # train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        # train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

        # test on test dataset
        test_acc = sum([test_metrics['num_correct'][i] / test_metrics['num_samples'][i] for i in
                        range(self.args.client_num_in_total)]) / self.args.client_num_in_total
        test_loss = sum([np.array(test_metrics['losses'][i]) / np.array(test_metrics['num_samples'][i]) for i in
                         range(self.args.client_num_in_total)]) / self.args.client_num_in_total

        # stats = {'training_acc': train_acc, 'training_loss': train_loss}
        # wandb.log({"Train/Acc": train_acc, "round": round_idx})
        # wandb.log({"Train/Loss": train_loss, "round": round_idx})
        # self.logger.info(stats)
        self.stat_info["test_acc"].append(test_acc)
        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        # wandb.log({"Test/Acc": test_acc, "round": round_idx})
        # wandb.log({"Test/Loss": test_loss, "round": round_idx})
        self.logger.info(stats)

    def record_mask_diffrence(self, mask_pers):
        mean = {}
        for name in mask_pers[0]:
            mean[name] = torch.zeros_like(mask_pers[0][name])
            for mask in mask_pers:
                mean[name] += mask[name]
            mean[name] /= len(mask_pers)
        distance = 0
        for mask in mask_pers:
            distance += model_difference(mask, mean)
        # wandb.log({"mask_distance": distance})

    def record_information(self):
        path = "../../results/" + self.args.dataset + "/" + self.args.identity
        output = open(path, 'wb')
        pickle.dump(self.stat_info, output)

    def record_avg_inference_flops(self, w_global, mask_pers=None):
        inference_flops = []
        for client_idx in range(self.args.client_num_in_total):

            if mask_pers == None:
                inference_flops += [self.model_trainer.count_inference_flops(w_global)]
            else:
                w_per = {}
                for name in mask_pers[client_idx]:
                    w_per[name] = w_global[name] * mask_pers[client_idx][name]
                inference_flops += [self.model_trainer.count_inference_flops(w_per)]
        avg_inference_flops = sum(inference_flops) / len(inference_flops)
        self.stat_info["avg_inference_flops"] = avg_inference_flops

    def init_stat_info(self):
        self.stat_info = {}
        self.stat_info["sum_comm_params"] = 0
        self.stat_info["sum_training_flops"] = 0
        self.stat_info["avg_inference_flops"] = 0
        self.stat_info["test_acc"] = []
        self.stat_info["final_masks"] = []



