import copy
import logging
import pickle
import random

import numpy as np
import torch

from fedml_api.standalone.local.client import Client


class LocalAPI(object):
    def __init__(self, dataset, device, args, model_trainer, logger):
        self.device = device
        self.args = args
        self.logger = logger
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
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
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer, self.logger)
            self.client_list.append(c)
        self.logger.info("############setup_clients (END)#############")

        # def random_initialize_channel_tensors(self):
        #     channel_tensor = {}
        #     for name, module in self.model_trainer.model.named_modules():
        #         if "meta" in name:
        #             max_input = int(module.weight.shape()[1])
        #             max_output = int(module.weight.shape()[0])
        #             input = int(max_input * random.random())
        #             output = int(max_output * random.random())
        #             channel_tensor[name] = torch.FloatTensor([input, output])
        #     return channel_tensor

    def train(self):
        # for first initialization, all the weights and the masks are the same
        w_global = self.model_trainer.get_model_params()
        # device = {device} cuda:0apply mask to init weights
        w_pers = [copy.deepcopy(w_global) for i in range(self.args.client_num_in_total)]
        for round_idx in range(self.args.comm_round):
            self.logger.info("################Communication round : {}".format(round_idx))
            w_locals = []
            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                   self.args.client_num_per_round)
            client_indexes = np.sort(client_indexes)
            self.logger.info("client_indexes = " + str(client_indexes))

            tst_results_ths_round = []
            for cur_clnt in client_indexes:
                self.logger.info('@@@@@@@@@@@@@@@@ Training Client CM({}): {}'.format(round_idx, cur_clnt))
                client = self.client_list[cur_clnt]
                w_per, training_flops,num_comm_params, tst_results = client.train(copy.deepcopy(w_pers[cur_clnt]), round_idx)
                w_pers[cur_clnt] = copy.deepcopy(w_per)
                self.stat_info["sum_training_flops"] += training_flops
                self.stat_info["sum_comm_params"] += num_comm_params
                tst_results_ths_round.append(tst_results)

            self._local_test_on_all_clients(tst_results_ths_round, round_idx)

            # # update masks after test
            # for idx, client in enumerate(self.client_list):
            #     client_idx = client_indexes[idx]
            #     c_pers[client_idx] = copy.deepcopy(new_masks_locals[idx])
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

    def _local_test_on_all_clients(self, tst_results_ths_round, round_idx):

            self.logger.info("################local_test_on_all_clients in communication round: {}".format(round_idx))

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
                test_metrics['num_samples'].append(copy.deepcopy(tst_results_ths_round[client_idx]['test_total']))
                test_metrics['num_correct'].append(copy.deepcopy(tst_results_ths_round[client_idx]['test_correct']))
                test_metrics['losses'].append(copy.deepcopy(tst_results_ths_round[client_idx]['test_loss']))

                """
                Note: CI environment is CPU-based computing. 
                The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
                """
                if self.args.ci == 1:
                    break

            test_acc = sum([test_metrics['num_correct'][i] / test_metrics['num_samples'][i] for i in range(self.args.client_num_in_total) ] )/self.args.client_num_in_total
            test_loss = sum([np.array(test_metrics['losses'][i]) / np.array(test_metrics['num_samples'][i]) for i in range(self.args.client_num_in_total)])/self.args.client_num_in_total


            stats = {'test_acc': test_acc, 'test_loss': test_loss}

            self.logger.info(stats)
            self.stat_info["test_acc"].append(test_acc)

    def _local_test_on_all_clients_orig(self, w_pers, round_idx):

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

        client = self.client_list[0]

        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(0, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])
            # # train data
            # train_local_metrics = client.local_test(False)
            # train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
            # train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
            # train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

            # test data
            test_local_metrics = client.local_test(w_pers[client_idx], True)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        # # test on training dataset
        # train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        # train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])

        # stats = {'training_acc': train_acc, 'training_loss': train_loss}
        # wandb.log({"Train/Acc": train_acc, "round": round_idx})
        # wandb.log({"Train/Loss": train_loss, "round": round_idx})
        # self.logger.info(stats)

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        # wandb.log({"Test/Acc": test_acc, "round": round_idx})
        # wandb.log({"Test/Loss": test_loss, "round": round_idx})
        self.stat_info["test_acc"].append(test_acc)
        self.logger.info(stats)

    def record_information(self):
        path = "../../results/" + self.args.dataset + "/" + self.args.identity
        output = open(path, 'wb')
        pickle.dump(self.stat_info, output)

    def record_avg_inference_flops(self, w_pers, mask_pers=None):
        inference_flops = []
        for client_idx in range(self.args.client_num_in_total):
            inference_flops += [self.model_trainer.count_inference_flops(w_pers[client_idx])]
        avg_inference_flops = sum(inference_flops) / len(inference_flops)
        self.stat_info["avg_inference_flops"] = avg_inference_flops

    def init_stat_info(self):
        self.stat_info = {}

        self.stat_info["sum_comm_params"] = 0
        self.stat_info["sum_training_flops"] = 0
        self.stat_info["avg_inference_flops"] = 0
        self.stat_info["test_acc"] = []
        self.stat_info["final_masks"] = []

