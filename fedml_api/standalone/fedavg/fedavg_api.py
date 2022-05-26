import copy
import logging
import pickle
import random
import pdb
import numpy as np
import torch

from fedml_api.standalone.fedavg.client import Client


class FedAvgAPI(object):
    def __init__(self, dataset, device, args, model_trainer, logger):
        self.logger = logger
        self.device = device
        self.args = args
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

    def train(self):
        w_global = self.model_trainer.get_model_params()
        w_per_mdls = []
        # 初始化
        for clnt in range(self.args.client_num_in_total):
            w_per_mdls.append(copy.deepcopy(w_global))
        # device = {device} cuda:0apply mask to init weights
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

            for cur_clnt in client_indexes:
                self.logger.info('@@@@@@@@@@@@@@@@ Training Client CM({}): {}'.format(round_idx, cur_clnt))
                # update dataset
                client = self.client_list[cur_clnt]
                # update meta components in personal network
                w_per,training_flops,num_comm_params = client.train(copy.deepcopy(w_global), round_idx)
                w_per_mdls[cur_clnt] = copy.deepcopy(w_per)
                # self.logger.info("local weights = " + str(w))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w_per)))
                self.stat_info["sum_training_flops"] += training_flops
                self.stat_info["sum_comm_params"] += num_comm_params
            # update global meta weights
            w_global = self._aggregate(w_locals)

            self._test_on_all_clients(w_global, w_per_mdls, round_idx)
            # self._local_test_on_all_clients(w_global, round_idx)
        # self.record_avg_inference_flops(w_global)

        # 为了查看finetune的结果，在global avged model上再进行一轮训练
        self.logger.info("################Communication round Last Fine Tune Round")
        for clnt_idx in range(self.args.client_num_in_total):
            self.logger.info('@@@@@@@@@@@@@@@@ Training Client: {}'.format(clnt_idx))
            w_local_mdl = copy.deepcopy(w_global)
            client = self.client_list[clnt_idx]
            w_local_mdl, training_flops, num_comm_params = client.train(copy.deepcopy(w_local_mdl), -1)
            # 更新local model
            w_per_mdls[clnt_idx] = copy.deepcopy(w_local_mdl)

        self._test_on_all_clients(w_global, w_per_mdls, -1)



    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        self.logger.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, _) = w_locals[idx]
            training_num += sample_num
        w_global ={}
        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    w_global[k] = local_model_params[k] * w
                else:
                    w_global[k] += local_model_params[k] * w
        return w_global

    def _test_on_all_clients(self, w_global, w_per_mdls, round_idx):

        self.logger.info("################global_test_on_all_clients : {}".format(round_idx))

        g_test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        p_test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        for client_idx in range(self.args.client_num_in_total):
            # test data
            client = self.client_list[client_idx]
            g_test_local_metrics = client.local_test(w_global, True)
            g_test_metrics['num_samples'].append(copy.deepcopy(g_test_local_metrics['test_total']))
            g_test_metrics['num_correct'].append(copy.deepcopy(g_test_local_metrics['test_correct']))
            g_test_metrics['losses'].append(copy.deepcopy(g_test_local_metrics['test_loss']))

            p_test_local_metrics = client.local_test(w_per_mdls[client_idx], True)
            p_test_metrics['num_samples'].append(copy.deepcopy(p_test_local_metrics['test_total']))
            p_test_metrics['num_correct'].append(copy.deepcopy(p_test_local_metrics['test_correct']))
            p_test_metrics['losses'].append(copy.deepcopy(p_test_local_metrics['test_loss']))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break
        # test on test dataset
        g_test_acc = sum([np.array(g_test_metrics['num_correct'][i]) / np.array(g_test_metrics['num_samples'][i]) for i in
                        range(self.args.client_num_in_total)]) / self.args.client_num_in_total
        g_test_loss = sum([np.array(g_test_metrics['losses'][i]) / np.array(g_test_metrics['num_samples'][i]) for i in
                         range(self.args.client_num_in_total)]) / self.args.client_num_in_total

        p_test_acc = sum(
            [np.array(p_test_metrics['num_correct'][i]) / np.array(p_test_metrics['num_samples'][i]) for i in
             range(self.args.client_num_in_total)]) / self.args.client_num_in_total
        p_test_loss = sum([np.array(p_test_metrics['losses'][i]) / np.array(p_test_metrics['num_samples'][i]) for i in
                           range(self.args.client_num_in_total)]) / self.args.client_num_in_total


        stats = {'global_test_acc': g_test_acc, 'global_test_loss': g_test_loss}
        self.stat_info["global_test_acc"].append(g_test_acc)
        self.logger.info(stats)

        stats = {'person_test_acc': p_test_acc, 'person_test_loss': p_test_loss}
        self.stat_info["person_test_acc"].append(p_test_acc)
        self.logger.info(stats)

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
        self.stat_info["global_test_acc"] = []
        self.stat_info["person_test_acc"] = []
        self.stat_info["final_masks"] = []
