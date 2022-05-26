import copy
import logging
import pickle
import random

import numpy as np
import torch
import pdb

from fedml_api.standalone.fedfomo.client import Client


class FEDFOMOAPI(object):
    def __init__(self, dataset, device, args, model_trainer,logger):
        self.logger = logger
        self.device = device
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, val_data_local_dict, test_data_local_dict, class_num] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.val_data_local_dict = val_data_local_dict
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
        w_global = self.model_trainer.get_model_params()
        w_per_mdls = []
        # 初始化
        for clnt in range(self.args.client_num_in_total):
            w_per_mdls.append(copy.deepcopy(w_global))
        # device = {device} cuda:0apply mask to init weights
        weights_locals = np.full((self.args.client_num_in_total, self.args.client_num_in_total), 1.0 / self.args.client_num_in_total)
        p_choose_locals = np.ones(shape=(self.args.client_num_in_total, self.args.client_num_in_total))
        for round_idx in range(self.args.comm_round):
            self.logger.info("################Communication round : {}".format(round_idx))
            final_tst_results_ths_round = []
            tst_results_ths_round = []
            w_per_mdls_lstrd = copy.deepcopy(w_per_mdls)
            
            for clnt_idx in range(self.args.client_num_in_total):
                self.logger.info('@@@@@@@@@@@@@@@@ Training Client CM({}): {}'.format(round_idx, clnt_idx))

                client = self.client_list[clnt_idx]
                w_local_mdl = w_per_mdls_lstrd[clnt_idx]

                w_local_mdl, training_flops, num_comm_params, tst_results = client.train(copy.deepcopy(w_local_mdl),
                                                                                   round_idx)

                tst_results_ths_round.append(tst_results)

                nei_indexs = self._benefit_choose(round_idx, clnt_idx, self.args.client_num_in_total,
                                                  self.args.client_num_per_round, p_choose_locals[clnt_idx])
                # 如果不是全选，则补上当前clint，进行聚合操作
                if self.args.client_num_in_total != self.args.client_num_per_round:
                    nei_indexs = np.append(nei_indexs, clnt_idx)

                nei_indexs = np.sort(nei_indexs)

                self.logger.info("client_indexes = " + str(nei_indexs))

                weights_locals[clnt_idx] = self._updates_weight_local(clnt_idx, nei_indexs, w_per_mdls_lstrd, copy.deepcopy(weights_locals[clnt_idx]), copy.deepcopy(w_local_mdl))
                weights_show = [weights_locals[clnt_idx][i] for i in nei_indexs]
                self.logger.info("Computing Agg weights: " + str(weights_show))

                p_choose_locals[clnt_idx] = p_choose_locals[clnt_idx] + weights_locals[clnt_idx]

                w_local_mdl = self._aggregate_func(clnt_idx, self.args.client_num_in_total,
                                                            self.args.client_num_per_round, nei_indexs,
                                                            w_per_mdls_lstrd, copy.deepcopy(weights_locals[clnt_idx]), copy.deepcopy(w_local_mdl))


                test_local_metrics = client.local_test(w_local_mdl, True)
                self.logger.info("test acc on this client after agg {} / {} : {:.2f}".format(test_local_metrics['test_correct'],
                                                                                      test_local_metrics['test_total'],
                                                                                      test_local_metrics['test_acc']))

                final_tst_results_ths_round.append(test_local_metrics)

                # 更新local model和local mask
                w_per_mdls[clnt_idx] = copy.deepcopy(w_local_mdl)

                self.stat_info["sum_training_flops"] += training_flops
                self.stat_info["sum_comm_params"] += num_comm_params


            self._local_test_on_all_clients(tst_results_ths_round, round_idx)
            self._local_test_on_all_clients_new_mask(final_tst_results_ths_round, round_idx)
            # self._local_test_on_all_clients(w_global, round_idx)
        # self.record_avg_inference_flops(w_global)
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

    def _benefit_choose(self, round_idx, cur_clnt, client_num_in_total, client_num_per_round, p_choose):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            p_choose[cur_clnt] = 0
            if random.random() >= 0.5:
                client_indexes = np.argsort(p_choose)[-num_clients:]
            else:
                client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
                while cur_clnt in client_indexes:
                    client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)

        # self.logger.info("client_indexes = %s" % str(client_indexes))
        return client_indexes
        # For fedslim, we update the meta network for training purpose

    def _updates_weight_local(self, curr_idx, nei_indexs, w_per_mdls_lstrd, weight_local, w_local):
        client = self.client_list[curr_idx]
        metrics = client.val_test(w_per_mdls_lstrd[curr_idx], self.val_data_local_dict[curr_idx])
        loss_cur_clnt = metrics["test_loss"]
        for nei_clnt in nei_indexs:
            if nei_clnt == curr_idx:
                metrics = client.val_test(w_local, self.val_data_local_dict[curr_idx])
            else:
                metrics = client.val_test(w_per_mdls_lstrd[nei_clnt], self.val_data_local_dict[curr_idx])

            loss_nei_clnt = metrics["test_loss"]
            params_dif = []
            for key in w_per_mdls_lstrd[curr_idx]:
                if nei_clnt == curr_idx:
                    params_dif.append((w_local[key] - w_per_mdls_lstrd[curr_idx][key]).view(-1))
                else:
                    params_dif.append((w_per_mdls_lstrd[nei_clnt][key] - w_per_mdls_lstrd[curr_idx][key]).view(-1))

            params_dif = torch.cat(params_dif)
            if torch.norm(params_dif) == 0:
                weight_local[nei_clnt] = 0.0
            else:
                weight_local[nei_clnt] = (loss_cur_clnt - loss_nei_clnt) / (torch.norm(params_dif))

        return weight_local

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

    def _avg_aggregate(self, per_mdls):
        w_tmp = copy.deepcopy(per_mdls[0])
        w = 1 / len(per_mdls)
        for k in w_tmp.keys():
            w_tmp[k] = w_tmp[k] - w_tmp[k]
            for clnt in range(len(per_mdls)):
                w_tmp[k] += per_mdls[clnt][k] * w

        return w_tmp

    def _aggregate_func(self, cur_clnt, client_num_in_total, client_num_per_round, nei_indexs, w_per_mdls_lstrd, weights_local, w_local_mdl):
        self.logger.info('Doing local aggregation!')
        w_easy = copy.deepcopy(weights_local[nei_indexs])
        w_easy = np.maximum(w_easy, 0)
        w_easy = np.sum(w_easy)
        if w_easy == 0.0:
            return w_per_mdls_lstrd[cur_clnt]

        weight_tmp = np.maximum(weights_local, 0)
        w_tmp = copy.deepcopy(w_per_mdls_lstrd[cur_clnt])
        for k in w_tmp.keys():
            for clnt in nei_indexs:
                if clnt == cur_clnt:
                    w_tmp[k] += (w_local_mdl[k] - w_per_mdls_lstrd[cur_clnt][k]) * weight_tmp[clnt] / w_easy
                else:
                    w_tmp[k] += (w_per_mdls_lstrd[clnt][k] - w_per_mdls_lstrd[cur_clnt][k]) * weight_tmp[clnt] / w_easy

        return w_tmp

    def _local_test_on_all_clients(self, tst_results_ths_round, round_idx):

        self.logger.info("################local_test_on_all_clients before agg in communication round: {}".format(round_idx))

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

            # # # train data
            # train_local_metrics = client.local_test(w_per,False)
            # train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
            # train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
            # train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

            # test data
            test_metrics['num_samples'].append(copy.deepcopy(tst_results_ths_round[client_idx]['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(tst_results_ths_round[client_idx]['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(tst_results_ths_round[client_idx]['test_loss']))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        # test on training dataset
        # train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        # train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

        # # test on test dataset
        test_acc = sum([test_metrics['num_correct'][i] / test_metrics['num_samples'][i] for i in
                        range(self.args.client_num_in_total)]) / self.args.client_num_in_total
        test_loss = sum([np.array(test_metrics['losses'][i]) / np.array(test_metrics['num_samples'][i]) for i in
                         range(self.args.client_num_in_total)]) / self.args.client_num_in_total

        # stats = {'training_acc': train_acc, 'training_loss': train_loss}
        # wandb.log({"Train/Acc": train_acc, "round": round_idx})
        # wandb.log({"Train/Loss": train_loss, "round": round_idx})
        # self.logger.info(stats)

        stats = {'test_acc': test_acc, 'test_loss': test_loss}

        self.logger.info(stats)
        self.stat_info["old_mask_test_acc"].append(test_acc)

    def _local_test_on_all_clients_new_mask(self, tst_results_ths_round, round_idx):

        self.logger.info("################local_test_on_all_clients after agg  in communication round: {}".format(round_idx))

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

            # # # train data
            # train_local_metrics = client.local_test(w_per,False)
            # train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
            # train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
            # train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

            # test data
            test_metrics['num_samples'].append(copy.deepcopy(tst_results_ths_round[client_idx]['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(tst_results_ths_round[client_idx]['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(tst_results_ths_round[client_idx]['test_loss']))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        # test on training dataset
        # train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        # train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

        # # test on test dataset
        test_acc = sum([test_metrics['num_correct'][i] / test_metrics['num_samples'][i] for i in
                        range(self.args.client_num_in_total)]) / self.args.client_num_in_total
        test_loss = sum([np.array(test_metrics['losses'][i]) / np.array(test_metrics['num_samples'][i]) for i in
                         range(self.args.client_num_in_total)]) / self.args.client_num_in_total

        # stats = {'training_acc': train_acc, 'training_loss': train_loss}
        # wandb.log({"Train/Acc": train_acc, "round": round_idx})
        # wandb.log({"Train/Loss": train_loss, "round": round_idx})
        # self.logger.info(stats)

        stats = {'test_acc': test_acc, 'test_loss': test_loss}

        self.logger.info(stats)
        self.stat_info["new_mask_test_acc"].append(test_acc)


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
        self.stat_info["global_test_acc"] = []
        self.stat_info["person_test_acc"] = []
        self.stat_info["final_masks"] = []
        self.stat_info["new_mask_test_acc"] = []
        self.stat_info["old_mask_test_acc"] = []
