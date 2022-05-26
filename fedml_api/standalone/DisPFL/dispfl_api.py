import copy
import logging
import math
import pickle
import random
import time

import pdb
import numpy as np
import torch

from fedml_api.standalone.DisPFL import client
from fedml_api.standalone.DisPFL.client import Client
from fedml_api.standalone.DisPFL.slim_util import model_difference
from fedml_api.standalone.DisPFL.slim_util import hamming_distance

class dispflAPI(object):
    def __init__(self, dataset, device, args, model_trainer, logger):
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
        self.class_counts = class_counts
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

        # for first initialization, all the weights and the masks are the same
        # 在加入decentralized training时，所有client公用一个personalized mask和一个global model
        # different_initial 控制初始的client personalized mask是否是相同的，默认是相同的 即different_initial=False
        # masks = self.model_trainer.init_masks()
        params = self.model_trainer.get_trainable_params()
        w_spa = [self.args.dense_ratio for i in range(self.args.client_num_in_total)]
        if self.args.uniform:
            sparsities = self.model_trainer.calculate_sparsities(params,distribution="uniform", sparse = self.args.dense_ratio)
        else:
            sparsities = self.model_trainer.calculate_sparsities(params,sparse = self.args.dense_ratio)
        if not self.args.different_initial:
            temp = self.model_trainer.init_masks(params, sparsities)
            mask_pers_local = [copy.deepcopy(temp) for i in range(self.args.client_num_in_total)]
        elif not self.args.diff_spa:
            mask_pers_local = [copy.deepcopy(self.model_trainer.init_masks(params, sparsities)) for i in range(self.args.client_num_in_total)]
        else:
            divide = 5
            p_divide = [0.2, 0.4, 0.6, 0.8, 1.0]
            mask_pers_local = []
            for i in range(self.args.client_num_in_total):
                sparsities = self.model_trainer.calculate_sparsities(params, sparse = p_divide[i % divide])
                temp = self.model_trainer.init_masks(params, sparsities)
                mask_pers_local.append(temp)
                w_spa[i] = p_divide[i % divide]
        # mask_pers_local 是clnt的pmask矩阵，只保存在本地
        # communicate时用的是mask_pers_shared，这一部分其实也可以从接受到的local_mdl推测到，为了方便coding使用的是mask_pers_shared
        w_global = self.model_trainer.get_model_params()
        w_per_mdls = []
        updates_matrix = []
        # 初始化
        for clnt in range(self.args.client_num_in_total):
            w_per_mdls.append(copy.deepcopy(w_global))
            updates_matrix.append(copy.deepcopy(w_global))
            for name in mask_pers_local[clnt]:
                w_per_mdls[clnt][name] = w_global[name] * mask_pers_local[clnt][name]
                updates_matrix[clnt][name] = updates_matrix[clnt][name] - updates_matrix[clnt][name]

        #本地维护一个w_per_globals，保存local client的状态
        w_per_globals = [copy.deepcopy(w_global) for i in range(self.args.client_num_in_total)]

        # mask_pers_shared 保存每一个client上一轮更新的部分的mask
        mask_pers_shared = copy.deepcopy(mask_pers_local)
        # 保存dist矩阵，存储每个client和其他client的距离
        dist_locals = np.zeros(shape=(self.args.client_num_in_total, self.args.client_num_in_total))

        for round_idx in range(self.args.comm_round):
            self.logger.info("################Communication round : {}".format(round_idx))

            active_ths_rnd = np.random.choice([0, 1], size = self.args.client_num_in_total, p = [1.0 - self.args.active, self.args.active])
            # 更新communication round时的所有personalized model
            w_per_mdls_lstrd = copy.deepcopy(w_per_mdls)
            mask_pers_shared_lstrd = copy.deepcopy(mask_pers_shared)

            # 在每一个communication rounds需要进行每个client的local training
            tst_results_ths_round = []
            final_tst_results_ths_round = []
            for clnt_idx in range(self.args.client_num_in_total):
                if active_ths_rnd[clnt_idx] == 0:
                    self.logger.info('@@@@@@@@@@@@@@@@ Client Drop this round CM({}) with spasity {}: {}'.format(round_idx, w_spa[clnt_idx], clnt_idx))

                self.logger.info('@@@@@@@@@@@@@@@@ Training Client CM({}) with spasity {}: {}'.format(round_idx, w_spa[clnt_idx], clnt_idx))
                # 记录当前mask变化了多少
                dist_locals[clnt_idx][clnt_idx], total_dis = hamming_distance(mask_pers_shared_lstrd[clnt_idx], mask_pers_local[clnt_idx])
                self.logger.info("local mask changes: {} / {}".format(dist_locals[clnt_idx][clnt_idx], total_dis))
                if active_ths_rnd[clnt_idx] == 0:
                    nei_indexs = np.array([])
                else:
                    nei_indexs = self._benefit_choose(round_idx, clnt_idx, self.args.client_num_in_total,
                                                  self.args.client_num_per_round, dist_locals[clnt_idx], total_dis, self.args.cs, active_ths_rnd)
                # 如果不是全选，则补上当前clint，进行聚合操作
                if self.args.client_num_in_total != self.args.client_num_per_round:
                    nei_indexs = np.append(nei_indexs, clnt_idx)

                nei_indexs = np.sort(nei_indexs)


                # 更新dist_locals 矩阵
                for tmp_idx in nei_indexs:
                    if tmp_idx != clnt_idx:
                        dist_locals[clnt_idx][tmp_idx],_ = hamming_distance(mask_pers_local[clnt_idx], mask_pers_shared_lstrd[tmp_idx])

                if self.args.cs!="full":
                    self.logger.info("choose client_indexes: {}, accoring to {}".format(str(nei_indexs), self.args.cs))
                else:
                    self.logger.info("choose client_indexes: {}, accoring to {}".format(str(nei_indexs), self.args.cs))
                if active_ths_rnd[clnt_idx] != 0:
                    nei_distances = [dist_locals[clnt_idx][i] for i in nei_indexs]
                    self.logger.info("choose mask diff: " + str(nei_distances))

                # Update each client's local model and the so-called consensus model
                if active_ths_rnd[clnt_idx] == 1:
                    w_local_mdl, w_per_globals[clnt_idx] = self._aggregate_func(clnt_idx, self.args.client_num_in_total, self.args.client_num_per_round, nei_indexs,
                                    w_per_mdls_lstrd, mask_pers_local, mask_pers_shared_lstrd)
                else:
                    w_local_mdl, w_per_globals[clnt_idx] = copy.deepcopy(w_per_mdls_lstrd[clnt_idx]), copy.deepcopy(w_per_mdls_lstrd[clnt_idx])

                # 聚合好模型后，更新shared mask
                mask_pers_shared[clnt_idx] = copy.deepcopy(mask_pers_local[clnt_idx])

                # 设置client进行local training
                client = self.client_list[clnt_idx]

                test_local_metrics = client.local_test(w_local_mdl, True)
                final_tst_results_ths_round.append(test_local_metrics)


                new_mask, w_local_mdl, updates_matrix[clnt_idx], training_flops, num_comm_params, tst_results = client.train(copy.deepcopy(w_local_mdl), copy.deepcopy(mask_pers_local[clnt_idx]), round_idx)
                tst_results_ths_round.append(tst_results)

                # 更新local model和local mask
                w_per_mdls[clnt_idx] = copy.deepcopy(w_local_mdl)
                mask_pers_local[clnt_idx] = copy.deepcopy(new_mask)

                # 更新w_per_globals w_per_globals里存储的是每个client的训练完的最后状态(dense models)
                for key in w_per_globals[clnt_idx]:
                    w_per_globals[clnt_idx][key] += updates_matrix[clnt_idx][key]
                self.stat_info["sum_training_flops"] += training_flops
                self.stat_info["sum_comm_params"] += num_comm_params

            self._local_test_on_all_clients(tst_results_ths_round, round_idx)
            self._local_test_on_all_clients_new_mask(final_tst_results_ths_round, round_idx)

        for index in range(self.args.client_num_in_total):
            tmp_dist = []
            for clnt in range(self.args.client_num_in_total):
                tmp, _ = hamming_distance(mask_pers_local[index], mask_pers_local[clnt])
                tmp_dist.append(tmp.item())
            self.stat_info["mask_dis_matrix"].append(tmp_dist)

        ## uncomment this if u like to save the final mask; Note masks for Resnet could be large, up to 1GB for 100 clients
        if self.args.save_masks:
            saved_masks = [{} for index in range(len(mask_pers_local))]
            for index, mask in enumerate(mask_pers_local):
                for name in mask:
                        saved_masks[index][name] = mask[name].data.bool()
            self.stat_info["final_masks"] =saved_masks
        return

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        self.logger.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _benefit_choose(self, round_idx, cur_clnt, client_num_in_total, client_num_per_round, dist_local, total_dist, cs = False, active_ths_rnd = None):
        if client_num_in_total == client_num_per_round:
            # If one can communicate with all others and there is no bandwidth limit
            client_indexes = [client_index for client_index in range(client_num_in_total)]
            return client_indexes

        if cs == "random":
            # Random selection of available clients
            num_clients = min(client_num_per_round, client_num_in_total)
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
            while cur_clnt in client_indexes:
                client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)

        elif cs == "ring":
            # Ring Topology in Decentralized setting
            left = (cur_clnt - 1 + client_num_in_total) % client_num_in_total
            right = (cur_clnt + 1) % client_num_in_total
            client_indexes = np.asarray([left, right])

        elif cs == "full":
            # Fully-connected Topology in Decentralized setting
            client_indexes = np.array(np.where(active_ths_rnd==1)).squeeze()
            client_indexes = np.delete(client_indexes, int(np.where(client_indexes==cur_clnt)[0]))
        return client_indexes

    def _aggregate_func(self, cur_idx, client_num_in_total, client_num_per_round, nei_indexs, w_per_mdls_lstrd, mask_pers, mask_pers_shared_lstrd):
        self.logger.info('Doing local aggregation!')
        # Use the received models to infer the consensus model
        count_mask = copy.deepcopy(mask_pers_shared_lstrd[cur_idx])
        for k in count_mask.keys():
            count_mask[k] = count_mask[k] - count_mask[k]
            for clnt in nei_indexs:
                count_mask[k] += mask_pers_shared_lstrd[clnt][k]
        for k in count_mask.keys():
            count_mask[k] = np.divide(1, count_mask[k], out = np.zeros_like(count_mask[k]), where = count_mask[k] != 0)
        w_tmp = copy.deepcopy(w_per_mdls_lstrd[cur_idx])
        for k in w_tmp.keys():
            w_tmp[k] = w_tmp[k] - w_tmp[k]
            for clnt in nei_indexs:
                w_tmp[k] += torch.from_numpy(count_mask[k]) * w_per_mdls_lstrd[clnt][k]
        w_p_g = copy.deepcopy(w_tmp)
        for name in mask_pers[cur_idx]:
            w_tmp[name] = w_tmp[name] * mask_pers[cur_idx][name]
        return w_tmp, w_p_g

    def _local_test_on_all_clients(self, tst_results_ths_round, round_idx):
            self.logger.info("################local_test_on_all_clients after local training in communication round: {}".format(round_idx))
            test_metrics = {
                'num_samples': [],
                'num_correct': [],
                'losses': []
            }
            for client_idx in range(self.args.client_num_in_total):
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

            # # test on test dataset
            test_acc = sum([test_metrics['num_correct'][i] / test_metrics['num_samples'][i] for i in range(self.args.client_num_in_total) ] )/self.args.client_num_in_total
            test_loss = sum([np.array(test_metrics['losses'][i]) / np.array(test_metrics['num_samples'][i]) for i in range(self.args.client_num_in_total)])/self.args.client_num_in_total

            stats = {'test_acc': test_acc, 'test_loss': test_loss}

            self.logger.info(stats)
            self.stat_info["old_mask_test_acc"].append(test_acc)

    def _local_test_on_all_clients_new_mask(self, tst_results_ths_round, round_idx):
        self.logger.info("################local_test_on_all_clients before local training in communication round: {}".format(round_idx))
        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }
        for client_idx in range(self.args.client_num_in_total):

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

        # # test on test dataset
        test_acc = sum([test_metrics['num_correct'][i] / test_metrics['num_samples'][i] for i in
                        range(self.args.client_num_in_total)]) / self.args.client_num_in_total
        test_loss = sum([np.array(test_metrics['losses'][i]) / np.array(test_metrics['num_samples'][i]) for i in
                         range(self.args.client_num_in_total)]) / self.args.client_num_in_total

        stats = {'test_acc': test_acc, 'test_loss': test_loss}

        self.logger.info(stats)
        self.stat_info["new_mask_test_acc"].append(test_acc)

    def record_avg_inference_flops(self, w_global, mask_pers=None):
        inference_flops=[]
        for client_idx in range(self.args.client_num_in_total):

            if mask_pers== None:
                inference_flops += [self.model_trainer.count_inference_flops(w_global)]
            else:
                w_per = {}
                for name in mask_pers[client_idx]:
                    w_per[name] = w_global[name] *mask_pers[client_idx][name]
                inference_flops+= [self.model_trainer.count_inference_flops(w_per)]
        avg_inference_flops = sum(inference_flops)/len(inference_flops)
        self.stat_info["avg_inference_flops"]=avg_inference_flops


    def init_stat_info(self, ):
        self.stat_info = {}
        self.stat_info["label_num"] =self.class_counts
        self.stat_info["sum_comm_params"] = 0
        self.stat_info["sum_training_flops"] = 0
        self.stat_info["avg_inference_flops"] = 0
        self.stat_info["old_mask_test_acc"] = []
        self.stat_info["new_mask_test_acc"] = []
        self.stat_info["final_masks"] = []
        self.stat_info["mask_dis_matrix"] = []


