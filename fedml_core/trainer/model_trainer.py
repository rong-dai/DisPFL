from abc import ABC, abstractmethod

import torch

from fedml_api.utils.main_flops_counter import count_training_flops, count_inference_flops


class ModelTrainer(ABC):
    """Abstract base class for federated learning trainer.
       1. The goal of this abstract class is to be compatible to
       any deep learning frameworks such as PyTorch, TensorFlow, Keras, MXNET, etc.
       2. This class can be used in both server and client side
       3. This class is an operator which does not cache any states inside.
    """
    def __init__(self, model, args=None):
        self.model = model
        self.id = 0
        self.args = args

    def set_id(self, trainer_id):
        self.id = trainer_id

    @abstractmethod
    def get_model_params(self):
        pass

    @abstractmethod
    def set_model_params(self, model_parameters):
        pass

    @abstractmethod
    def train(self, train_data, device, args=None):
        pass

    @abstractmethod
    def test(self, test_data, device, args=None):
        pass

    def count_training_flops_per_sample(self):
        return count_training_flops(self.model,self.args.dataset)

    def count_full_flops_per_sample(self):
        return count_training_flops(self.model,self.args.dataset, full=True)

    def count_inference_flops(self, w):
        self.set_model_params(w)
        return count_inference_flops(self.model,self.args.dataset)

    def count_communication_params(self, update_to_server):
        num_non_zero_weights = 0
        for name in update_to_server:
            num_non_zero_weights += torch.count_nonzero(update_to_server[name])
        return num_non_zero_weights


    @abstractmethod
    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        pass

