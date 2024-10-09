import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils
import torch.utils.data
from tqdm import tqdm
from utils.tools import Every
from utils.algorithm_utils import DerivativeFreeOptimizer
import pathlib
from typing import Union
from utils.pytorch_utils import dict_apply, get_leading_dims

# adapted from IBC Pytorch implementation
class TrainerIBC:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
    # stochastic_optimizer: DerivativeFreeOptimizer
    device: torch.device
    steps: int

    def __init__(self, model: nn.Module, dataloader, logger, config: dict) -> None:
        from networks.ibc import EBMConvMLP

        device = config["device"]
        self.device = device
        self.config = config

        self.model = model
        self.model.to(device)
        self.model.train()

        self.dataloader = dataloader

        obs_shape_meta = config["shape_meta"]["obs"]
        # get observation keys
        self.obs_keys = obs_shape_meta.keys()

        action_shape_meta = config["shape_meta"]["action"]
        self.action_shape = list(action_shape_meta["shape"])

        self.n_obs_steps = config.get("n_obs_steps", 1)
        self.n_predict_steps = config.get('n_predict_steps', 1)

        # get optimizer for IBC
        self.ibc_optimizer_train = DerivativeFreeOptimizer(dataloader["train"], config)
        self.ibc_optimizer_test = DerivativeFreeOptimizer(dataloader["test"], config)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr= config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 0.0),
            betas=(config.get("beta1", 0.9), config.get("beta2", 0.999)))
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
            step_size=config.get("lr_scheduler_step", 1000),
            gamma=config.get("lr_scheduler_gamma", 0.999))
        
        self.metrics = dict(IBC_train_loss = [], IBC_test_loss =0)   
        self.logger = logger
        self.steps = 0

        # self.online_eval_episodes = config.get("online_eval_episodes", 10)
        # self.max_online_eval_steps = config.get("max_online_eval_steps", 1000)

        
        
    def _train_step(self, 
                    batch_input_dict: dict, 
                    batch_target: torch.Tensor):
        # TODO: to always use B,T,D
        batch_dict_input = {key: batch_input_dict[key].to(self.device) for key in self.obs_keys}
        batch_target = batch_target.to(self.device)
        batch_target = batch_target[:, :self.n_predict_steps, ...]
        batch_input_dict = dict_apply(batch_dict_input, lambda x:  x[:, :self.n_obs_steps, ...])
        

        # Generate N negatives, one for each element in the batch: (B, N, D).
        # if batch_input is B,T,C,H,W, batch_target is B,T,D, then sample B*T*N, D
        sample_num = batch_target.size(0) * self.n_predict_steps
        negatives = self.ibc_optimizer_train.sample(sample_num, self.model)
        # reshape to B,N,T,D
        negatives = negatives.view(batch_target.size(0), -1, self.n_predict_steps, *self.action_shape)

        # Merge target and negatives: (B, N+1, T, D).
        targets = torch.cat([batch_target.unsqueeze(1), negatives], dim=1)

        # Generate a random permutation of the positives and negatives.
        idxs = torch.randn(targets.size(0),targets.size(1)).argsort(dim=1)
        targets = targets[torch.arange(targets.size(0)).unsqueeze(-1), idxs]

        # Get the original index of the positive. This will serve as the class label
        # for the loss.
        ground_truth = (idxs == 0).nonzero()[:, 1].to(self.device)

        # For every element in the mini-batch, there is 1 positive for which the EBM
        # should output a low energy value, and N negatives for which the EBM should
        # output high energy values.
        energy = self.model(batch_dict_input, targets)
                # Interpreting the energy as a negative logit, we can apply a cross entropy loss
        # to train the EBM.
        logits = -1.0 * energy
        loss = F.cross_entropy(logits, ground_truth)

        self.metrics["IBC_train_loss"].append(loss.item())

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

                
    # train one epoch
    def train(self):
        self.model.train()
        max_training_epochs = self.config.get("epochs_per_round", 10)
        max_training_steps = self.config.get("max_training_steps", None)
        assert max_training_steps or max_training_epochs, "Either max_training_steps or max_training_epochs must be specified"

        eval_every_n_steps = self.config.get("eval_every", 2000)
        log_every_n_steps = self.config.get("log_every", 2000)

        # eval_every = Every(eval_every_n_steps)
        # log_every = Every(log_every_n_steps)

        train_data = self.dataloader["train"]
        # eval_data = self.dataloader["test"]

            
        self.metrics["IBC_train_loss"] = []
        for input, target in tqdm(train_data, leave=False):
            # input_image = input["image"]
            # train an epoch
            self._train_step(input, target)
            self.steps += 1
        
        self.metrics["IBC_train_loss"] = np.mean(self.metrics["IBC_train_loss"])
        
        # # do evaluation if step matches
        # if eval_every(steps):
        #     self.eval(eval_data)
        #     # online evaluation
        # # log metrics if step matches
        # if log_every(steps):
        #     self.logger.scalar("IBC_train_loss", ibc_train_loss_mean)
        #     self.logger.scalar("IBC_test_loss", self.metrics["IBC_test_loss"])
        #     self.logger.write(step = steps)
        #     self.metrics["IBC_train_loss"] = []
        #     self.metrics["IBC_test_loss"] = 0

            
            # return steps for next training
        return self.steps, self.metrics

            
    
    def eval(self, dataloader):
        losses = []
        with torch.no_grad():
            self.model.eval()
            for input_dict, target in tqdm(dataloader, leave=False):
                target = target[:, :self.n_predict_steps, ...]
                # input_image = input["image"]    
                # input_image = input_image.to(self.device)
                input_dict = {key: input_dict[key].to(self.device) for key in self.obs_keys}
                target = target.to(self.device)
                inferred_result = self.ibc_optimizer_test.infer(input_dict, self.model)
                # print(inferred_result.shape, target.shape)
                loss = F.mse_loss(inferred_result, target)
                losses.append(loss.item())
        self.metrics["IBC_test_loss"] = np.mean(losses)
        return  self.metrics
    
    def save(self, checkpoint_dir, run_name, steps):
        default_dir = pathlib.Path("IBC_model")
        model_dir = checkpoint_dir / run_name if str(run_name) != "." else checkpoint_dir/ default_dir
        if not model_dir.exists():
            model_dir.mkdir(parents=True)
        model_path = model_dir / f"IBC_model_{steps}.pt"
        torch.save(self.model.state_dict(), model_path)
            


