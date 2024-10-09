from networks.ibc import EBMConvMLP
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.algorithm_utils import DerivativeFreeOptimizer
import numpy as np
from utils.tools import recursive_update_dict


# TODO: add robomimic encoder 
class IBCAgent:
    def __init__(self, model, dataloader, config: dict) -> None:
        device = config["device"]
        self.device = device
        self.config = config
        self.model = model
        self.model.to(device)
        self.model.train()

        self.ibc_agent_optimizer = DerivativeFreeOptimizer(dataloader["train"], config)
        self.dataloader = dataloader


    def give_default_config(self):
        # TODO: standarize this to make sure it dosn't conflict with robomimic config
        CNN_config = {
            "in_channels": 1,
            "blocks": (16, 32, 32),
            "activation_fn": "ReLU"
        }
        MLP_config = dict(
            input_dim=16,
            hidden_dim=256,
            output_dim=1,
            hidden_depth=1,
            dropout_prob=0.0, 
            activation_fn="ReLU")
        
        # update MLP default input according to reducer type
        reducer = self.config.get("reducer", "SpatialSoftArgmax")
        if reducer == "SpatialSoftArgmax":
            MLP_config["input_dim"] *= 2

        # add target dim
        MLP_config["input_dim"] += self.config.get("action_dim", 2)

        spatial_reduction = "SpatialSoftArgmax"
        coord_conv = False
        return dict(
            CNN_config=CNN_config,
            MLP_config=MLP_config,
            spatial_reduction=spatial_reduction,
            coord_conv=coord_conv
        )

    def get_actions(self, obs: dict) -> torch.Tensor:
        # TODO: change this for adapting to stack of images
        # make obs a batch
        self.model.eval()
        # obs_image = obs["image"]
        # obs_image = torch.tensor(obs_image, dtype=torch.float32).to(self.device)
        # obs_image = obs_image.unsqueeze(0)
        obs = {key: torch.tensor(obs[key], dtype=torch.float32).to(self.device) for key in obs.keys()}
        actions = self.ibc_agent_optimizer.infer(obs, self.model)
        actions = actions.squeeze(0).detach().cpu().numpy()

        # random_idx = np.random.randint(0, 100)
        # test_obs = self.dataloader["test"].dataset.normalized_train_data["image"][random_idx]
        # test_obs = torch.tensor(test_obs, dtype=torch.float32).to(self.device)
        # test_obs = test_obs.unsqueeze(0)
        # test_actions = self.ibc_agent_optimizer.infer(test_obs, self.model)
        # print("test_actions", test_actions)

        return actions
    
    def eval(self):
        self.model.eval()
    
    def train(self):
        self.model.train()



    