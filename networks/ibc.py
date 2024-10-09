# from IBC Pytorch implementation:
# only changed config format from clas to dict


import dataclasses
import enum
from functools import partial
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import networks.ibc_base as ibc_base
from networks.ibc_base import CoordConv, GlobalAvgPool2d, GlobalMaxPool2d, SpatialSoftArgmax
from utils.robomimic_utils import get_robomimic_config

import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
# TODO: check if this croprandomizer module is correct
import robomimic.models.obs_core as obs_core
import networks.crop_randomizer as dmvc

from utils.pytorch_utils import replace_submodules, dict_apply
from utils.tools import recursive_update_dict





class ActivationType(enum.Enum):
    RELU = nn.ReLU
    SELU = nn.SiLU


@dataclasses.dataclass(frozen=True)
class MLPConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int
    hidden_depth: int
    dropout_prob: Optional[float] = None
    activation_fn: ActivationType = ActivationType.RELU


class MLP(nn.Module):
    """A feedforward multi-layer perceptron."""

    def __init__(self, config: dict) -> None:
        super().__init__()

        dropout_layer: Callable
        if config["dropout_prob"] is not None:
            dropout_layer = partial(nn.Dropout, p=config["dropout_prob"])
        else:
            dropout_layer = nn.Identity
        activation = getattr(nn, config["activation_fn"])
        layers: Sequence[nn.Module]
        if config["hidden_depth"] == 0:
            layers = [nn.Linear(config["input_dim"], config["output_dim"])]
        else:
            layers = [
                nn.Linear(config["input_dim"], config["hidden_dim"]),
                activation(),
                dropout_layer(),
            ]
            for _ in range(config["hidden_depth"] - 1):
                layers += [
                    nn.Linear(config["hidden_dim"], config["hidden_dim"]),
                    activation(),
                    dropout_layer(),
                ]
            layers += [nn.Linear(config["hidden_dim"], config["output_dim"])]
        layers = [layer for layer in layers if not isinstance(layer, nn.Identity)]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        # print("MLP INPUT SHAPE", x.shape)
        # print("input x mlp", x.shape)   
        # print("dtype of x", x.dtype)
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        depth: int,
        activation_fn: classmethod,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(depth, depth, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(depth, depth, 3, padding=1, bias=True)
        self.activation = activation_fn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(x)
        out = self.conv2(out)
        return out + x


@dataclasses.dataclass(frozen=True)
class CNNConfig:
    in_channels: int
    blocks: Sequence[int] = dataclasses.field(default=(16, 32, 32))
    activation_fn: ActivationType = ActivationType.RELU


class CNN(nn.Module):
    """A residual convolutional network."""

    def __init__(self, config: dict) -> None:
        super().__init__()

        depth_in = config["in_channels"]

        layers = []
        activation = getattr(nn, config["activation_fn"])
        for depth_out in config["blocks"]:
            layers.extend(
                [
                    nn.Conv2d(depth_in, depth_out, 3, padding=1),
                    ResidualBlock(depth_out, activation),
                ]
            )
            depth_in = depth_out

        self.net = nn.Sequential(*layers)
        self.activation = activation()

    def forward(self, x: torch.Tensor, activate: bool = False) -> torch.Tensor:
        out = self.net(x)
        if activate:
            return self.activation(out)
        return out


class SpatialReduction(enum.Enum):
    SPATIAL_SOFTMAX = SpatialSoftArgmax
    AVERAGE_POOL = GlobalAvgPool2d
    MAX_POOL = GlobalMaxPool2d


@dataclasses.dataclass(frozen=True)
class ConvMLPConfig:
    cnn_config: CNNConfig
    mlp_config: MLPConfig
    spatial_reduction: SpatialReduction = SpatialReduction.AVERAGE_POOL
    coord_conv: bool = False


# class ConvMLP(nn.Module):
#     def __init__(self, config: ConvMLPConfig) -> None:
#         super().__init__()

#         self.coord_conv = config.coord_conv

#         self.cnn = CNN(config.cnn_config)
#         self.conv = nn.Conv2d(config.cnn_config.blocks[-1], 16, 1)
#         self.reducer = config.spatial_reduction.value()
#         self.mlp = MLP(config.mlp_config)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if self.coord_conv:
#             x = CoordConv()(x)
#         out = self.cnn(x, activate=True)
#         out = F.relu(self.conv(out))
#         out = self.reducer(out)
#         out = self.mlp(out)
#         return out


class EBMConvMLP(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()

        self.coord_conv = config["coord_conv"]

        self.cnn = CNN(config["CNN_config"])
        self.conv = nn.Conv2d(config["CNN_config"]["blocks"][-1], 16, 1)
        # self.reducer = config.spatial_reduction.value()
        self.reducer = getattr(ibc_base, config["spatial_reduction"])()
        self.mlp = MLP(config["MLP_config"])
        self.coord_conv_layers = CoordConv()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # TODO: change to adapt to new input format
        if self.coord_conv:
            x = self.coord_conv_layers(x)

        out = self.cnn(x, activate=True)
        out = F.relu(self.conv(out))
        out = self.reducer(out)
        # print("y shape", y.shape)
        fused = torch.cat([out.unsqueeze(1).expand(-1, y.size(1), -1), y], dim=-1)
        B, N, D = fused.size()
        # print("B N D", B, N, D)
        fused = fused.reshape(B * N, D)
        out = self.mlp(fused)
        return out.view(B, N)

class EBMWithRoboMimicEnoder(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        # get default config and update with given config
        default_config = give_default_config(config)
        config = recursive_update_dict(default_config, config)

        action_shape = config['shape_meta']['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        # get action prediction steps and observation prediction steps
        n_predict_steps = config.get('n_predict_steps', 1)
        n_obs_steps = config.get('n_obs_steps', 1)
        self.n_predict_steps = n_predict_steps
        self.n_obs_steps = n_obs_steps

        # generate obs config
        # need meta data for this, unlike case without RoboMimic encoder
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        # 
        obs_shapes = dict()
        # decide whether to use low-dim or image based on config
        obs_shape_meta:dict = config['shape_meta']['obs']
        # crop image config
        crop_image = config.get('crop_image', False)
        if crop_image:
            crop_shape = config.get('crop_shape', (76, 76))
        else:
            crop_shape = None
        # setup observations 
        for key, value in obs_shape_meta.items():
            shape = value['shape']
            obs_shapes[key] = list(shape)
            type = value.get('type', 'low_dim')

            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}") 
            
        self.obs_config = obs_config
        # get config for encoder from RoboMimic
        robomimic_config = get_robomimic_config(
            algo_name='bc_rnn', 
            hdf5_type='image', 
            task_name='square', 
            dataset_type='ph'
        )
        # modify config for our use case
        with robomimic_config.unlocked():
            robomimic_config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in robomimic_config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in robomimic_config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(robomimic_config)

        # load model
        # we only need the encoder part
        robomimic_policy: PolicyAlgo = algo_factory(
                algo_name=robomimic_config.algo_name,
                config=robomimic_config,
                obs_key_shapes=obs_shapes,
                ac_dim=action_dim,
                device='cpu',
            )
        
        self.encoder = obs_encoder = robomimic_policy.nets['policy'].nets['encoder'].nets['obs']
        print("obs_encoder", self.encoder)
        obs_encoder_group_norm = config.get('obs_encoder_group_norm', True)
        # replace batch norm with group norm
        if obs_encoder_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda m: isinstance(m, nn.BatchNorm2d),
                func=lambda m: nn.GroupNorm(num_groups = m.num_features // 16,
                                            num_channels=m.num_features)
            )

        # replace random crop with fixed crop if eval_fixed_crop is True
        eval_fixed_crop = config.get('eval_fixed_crop', True)
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                # TODO: check if this is correct
                predicate=lambda x: isinstance(x, obs_core.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )
        # get input dim for MLP
        obs_feature_dim = obs_encoder.output_shape()[0]
        in_action_dim = action_dim * n_predict_steps
        in_embed_channels = obs_feature_dim * n_obs_steps
        in_channels = in_embed_channels + in_action_dim
        # get MLP config
        config['MLP_config']['input_dim'] = in_channels
        # print("in_channels", in_channels)
        # print("obs_feature_dim", obs_feature_dim)
        self._mlp = MLP(config['MLP_config'])
        # TODO: check if can run with coord_conv == True
        self.coord_conv = config.get('coord_layer', False)
        if self.coord_conv:
            # TODO: implement CoordConv
            raise NotImplementedError("CoordConv is not implemented yet")
        else:
            self.coord_layer = nn.Identity()

    def forward(self, batch_dict_obs: dict, batch_samples: torch.Tensor) -> torch.Tensor:
        '''
        input:
            batch_dict_obs: dict, must have and only have keys in obs_config
            batch_samples: torch.Tensor
        '''
        # flatten if y have 4 dimensions
        if batch_samples.dim() == 4:
            # in this case, x is B,To,C,H,W and y is B,N,T,D
            # batch_samples = batch_samples[:,:,:self.n_predict_steps,...]
            B, N, Ta, Da = batch_samples.shape
            # flatten all items in batch_dict_obs
            # B,To,C,H,W -> B*To, C,H,W  
            batch_dict_obs = dict_apply(batch_dict_obs,
                                        lambda x: x[:,:self.n_obs_steps,...].view(B*self.n_obs_steps, *x.shape[2:]))
            # B,N,T,D -> B,N,T*D
            batch_samples = batch_samples.view(B, N, Ta*Da)
        
        # pass through encoder
        # B*To, Do
        embed:torch.Tensor = self.encoder(batch_dict_obs)
        # B*To, Do -> B,1, To*Do
        embed = embed.view(B, 1, self.n_obs_steps * embed.shape[-1])
        # B,1, To*Do -> B, N, To*Do
        embed = embed.expand(-1, N, -1)
        # concatenate with action
        # B, N, To*Do + B, N, Ta*Da -> B* N, To*Do + Ta*Da
        fused = torch.cat([embed, batch_samples], dim=-1).reshape(B*N, -1)
        # pass through MLP
        out = self._mlp(fused)
        # reshape to B, N
        return out.view(B, N)



def give_default_config(config:dict):
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
        reducer = config.get("reducer", "SpatialSoftArgmax")
        if reducer == "SpatialSoftArgmax":
            MLP_config["input_dim"] *= 2

        # add target dim
        MLP_config["input_dim"] += config.get("action_dim", 2)

        spatial_reduction = "SpatialSoftArgmax"
        coord_conv = False
        return dict(
            CNN_config=CNN_config,
            MLP_config=MLP_config,
            spatial_reduction=spatial_reduction,
            coord_conv=coord_conv
        )

if __name__ == "__main__":
    # config = ConvMLPConfig(
    #     cnn_config=CNNConfig(5),
    #     mlp_config=MLPConfig(32, 128, 2, 2),
    #     spatial_reduction=SpatialReduction.AVERAGE_POOL,
    #     coord_conv=True,
    # )

    # net = ConvMLP(config)
    # print(net)

    # x = torch.randn(2, 3, 96, 96)
    # with torch.no_grad():
    #     out = net(x)
    # print(out.shape)
    
    # from configs.configs_tools import load_config
    # config = load_config('robomimic_ibc.yaml')
    import yaml
    config_dir = 'configs/'
    config_name = 'robomimic_ibc.yaml'
    config_name = config_dir + config_name
    with open(config_name, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            config = None
    config['model_config']['MLP_config']['output_dim'] = 1
    net = EBMWithRoboMimicEnoder(config['model_config'])