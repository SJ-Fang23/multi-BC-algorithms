import torch
import torchvision
import torch.nn as nn

from utils.pytorch_utils import get_device
from diffusers.training_utils import EMAModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from typing import Union

import numpy as np

DEVICE = get_device()

# network modules
class SinPositionalEmbedding(nn.Module):
    def __init__(self,
                 embedding_dim) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
    
    # x: (seq_len)
    def forward(self, x):
        # if isinstance(x, int):
        #     x = torch.tensor([x],dtype=torch.long)
        # if not isinstance(x, torch.Tensor):
        #     x = torch.tensor(x, device=DEVICE)
        half_dim = self.embedding_dim // 2
        # 1/10000^(2i/d)
        embedding_weights = torch.exp(-torch.log(torch.tensor(10000)) * (torch.arange(half_dim, device=DEVICE)/(half_dim - 1)))
        # output shape: (seq_len, embedding_dim)
        pos_embedding = x.unsqueeze(-1) * embedding_weights.unsqueeze(0)
        pos_embedding = torch.cat((torch.sin(pos_embedding), torch.cos(pos_embedding)), dim=-1)
        return pos_embedding

# conv1d + groupnoim + mish
class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=8 ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel size for 1D Conv must be odd"
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.net = nn.Sequential(
            self.conv,
            nn.GroupNorm(groups, out_channels),
            nn.Mish()
        ) 
    
    def forward(self, x):
        return self.net(x)

# 1D residual block with FILM embedding 
class ResidualBlockwithEmbed(nn.Module):
    def __init__(self,
                 in_channels, 
                 out_channels,
                 raw_embedding_dim,
                 kernel_size=3,
                 groups=8):
        super().__init__()
        self.inputConvLayer = Conv1dBlock(in_channels, out_channels, kernel_size, groups)
        self.outputConvLayer = Conv1dBlock(out_channels, out_channels, kernel_size, groups)

        self.embedding_dim = out_channels * 2
        self.embedding_layers = nn.Sequential(
            nn.Mish(),
            nn.Linear(raw_embedding_dim, self.embedding_dim)
        )
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()
    
    def forward(self, 
                x:torch.Tensor,
                embedding):
        res = x.clone()
        x = self.inputConvLayer(x)

        embedding = self.embedding_layers(embedding)
        weights, bias = torch.chunk(embedding, 2, dim=-1)
        x = weights.unsqueeze(-1) * x + bias.unsqueeze(-1)
        x = self.outputConvLayer(x)
        # print(x.shape, res.shape )
        return x + self.residual_conv(res)


class NoisePredictionNet(nn.Module):
    '''
    Noise prediction network based on UNet
    diffusion_step_positional_encoding_dim: embedding to indicate step of diffusion
    '''
    def __init__(self,
                  input_latent_dim, 
                  observation_embedding_dim, 
                  diffusion_step_positional_encoding_dim=256,
                  hidden_dims = [256,512,1024],
                  kernel_size = 5,
                  groups=8):
        super().__init__()
        step_dim = diffusion_step_positional_encoding_dim
        self.step_embedding_layers =  nn.Sequential(
            SinPositionalEmbedding(step_dim),
            nn.Linear(step_dim, step_dim * 4),
            nn.Mish(),
            nn.Linear(step_dim * 4, step_dim)
        )

        dims = [input_latent_dim] + hidden_dims
        embedding_dim = observation_embedding_dim + step_dim

        '''
        for each step of down sampling:
        two residual blocks with embedding and one downsample layer
        '''
        self.downlayers = nn.ModuleList([])
        for i in range(len(hidden_dims)):
            resblock1 = ResidualBlockwithEmbed(dims[i], 
                                               dims[i+1], 
                                               embedding_dim,
                                               kernel_size,
                                               groups)
            resblock2 = ResidualBlockwithEmbed(dims[i+1], 
                                               dims[i+1], 
                                               embedding_dim,
                                               kernel_size,
                                               groups)
            # this downsample layer cut length of sequence by half
            downsample = nn.Conv1d(dims[i+1], dims[i+1], kernel_size=3, stride=2, padding=1) \
                if not i == len(hidden_dims) - 1 else nn.Identity()
            self.downlayers.append(nn.ModuleList([resblock1, resblock2, downsample]))

        '''
        middle layers: two residual blocks with embedding
        '''
        self.middlelayers = nn.ModuleList([
            ResidualBlockwithEmbed(dims[-1], 
                                   dims[-1], 
                                   embedding_dim,
                                   kernel_size,
                                   groups),
            ResidualBlockwithEmbed(dims[-1], 
                                   dims[-1], 
                                   embedding_dim,
                                   kernel_size,
                                   groups)]
        )
        
        '''
        for each step of up sampling:
        1. concat with each latent generated by down sampling with same dim
        2. two residual block with embedding and one upsample layer
        '''
        self.uplayers = nn.ModuleList([])
        reversed_hidden_dims = hidden_dims[::-1]
        for i in range(len(hidden_dims)-1):
            resblock1 = ResidualBlockwithEmbed(reversed_hidden_dims[i] * 2,
                                               reversed_hidden_dims[i+1],
                                               embedding_dim,
                                               kernel_size,
                                               groups)
            resblock2 = ResidualBlockwithEmbed(reversed_hidden_dims[i+1], 
                                                reversed_hidden_dims[i+1], 
                                                embedding_dim,
                                                kernel_size,
                                                groups)
            # this upsample layer double length of sequence
            upsample = nn.ConvTranspose1d(reversed_hidden_dims[i+1], reversed_hidden_dims[i+1], 
                                          kernel_size=4, stride=2, padding=1) 
            self.uplayers.append(nn.ModuleList([resblock1, resblock2, upsample]))

            self.final_conv = nn.Sequential(
            Conv1dBlock(hidden_dims[0], hidden_dims[0], kernel_size=kernel_size),
            nn.Conv1d(hidden_dims[0], input_latent_dim, 1),
        )
        
    def forward(self, 
                x:torch.Tensor,
                step:Union[int, float, torch.Tensor],
                embedding:torch.Tensor):
        '''
        x: (B, T, C)
        step:(int) or (B)
        embedding: (B, embedding_dim)
        '''
        x = x.permute(0, 2, 1) # to (B,C,T)
        # step need to be (batch, seqlen) tensor
        if not isinstance(step, torch.Tensor):
            step = torch.tensor([step], device=DEVICE, dtype=torch.long)

        # step to (B)
        step = step.expand(x.shape[0])
        step_embedding = self.step_embedding_layers(step)
        # return shape: (B,embedding_dim)
        if embedding is not None:
            embedding_whole = torch.cat((embedding, step_embedding), dim=-1)
        else:
            embedding_whole = step_embedding
        # down sampling
        latent_states = []
        for (resblock1, resblock2, downsample) in self.downlayers:
            x = resblock1(x, embedding_whole)
            x = resblock2(x, embedding_whole)
            latent_states.append(x)
            x = downsample(x)
        # middle layers
        for resblock in self.middlelayers:
            x = resblock(x, embedding_whole)
        # up sampling
        for (resblock1, resblock2, upsample) in self.uplayers:
            latent = latent_states.pop()
            x = torch.cat((x, latent), dim=1)
            x = resblock1(x, embedding_whole)
            x = resblock2(x, embedding_whole)
            x = upsample(x)
        x = self.final_conv(x)
        return x.permute(0, 2, 1)

class DiffusionAgent:

    def __init__(self, 
                 num_diffusion_iters = 100,
                 vision_encoder = "resnet18",
                 vision_encoder_weights = "IMAGENET1K_V1",
                 vision_dim = 512,
                 low_dim_obs_dim = 2,
                 observation_steps = 2,
                 prediction_horizon = 16,
                 prediction_threshold = False
                 ):
        self.num_diffusion_iters = num_diffusion_iters
        self.betas = self.sample_beta()
        # alpha for noise in each step
        self.alphas = 1.0 - self.betas
        self.alpha_prod = torch.cumprod(self.alphas, dim=0, dtype=torch.float32).to(DEVICE)

        get_vision_encoder:torchvision = getattr(torchvision.models, vision_encoder)
        self.vision_encoder = get_vision_encoder(weights=vision_encoder_weights)
        self.vision_encoder = self.vision_encoder.to(DEVICE)
        self.vision_encoder.fc = nn.Identity()

        self.action_dim = 2
        self.obs_embedding_dim = (vision_dim + low_dim_obs_dim)*observation_steps
        self.noise_prediction_net = NoisePredictionNet(self.action_dim, self.obs_embedding_dim).to(DEVICE)

        self.pred_horizon = prediction_horizon
        self.pred_threshold = prediction_threshold

    def alpha_function(self,x):
        '''
        alpha function for beta
        '''
        return np.cos((x + 0.008) / 1.008 * np.pi / 2) ** 2

    def sample_beta(self, max_beta = 0.999):
        '''
        sample beta for each diffusion step
        '''
        betas = []
        for i in range(self.num_diffusion_iters):
            beta1 = i / self.num_diffusion_iters
            beta2 = ( i + 1 ) / self.num_diffusion_iters
            betas.append(min(1 - self.alpha_function(beta2)/self.alpha_function(beta1), max_beta))

        return torch.tensor(betas, device=DEVICE)
    
    def add_noise(self, 
                  batch_sample: torch.Tensor, 
                  time_steps:torch.IntTensor):
        '''
        add noise to sample
        batch_sample: (B, T, C)
        time_steps: (B)
        '''
        # generate noise randomly
        original_sample_shape = batch_sample.shape
        batch_sample = batch_sample.view(original_sample_shape[0], -1)

        noise = torch.randn(batch_sample.shape, device=DEVICE)
        alphas = self.alpha_prod[time_steps]

        # (B,1)
        sqrt_alphas = torch.sqrt(alphas).unsqueeze(-1).to(DEVICE)
        sqrt_1m_alphas = torch.sqrt(1 - alphas).unsqueeze(-1).to(DEVICE)
        # TODO: check for more dimensions

        sample_with_noise = sqrt_alphas * batch_sample + sqrt_1m_alphas * noise
        return sample_with_noise.view(original_sample_shape)
    
    def train_step(self, 
              batch_action: torch.Tensor, 
              batch_image: torch.Tensor,
              batch_low_dim_obs: torch.Tensor):
        '''
        train step for diffusion agent
        batch_action: (B, T, action_dim)
        batch_image: (B, T, C, H, W)
        batch_low_dim_obs: (B, T, low_dim_obs_dim)
        '''
        # get vision embedding
        batch_image_shape = batch_image.shape
        batch_image = batch_image.flatten(start_dim=0,end_dim=1)
        vision_embedding = self.vision_encoder(batch_image)
        vision_embedding = vision_embedding.view(batch_image_shape[0], batch_image_shape[1], -1)
        whole_embedding = torch.cat((vision_embedding, batch_low_dim_obs), dim=-1)
        # (B, T * embedding_dim)
        whole_embedding = whole_embedding.view(batch_image_shape[0], -1)
        # generate time steps (B,)
        time_steps = torch.randint(0, self.num_diffusion_iters, (batch_action.shape[0],), device=DEVICE, dtype=torch.long)
        # add noise
        sample_with_noise = self.add_noise(batch_action, time_steps)
        # predict 
        noise_pred = self.noise_prediction_net(sample_with_noise, time_steps, whole_embedding)
        loss = nn.functional.mse_loss(noise_pred, sample_with_noise)
        return loss
    
    def denoise_step(self,
                     noisy_sample: torch.Tensor,
                     time_step: int, 
                     step_ratio: int):
        '''
        denoise step for diffusion agent
        noisy_sample: (B, T, action_dim)
        '''
        prev_time_step = time_step - step_ratio
        alpha_prod_t = self.alpha_prod[time_step]
        beta_t = self.betas[time_step]
        alpha_prod_t_prev = self.alpha_prod[prev_time_step] if prev_time_step > 0 else 1
        beta_t_prev = self.betas[prev_time_step]
        current_alpha = alpha_prod_t / alpha_prod_t_prev
        current_beta = 1 - current_alpha
        sqrt_alpha = torch.sqrt(current_alpha)
        sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t)

        pred_original_sample = 1/sqrt_alpha_prod_t * (noisy_sample - torch.sqrt(beta_t) * self.noise_prediction_net(noisy_sample, time_step))
        if self.pred_threshold:
        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            raise NotImplementedError("Prediction threshold is not implemented yet")
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta) / beta_t
        current_sample_coeff = current_alpha ** (0.5) * beta_t_prev / beta_t
        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * noisy_sample

        variance = 0
        if time_step > 0:
            varaince_noise = torch.randn(pred_original_sample.shape, device=DEVICE, dtype=pred_original_sample.dtype)
            # TODO: only fixed small variance is implemented
            # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta * varaince_noise
        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample


    # denoising process
    def inference(self,
                  batch_obervation: torch.Tensor, 
                  reverse_diffusion_steps:int):
        noisy_action = torch.randn((batch_obervation.shape[0], self.pred_horizon, self.action_dim), device=DEVICE)
        # generate time steps (B,)
        step_ratio =  self.num_diffusion_iters // reverse_diffusion_steps
        assert step_ratio >= 1, "reverse_diffusion_steps should be smaller or equal to num_diffusion_iters"
        # gernerate time steps, reverse with step_ratio
        time_steps = torch.range(0, reverse_diffusion_steps, step=-step_ratio, device=DEVICE, dtype=torch.long)

        for time_step in time_steps:
            noisy_action = self.denoise_step(noisy_action, time_step, step_ratio)
        return noisy_action.detach().to("cpu").numpy()
            


    
    @property
    def parameters(self):
        return list(self.noise_prediction_net.parameters()) + list(self.vision_encoder.parameters())
    @property
    def device(self):
        return DEVICE
    




if __name__  == "__main__":
    # testres = ResidualBlockwithEmbed(32, 32, 24)
    # in_tensor = torch.randn(1, 32, 10)
    # embedding = torch.randn(1, 24)
    # out = testres(in_tensor, embedding)
    # print(out.shape)
    testnet = NoisePredictionNet(2, 24, 24).to(DEVICE)
    in_tensor = torch.randn(1, 16, 2).to(DEVICE)
    step = 1
    embedding = torch.randn(1, 24).to(DEVICE)
    out = testnet(in_tensor, step, embedding)
    print(out.shape)