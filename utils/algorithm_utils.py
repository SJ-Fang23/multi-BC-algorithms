# class for all the special classes and functions needed by algorithms


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.pytorch_utils import dict_apply


class DerivativeFreeOptimizer:
    """A simple derivative-free optimizer. Great for up to 5 dimensions."""

    device: torch.device
    noise_scale: float
    noise_shrink: float
    iters: int
    train_samples: int
    inference_samples: int
    bounds: np.ndarray

    def __init__(self, 
                 dataloader: torch.utils.data.DataLoader, 
                 config: dict) -> None:

        # TODO： change this to read shape meta from config
        self.bounds = dataloader.dataset.get_target_bounds()
        self.device = config["device"]
        optimizer_config = config["optimizer_config"] if "optimizer_config" in config else {}
        self.noise_scale = optimizer_config.get("noise_scale", 0.33)
        self.noise_shrink = optimizer_config.get("noise_shrink", 0.5)
        self.iters = optimizer_config.get("iters", 3)
        self.train_samples = optimizer_config.get("train_samples", 256)
        self.inference_samples = optimizer_config.get("inference_samples", 2 ** 12)
        
        
        action_shape_meta = config["shape_meta"]["action"]
        self.action_shape = list(action_shape_meta["shape"])

        self.n_obs_steps = config.get("n_obs_steps", 1)
        self.n_predict_steps = config.get('n_predict_steps', 1)
        obs_shape_meta = config["shape_meta"]["obs"]
        # get observation keys
        self.obs_keys = obs_shape_meta.keys()


    def _sample(self, num_samples: int) -> torch.Tensor:
        """Helper method for drawing samples from the uniform random distribution."""
        size = (num_samples, self.bounds.shape[1])
        samples = np.random.uniform(self.bounds[0, :], self.bounds[1, :], size=size)
        return torch.as_tensor(samples, dtype=torch.float32, device=self.device)

    def sample(self, batch_size: int, ebm: nn.Module) -> torch.Tensor:
        del ebm  # The derivative-free optimizer does not use the ebm for sampling.
        samples = self._sample(batch_size * self.train_samples)
        return samples.reshape(batch_size, self.train_samples, -1)

    @torch.no_grad()
    def infer(self, batch_input_dict: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """Optimize for the best action given a trained EBM.
        input: batch_input_dict: dict of torch.Tensor, B,T,D...
        """

        batch_dict_input = {key: batch_input_dict[key].to(self.device) for key in self.obs_keys}
        batch_input_dict = dict_apply(batch_dict_input, lambda x:  x[:, :self.n_obs_steps, ...])

        noise_scale = self.noise_scale
        bounds = torch.as_tensor(self.bounds).to(self.device)
        batch_size = batch_input_dict[list(self.obs_keys)[0]].size(0)
        # sample B * T * N samples, D
        sample_num = batch_size * self.n_predict_steps * self.inference_samples


        samples = self._sample(sample_num)
        # reshape to B, N, T, D
        samples = samples.reshape(batch_size, self.inference_samples,self.n_predict_steps, -1)
        # print("samples shape", samples.shape)

        for i in range(self.iters):
            # Compute energies.
            energies = ebm(batch_dict_input, samples)
            probs = F.softmax(-1.0 * energies, dim=-1)

            # Resample with replacement.
            idxs = torch.multinomial(probs, self.inference_samples, replacement=True)
            samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]

            # Add noise and clip to target bounds.
            samples = samples + torch.randn_like(samples) * noise_scale
            samples = samples.clamp(min=bounds[0, :], max=bounds[1, :])

            noise_scale *= self.noise_shrink

        # Return target with highest probability.
        energies = ebm(batch_input_dict, samples)
        probs = F.softmax(-1.0 * energies, dim=-1)
        best_idxs = probs.argmax(dim=-1)
        return samples[torch.arange(samples.size(0)), best_idxs,...]
    