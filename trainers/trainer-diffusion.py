from agents.diff_agent import  DiffusionAgent
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from utils.dataset_utils import pushTimgdataset
import tqdm
import torch
import numpy as np



def train_diffusion(
        dataset_path: str,
):
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    num_epochs = 100
    dataset = pushTimgdataset(dataset_path=dataset_path,
                                pred_horizon=pred_horizon,
                                obs_horizon=obs_horizon,
                                action_horizon=action_horizon)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )
    agent = DiffusionAgent(
        observation_steps=obs_horizon,
        prediction_horizon=pred_horizon)
    optimizer = torch.optim.AdamW(
        params=agent.parameters,
        lr=1e-4, weight_decay=1e-6)
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs)
    ema = EMAModel(
        parameters=agent.parameters,
        power=0.75)
    with tqdm.tqdm(range(num_epochs), desc='Epoch') as tglobal:
        for epoch in tglobal:
            epoch_loss = list()
            with tqdm.tqdm(dataloader, desc='Batch', leave=False) as tlocal:
                for batch in tlocal:
                    obs_img = batch['image'][:,:obs_horizon,...]
                    obs_low_dim = batch['agent_pos'][:,:obs_horizon,...]
                    actions = batch['action']
                    obs_img = obs_img.to(agent.device)
                    obs_low_dim = obs_low_dim.to(agent.device)
                    actions = actions.to(agent.device)
                    loss = agent.train_step(actions, obs_img, obs_low_dim)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    ema.step(agent.parameters)
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tlocal.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))


if __name__ == '__main__':
    train_diffusion(
        dataset_path="/home/wenchang/ilproj/dataset/pusht/pusht_cchi_v7_replay.zarr"
    )

                              