# class for online evaluation of the model
# TODO: how to ensure obs from env have the correct dimension?
# TODO: probably add multi-threading for evaluation?
import torch
import gymnasium as gym
from utils.logger import Logger
import numpy as np
from tqdm import tqdm
from collections import deque


def evaluate_single_step(
        agent,
        env: gym.Env,
        logger: Logger,
        eval_episodes: int,
        max_episode_steps: int = 1000, 
        use_low_dim_observation: bool = False, 
        normalize_image: bool = False):
    '''
    Evaluate the model in the environment
    This is for single step models
    '''
    metrics = {"mean_episode_reward": 0, "episode_success_rate": 0}
    mean_episode_reward = []
    episode_success_rate = []
    obs_images_for_video = []

    # print("Evaluating the agent for {} episodes".format(eval_episodes))
    agent.eval()
    with tqdm(total=eval_episodes) as pbar:
        for i in range(eval_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0
            while not done and episode_steps < max_episode_steps:
                obs_image = obs["image"]
                if normalize_image:
                      obs_image = np.clip(obs_image, 0, 255) / 255
                # TODO: modify this for other low-dim names
                if use_low_dim_observation:
                    obs_low_dim = obs["low_dim"]
                    obs_whole = {"image": obs_image, "low_dim": obs_low_dim}
                else:
                    obs_whole = {"image": obs_image}
                with torch.no_grad():
                    action = agent.get_actions(obs_whole)
                    # print("action", action)
                # if action.any() != -1. and action.any() != 1.:
                #     print("action", action)
                obs, reward, done, _ = env.step(action)
                # save images on first episode for visualization
                if i == 0:
                    #  C, H, W -> H, W, C for adapting to logger
                    # change to uint8 for not confusing with normalized images
                    # this is to minimize the number of changes in the logger
                    # TODO: change this for adapting to normalized images and unnormalized images
                    obs_image_processed = obs_image.copy().transpose(1, 2, 0).astype(np.uint8)
                    obs_images_for_video.append(obs_image_processed)
                episode_reward += reward
                episode_steps += 1

            if done:
                episode_success_rate.append(1)
            else:
                episode_success_rate.append(0)
            mean_episode_reward.append(episode_reward)
            pbar.update(1)
            pbar.set_postfix({"mean_episode_reward": np.mean(mean_episode_reward), "episode_success_rate": np.mean(episode_success_rate)})
    metrics["mean_episode_reward"] = np.mean(mean_episode_reward)
    metrics["episode_success_rate"] = np.mean(episode_success_rate)
    agent.train()
    # need to make video a batch of 1
    return metrics, [obs_images_for_video]


def evaluate_multi_step(
        agent,
        envs: list[gym.Env],
        logger,
        eval_episodes: int,
        max_episode_steps: int = 1000, 
        use_low_dim_observation: bool = False, 
        normalized_image: bool = False,
        image_shape: tuple = (3, 64, 64),
        n_obs_steps: int = 1,
        action_dim: int = 2,
        n_action_steps: int = 1,
        n_predict_steps: int = 1,
        video_image_key: str = "image"):
    '''
    Evaluate the model in the environment
    This is for multi-step models
    Also adapted for multi-environment evaluation
    '''
    metrics = {"mean_episode_reward": 0, "episode_success_rate": 0}
    mean_episode_reward = []
    episode_success_rate = []
    obs_images_for_video = []
    agent.eval()

    
    # reset all the environments and get the initial observations
    obs_list = []
    obs_list = [env.reset() for env in envs]
    obs_list = [obs() for obs in obs_list]
    # use queue to store the observations
    obs_queues = [deque(maxlen=n_obs_steps) for _ in range(len(envs))]
    obs_images_for_video = deque(maxlen=max_episode_steps)
    # push the initial observations to the queue, n_obs_steps times
    for i in range(n_obs_steps):
        for j, obs in enumerate(obs_list):
            obs_queues[j].append(obs)
    
    reward_list = [0 for _ in range(len(envs))]
    done_list = [0 for _ in range(len(envs))]
    episode_steps_list = [0 for _ in range(len(envs))]

    mean_episode_reward = []
    current_eval_episodes = 0
    previous_eval_episodes = 0
    with tqdm(total=eval_episodes) as pbar:
        while current_eval_episodes < eval_episodes:
            # if done, reset the environment and get the initial observations
            if any(done_list) or any([episode_steps >= max_episode_steps for episode_steps in episode_steps_list]):
                # get the indicies of the environments that are done
                indicies = [i for i, done in enumerate(done_list) if done or episode_steps_list[i] >= max_episode_steps]
                # reset the environments
                input_obs = [envs[i].reset() for i in indicies]
                # get the observations from returned Future objects
                input_obs = [i() for i in input_obs]
                # clear the queue and push the initial observations to the queue
                for i in indicies:
                    obs_queues[i].clear()
                    for _ in range(n_obs_steps):
                        obs_queues[i].append(input_obs[i])
                # update current_eval_episodes
                current_eval_episodes += len(indicies)
                # update the done_list and record the success rate
                for i in indicies:
                    if done_list[i]:
                        episode_success_rate.append(1)
                    else:
                        episode_success_rate.append(0)
                    done_list[i] = 0
                # put rewards to the whole_episode_reward
                for i in indicies:
                    mean_episode_reward.append(reward_list[i])
                    reward_list[i] = 0
            
            # get the observations from the queue
            # input_obs: list of list of observations
            input_obs = [list(obs_queue) for obs_queue in obs_queues]
            # print("input_obs", input_obs)
            # make obs a dictionary, with keys as the keys of the first observation and values as the stacked observations
            # input_obs = {k: np.stack([obs[k] for obs in input_obs]) for k in input_obs[0].keys()}
            for i in range(len(input_obs)):
                # we first stack the observations for each step
                input_obs[i] = {k: np.stack([obs[k] for obs in input_obs[i]]) for k in input_obs[i][0].keys()}
            # then we stack the observations for each environment
            input_obs = {k: np.stack([obs[k] for obs in input_obs]) for k in input_obs[0].keys()}
            # get actions from the agent
            actions = agent.get_actions(input_obs)
            # print("actions shape", actions.shape)  
            for i in range(n_action_steps):
                action = actions[:, i,...]
                # step the environments
                results = [envs[j].step(action[j]) for j in range(len(envs))]
                results = [result() for result in results]
                # update the observations and rewards
                for i, (obs, reward, done, _) in enumerate(results):
                    obs_queues[i].append(obs)
                    reward_list[i] += reward
                    done_list[i] = done
                    # save images of the first environment for visualizationq
                    if i == 0:
                        obs_image = obs[video_image_key]
                        obs_image_processed = process_image_for_logger(obs_image, normalized_image, image_shape)
                        obs_images_for_video.append(obs_image_processed)
            episode_steps_list = [episode_steps + n_action_steps for episode_steps in episode_steps_list]
            pbar.update(current_eval_episodes - previous_eval_episodes)
            previous_eval_episodes = current_eval_episodes
    
    metrics["mean_episode_reward"] = np.mean(mean_episode_reward)
    metrics["episode_success_rate"] = np.mean(episode_success_rate)
    return metrics, list(obs_images_for_video)


def process_image_for_logger(
        obs_image: np.ndarray,
        is_normalized_image: bool = False,
        image_shape: tuple = (3, 64, 64)):
    '''
    Process the image for logger
    output unnormalized , H, W, C image
    '''
    if is_normalized_image:
        obs_image = np.clip(obs_image, 0, 1) * 255
    obs_image = obs_image.astype(np.uint8)
    assert image_shape[0] == 3 or image_shape[-1] == 3, "first or last dimension of image_shape must be 3"
    if image_shape[0] == 3:
        obs_image = obs_image.transpose(1, 2, 0)
    return obs_image