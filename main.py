import pathlib
import argparse
import yaml
import ibc.models
import networks.ibc
import utils.pytorch_utils as ptu
import utils.logger as logger
import utils.dataset_utils as dataset_utils
import torch.utils.data.dataloader as dataloader
from envs.wrappers import DictReturnWrapper, ImageUnnomalizeWrapper
import trainers.trainers as trainers
import torch
import ibc
import agents.agents as agents
from utils.tools import Every
from tqdm import tqdm
from evaluation import evaluate_single_step, evaluate_multi_step
import numpy as np
import networks
from utils.parallel import Parallel, Damy


TRAINER_CLASSES = {"ibc": trainers.TrainerIBC,"ibc_robomimic":trainers.TrainerIBC}
MODEL_CLASSES = {"ibc": ibc.models.EBMConvMLP, "ibc_robomimic": networks.ibc.EBMWithRoboMimicEnoder}
AGENT_CLASSES = {"ibc": agents.IBCAgent,"ibc_robomimic":agents.IBCAgent}


def make_env(env_name, env_config: dict):



    if env_name == "pusht":
        render_size = env_config.get("render_size", 96)
        from envs.pusht import PushTImageEnv
        env = PushTImageEnv(
            render_size=render_size
        )
        # env = DictReturnWrapper(env) 
        env = ImageUnnomalizeWrapper(env)
        
    else:
        raise NotImplementedError(f"Environment {env_name} not supported")
    
    return env


def main(config):
    # Seed RNGs.
    ptu.set_seed_everywhere(config["seed"])
    if config.get("deterministic_run", False):
        ptu.enable_deterministic_run()
    file_dir = pathlib.Path(__file__).parent
    log_dir = pathlib.Path(config["logdir"])
    # relative to drimer directory
    log_dir = file_dir / log_dir
    print(f"logging to {log_dir.resolve()}")

    checkpoint_dir = config.get("checkpoint_dir", "./checkpoints")
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    checkpoint_dir = file_dir / checkpoint_dir

    env_name = config["env_name"]
    action_repeat = config[f"{env_name}_config"]["action_repeat"]

    # allow training for a fixed number of steps or a fixed number of epochs
    max_training_steps = config["max_training_steps"] // action_repeat if "max_training_steps" in config else None
    max_training_epochs = config["epochs"] if "epochs" in config else None
    config.update({"max_training_steps": max_training_steps, "epochs": max_training_epochs})

    eval_every_n_steps = config["eval_every"] // action_repeat
    log_every_n_steps = config["log_every"] // action_repeat
    config.update({"eval_every": eval_every_n_steps, "log_every": log_every_n_steps})
    # TODO: check whether need use time_limit
    # config.time_limit //= config.action_repeat

    assert max_training_steps or max_training_epochs, "Either max_training_steps or max_training_epochs must be specified"

    # start making log directory
    log_dir.mkdir(parents=True, exist_ok=True)
    # make checkpoint directory
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ask user to input a name for this run
    run_name = input("Enter a name for this run: ")
    run_name = pathlib.Path(run_name)
    # create a logger object
    # TODO: add functions to give steps if continue trainings
    log = logger.Logger(log_dir, run_name, step=0) if not config["use_dummy_logger"] else logger.DummyLogger(log_dir, run_name, step=0)
    # log the config file
    log.log_config(config)

    device_name = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = ptu.get_device(device_name)
    config["device"] = device
    
    # get the dataset
    dataset_class_name = config["dataset"]
    dataset_class = getattr(dataset_utils, dataset_class_name)
    train_dataset = dataset_class(test_set = False, config = config)
    test_dataset = dataset_class(test_set = True, config = config)

    print(config)

    dataloaders = {
        "train": dataset_utils.get_dataloader(train_dataset, config),
        "test": dataset_utils.get_dataloader(test_dataset, config)
    }

    # get action dim for helping initialize model in correct dimensions

    # TODO: change this to read shape meta from config
    action_dim = train_dataset.action_dim
    config["action_dim"] = action_dim


    #get the environment
    assert "env_config" in config, "env_config must be specified in the config file"
    env_config = config["env_config"]
    env_num = config.get("env_num", 1)
    # env = make_env(env_name, env_config)
    envs = [make_env(env_name, env_config) for _ in range(env_num)]

    # make env parallel
    parallel_env = config.get("parallel", False)
    if parallel_env:
        # use multiprocessing to parallelize the environment
        envs = [Parallel(env, "process") for env in envs]
    else:
        # use dummy env wrapper
        envs = [Damy(env) for env in envs]

    model_name = config.get("model", "ibc")
    # get the model
    model = MODEL_CLASSES[model_name](config["model_config"])
    # get the agent
    agent = AGENT_CLASSES[model_name](model, dataloaders, config)

    # get the trainer

    trainer = TRAINER_CLASSES[model_name](model, dataloaders, log, config)

    # start training
    max_training_epochs = config.get("epochs", 100)
    max_training_steps = config.get("max_training_steps", None)
    eval_every_n_steps = config.get("eval_every", 1000)
    log_every_n_steps = config.get("log_every", 1000)

    use_online_eval = config.get("use_online_eval", False)
    online_eval_episodes = config.get("online_eval_steps", 10)
    max_online_episode_length = config.get("max_online_episode_length", 1000)
    # TODO: sync this with dataset settings
    # get multistep online eval settings
    use_normalized_image = config.get("use_normalized_image", False)
    image_to_visualize_key = config.get("image_to_visualize_key", "image")
    image_shape = config["shape_meta"]["obs"][image_to_visualize_key]["shape"]
    n_obs_steps = config.get("n_obs_steps", 1)
    n_action_steps = config.get("n_action_steps", 1)
    n_predict_steps = config.get("n_predict_steps", 1)

    save_every_n_steps = config.get("save_every", 3000)

    eval_every = Every(eval_every_n_steps)
    log_every = Every(log_every_n_steps)
    save_every = Every(save_every_n_steps)
    steps = 0
    train_metrics = {}
    with tqdm(total=max_training_epochs) as pbar:
        
        for epoch in range(max_training_epochs):
            steps += 350
            steps, metrics = trainer.train()
            train_metrics.update(metrics)
            if eval_every(steps):
                # pass
                # eval_metrics = trainer.eval(dataloaders["test"])
                # train_metrics.update(eval_metrics)
                # if use_online_eval:
                #     online_metrics, video_frames = evaluate_single_step(agent, env, log, online_eval_episodes, 
                #                         max_online_episode_length, 
                #                         use_low_dim_observation=False, 
                #                         normalize_image=False)
                #     train_metrics.update(online_metrics)
                if use_online_eval:
                    online_metrics, video_frames = evaluate_multi_step(
                        agent,
                        envs,
                        log,
                        online_eval_episodes,
                        max_online_episode_length,
                        use_low_dim_observation=False,
                        normalized_image=use_normalized_image,
                        image_shape=image_shape, 
                        n_obs_steps = n_obs_steps,
                        action_dim = action_dim,
                        n_action_steps=n_action_steps,
                        n_predict_steps=n_predict_steps,
                        video_image_key=image_to_visualize_key
                    )
                    train_metrics.update(online_metrics)

            pbar.set_postfix(train_metrics)

            if log_every(steps):
                # log.scalar("IBC_train_loss", train_metrics["IBC_train_loss"])
                # log.scalar("IBC_test_loss", train_metrics["IBC_test_loss"])
                for name, value in train_metrics.items():
                    log.scalar(name, value)
                video_frames = None
                if video_frames is not None:
                    # video_frames[0].append(np.array(train_dataset.normalized_train_data["image"][20].transpose(1, 2, 0).astype(np.uint8)))
                    log.video("BC_online_eval", video_frames)
                # print(train_dataset.normalized_train_data["image"][20])
                log.image("BC_train_image", train_dataset.normalized_train_data["image"][20].astype(np.uint8))
                log.write(step=steps)

            # save model
            if save_every(steps):
                trainer.save(checkpoint_dir, run_name, steps)
            pbar.update(1)
            pbar.refresh()
            if max_training_steps and steps >= max_training_steps:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,default="default")
    args, remaining = parser.parse_known_args()
    print(f"reading config file {args.config}.yaml")

    file_dir = pathlib.Path(__file__).parent
    config_path = file_dir / "configs" / f"{args.config}.yaml"
    # Load config file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Override config values with command line arguments
    for arg in remaining:
        key, value = arg.split("=")
        config[key] = value

    main(config)
    


