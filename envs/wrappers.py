import gymnasium as gym


LOW_DIM_OBS_KEY = {"PushTImageEnv": "agent_pos"}

class DictReturnWrapper(gym.Wrapper):
    '''
    Wrapper that returns a dictionary of observations instead of a single observation
    keys:
    - "image": the image observation
    - "low_dim": the low dimensional observation
    - "is_first": whether this is the first observation in the episode
    - "is_last": whether this is the last observation in the episode
    '''
    def __init__(self, env):
        super().__init__(env)
        self._env: gym.Env = env
        # to record the low dimensional observation key for different environments
        self._low_dim_obs_key = LOW_DIM_OBS_KEY.get(env.__class__.__name__, "agent_pos")

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs["low_dim"] = obs[self._low_dim_obs_key]
        del obs[self._low_dim_obs_key]
        obs["is_first"] = False
        obs["is_last"] = done
        return obs, reward, done, {"info": info}

    def reset(self):
        obs = self._env.reset()
        obs["low_dim"] = obs[self._low_dim_obs_key]
        del obs[self._low_dim_obs_key]
        obs["is_first"] = True
        obs["is_last"] = False
        return obs
    
class ImageUnnomalizeWrapper(gym.Wrapper):

    def __init__(self, env):
        self._env: gym.Env = env 

    def step(self, action):
        # unnomalize action from [-1, 1] to [0, 512]
        action = (action + 1) * 256
        obs, reward, done, info = self._env.step(action) 
        obs['image'] *= 255 
        # TODO:check whether the image is flipped alone H and W
        # obs['image'] = obs['image'].swapaxes(1,2)
        return obs, reward, done, info
    
    def reset(self):
        obs = self._env.reset() 
        obs['image'] *= 255 
        # obs['image'] = obs['image'].swapaxes(1,2)
        return obs