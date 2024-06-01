# import isaacgym

# assert isaacgym, "import isaacgym before pytorch"
import torch


class HistoryWrapper:
    def __init__(self, env):
        self.env = env

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
      
        return obs, rew, done, info

    def get_observations(self):
        obs = self.env.get_observations()
        return obs

    def get_obs(self):
        obs = self.env.get_obs()
        return obs

    def reset_idx(self, env_ids):  # it might be a problem that this isn't getting called!!
        ret = self.env.reset_idx(env_ids)
        return ret

    def reset(self):
        ret = self.env.reset()
        return ret

    def __getattr__(self, name):
        return getattr(self.env, name)
