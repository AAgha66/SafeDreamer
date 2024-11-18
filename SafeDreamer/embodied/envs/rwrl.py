import functools
import os

import embodied
import numpy as np

SAFETY_COEFFS = {"cartpole": 0.3,"walker": 0.3,"quadruped": 0.5}
TASKS = {"cartpole": "realworld_swingup","walker": "realworld_walk","quadruped": "realworld_walk"}
CONSTRAINT_IDX = {"cartpole": 0,"walker": 1,"quadruped": 0}
class RWRL(embodied.Env):

  def __init__(self, env, platform='gpu', repeat=1, obs_key='observation', render=False, size=(64, 64), mode='train', camera_name='vision'):
    # TODO: This env variable is meant for headless GPU machines but may fail
    # on CPU-only machines.
    if platform =='gpu' and 'MUJOCO_GL' not in os.environ:
        os.environ['MUJOCO_GL'] = 'egl'
    import realworldrl_suite.environments as rwrl    
    self.constraint_idx = CONSTRAINT_IDX[env]
    self._camera = 2 if env == "quadruped" else 0
    env = rwrl.load(
            domain_name=env,
            task_name=TASKS[env],
            safety_spec=dict(
                enable=True, observations=True, safety_coeff=SAFETY_COEFFS[env]
            ),
            environment_kwargs={'flat_observation': False}
        )
    self._dmenv = env
    from . import from_dm
    self._env = from_dm.FromDM(self._dmenv,obs_key=obs_key)
    self._render = render if mode=='train' else True
    self._size = size
    self._repeat = repeat

  @property
  def repeat(self):
    return self._repeat

  @functools.cached_property
  def obs_space(self):
    spaces = self._env.obs_space.copy()
    del spaces["constraints"]
    spaces["cost"] = embodied.Space(np.bool, ())
    if self._render:
      spaces['image'] = embodied.Space(np.uint8, self._size + (3,))
      keys = list(spaces.keys())
      for k in keys:
        if k not in ["reward", "is_first", "is_last", "is_terminal", "image", "cost"]:
            del spaces[k]
    return spaces

  @functools.cached_property
  def act_space(self):
    return self._env.act_space

  def step(self, action):
    for key, space in self.act_space.items():
      if not space.discrete:
        assert np.isfinite(action[key]).all(), (key, action[key])
    action = action.copy()
    if action['reset']:
      obs = self._reset()
    else:
        reward = 0.0
        cost = 0.0
        for i in range(self._repeat):
            obs = self._env.step(action)
            reward += obs['reward']
            obs["cost"] = obs["constraints"][self.constraint_idx]
            del obs["constraints"]
            if "cost" in obs.keys():
                cost += obs['cost']
            if obs['is_last'] or obs['is_terminal']:
                break
        obs['reward'] = np.float32(reward)
        if "cost" in obs.keys():
            obs['cost'] = np.float32(cost)
    if self._render:
      obs['image'] = self.render()
      keys = list(obs.keys())
      for k in keys:
        if k not in ["reward", "is_first", "is_last", "is_terminal", "image", "cost"]:
            del obs[k]
    return obs

  def _reset(self):
    obs = self._env.step({'reset': True})
    obs["cost"] = obs["constraints"][self.constraint_idx]
    del obs["constraints"]
    if self._render:
      obs['image'] = self.render()
      keys = list(obs.keys())
      for k in keys:
        if k not in ["reward", "is_first", "is_last", "is_terminal", "image", "cost"]:
            del obs[k]
    return obs

  def render(self):
     return self._dmenv.physics.render(
            *self._size, camera_id=self._camera
        )