#@title my_envs/environment.py
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlagents.my_envs")

from mlagents.envs.base_unity_environment import BaseUnityEnvironment
from mlagents.envs.timers import timed
from mlagents.envs.brain import BrainInfo, BrainParameters, AllBrainInfo

import gym

class MyUnityEnvironment(BaseUnityEnvironment):
  SCALAR_ACTION_TYPES = (int, np.int32, np.int64, float, np.float32, np.float64)
  SINGLE_BRAIN_ACTION_TYPES = SCALAR_ACTION_TYPES + (list, np.ndarray)
  SINGLE_BRAIN_TEXT_TYPES = list

  def __init__(
        self,
        file_name: Optional[str] = None,
        seed: int = 0,
        no_graphics: bool = False,
        train_model: bool = True, #Add Original
        trainer_config: Dict = {},
        args: Optional[List[str]] = None
  ):

    self._brain_name = "CartPoleBrain"
    self._env = gym.make("CartPole-v0")
    self._n_actions, observation_size = 2, 4

    self._loaded = True
    self._n_agents = {}
    self._global_done = None
    self._academy_name = "MockAcademy"
    self._log_path = ""
    self._brains = {}
    self._brain_names = []
    self._external_brain_names = []
    self._brain_names += [self._brain_name]
    self._action_type = 0 #discrete:0 / or continue 1
    self._brains[self._brain_names[0]] = \
        BrainParameters(self._brain_names[0],
            observation_size, #vector_observation_space_size
            1, #num_stacked_vector_observations
            [], #camera_resolutions
            [self._n_actions], #vector_action_space_size
            ['']*self._n_actions, #vector_action_descriptions
            self._action_type #vector_action_space_type
        )
    self._external_brain_names += [self._brain_names[0]]
    self._num_brains = len(self._brain_names)
    self._num_external_brains = len(self._external_brain_names)
    self._resetParameters = {}
    try:
      self._memory_size = int(trainer_config[self._brain_name]['memory_size'])
    except:
      self._memory_size = int(trainer_config['default']['memory_size'])
    logger.info("\n'{0}' started successfully! observation size={1}\n\n".format(self._academy_name, observation_size))

  @property
  def logfile_path(self):
    return self._log_path

  @property
  def brains(self):
    return self._brains

  @property
  def global_done(self):
    return self._global_done

  @property
  def academy_name(self):
    return self._academy_name

  @property
  def number_brains(self):
    return self._num_brains

  @property
  def number_external_brains(self):
    return self._num_external_brains

  @property
  def brain_names(self):
    return self._brain_names

  @property
  def external_brain_names(self):
    return self._external_brain_names

  def get_communicator(self, worker_id, base_port):
    return RpcCommunicator(worker_id, base_port)
    # return SocketCommunicator(worker_id, base_port)

  @property
  def external_brains(self):
    external_brains = {}
    for brain_name in self.external_brain_names:
      external_brains[brain_name] = self.brains[brain_name]
    return external_brains

  @property
  def reset_parameters(self):
    return self._resetParameters

  def reset(
        self,
        config: Dict = None,
        train_mode: bool = True,
        custom_reset_parameters: Any = None,
  ) -> AllBrainInfo:


    s = self._env.reset()
    try:
      action_mask = self._env.portfolio.action_mask
    except:
      action_mask = [1.0]*self._n_actions
    self._global_done = False
    state = {}
    state[self._brain_name] = BrainInfo(
        visual_observation=[],
        vector_observation=np.nan_to_num([s]),
        text_observations= [''],
        memory=np.array([[0.0]*self._memory_size]),
        reward=[0.0],
        agents=[-1000],
        local_done=[False],
        vector_action=np.array([[0.0]]),
        text_action=[''],
        max_reached=[False],
        action_mask=np.array([action_mask])
#        action_mask=np.array([[1.0]*self._n_actions])
        )
    return state

  @timed
  def step(
        self,
        vector_action: Dict[str, np.ndarray] = None,
        memory: Optional[Dict[str, np.ndarray]] = None,
        text_action: Optional[Dict[str, List[str]]] = None,
        value: Optional[Dict[str, np.ndarray]] = None,
        custom_action: Dict[str, Any] = None,
  ) -> AllBrainInfo:

    vector_action = {} if vector_action is None else vector_action
    memory = {} if memory is None else memory
    text_action = {} if text_action is None else text_action
    value = {} if value is None else value
    custom_action = {} if custom_action is None else custom_action

    max_reached = False
    action_mask = [1.0]*self._n_actions
    action = int(vector_action[self._brain_name][0])
    s_, r, done, info = self._env.step(action)
    if 'max_reached' in info:
      max_reached = info['max_reached']
    if 'action_mask' in info:
      action_mask  = info['action_mask']
    if done:
      self._global_done = done
    state = {}
    state[self._brain_name] = BrainInfo(
        visual_observation=[],
        vector_observation=np.nan_to_num([s_]),
        text_observations= [''],
        memory=np.array(memory[self._brain_name]),
        reward=[r],
        agents=[-1000],
        local_done=[done],
        vector_action=np.array(vector_action[self._brain_name]),
        text_action=[''],
        max_reached=[max_reached],
        action_mask=np.array([action_mask])
        )
    if not self._env.no_graphics:
      self._env.render()
    return state

  def close(self):
    if self._loaded:
      self._close()
    else:
      raise Exception("No My environment is loaded.")

  def _close(self):
    pass

