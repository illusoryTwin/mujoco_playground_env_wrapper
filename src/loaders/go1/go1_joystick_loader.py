# @title Import MuJoCo, MJX, and Brax
from datetime import datetime
import functools
import os
from typing import Any, Dict, Sequence, Tuple, Union
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.io import html, mjcf, model
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from etils import epath
from flax import struct
from flax.training import orbax_utils
from IPython.display import HTML, clear_output
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import mediapy as media
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
from orbax import checkpoint as ocpfrom 

from typing import Any, Dict, Optional, Union, Type, Callable

import functools
import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math

from mujoco_playground._src import mjx_env
from mujoco_playground._src import collision



from configs.default_config import default_config


class Go1JoystickFlatTerrainLoader(TrainingEnvLoader):

  def set_model_constants(self):
    return Go1Constants

  def get_assets(self) -> Dict[str, bytes]:
    assets = {}
    xml_path = self.model_constants.XML_ROOT_PATH
    assets_path = xml_path / "assets"

    # menagerie_xml_path = mjx_env.MENAGERIE_PATH / "unitree_go1"
    menagerie_xml_path = self.model_constants.MENAGERIE_XML_PATH
    menagerie_assets_path = menagerie_xml_path / "assets"

    mjx_env.update_assets(assets, xml_path, "*.xml")
    mjx_env.update_assets(assets, assets_path)

    mjx_env.update_assets(assets, menagerie_xml_path, "*.xml")
    mjx_env.update_assets(assets, menagerie_assets_path)
    return assets


  def set_default_config(self):
    return default_config()

  def get_brax_ppo_config(self):
    return brax_ppo_config(self.config)

  def get_randomizer(self):
    return domain_randomize

  def get_env(self):
    return Go1JoystickEnv(
        self.model_constants,
        get_assets=self.get_assets,
        task="flat_terrain",
        config=self.config,
    )



