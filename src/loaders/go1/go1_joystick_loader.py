import functools

from brax.base import State as PipelineState
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac

from mujoco_playground._src import mjx_env
from mujoco_playground._src import collision


import functools
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import registry
from mujoco_playground import wrapper
from src.configs.go1.joystick_default_config import default_config
from src.brax_ppo_configs.go1.joystick_brax_ppo_config import brax_ppo_config

from src.randomizers.go1.go1_randomizer import domain_randomize
from src.configs.constants.go1_constants import Go1Constants
from src.loaders.training_env_loader import TrainingEnvLoader
from src.environments.go1.go1_handstand_env import Go1JoystickEnv
from brax.training.agents import ppo
from src.utils.progress import progress
from datetime import datetime




go1_default_config = default_config
go1_brax_ppo_config = brax_ppo_config
go1_randomizer = domain_randomize  # `randomizer` refers to the function directly
go1_constants = Go1Constants


go1_loader = TrainingEnvLoader(Go1JoystickEnv,
                                          go1_constants,
                                          go1_default_config,
                                          go1_brax_ppo_config,
                                          domain_randomize)
go1_js_env_test = go1_loader.get_env()
ppo_params = go1_loader.get_brax_ppo_config()

env_name = 'Go1JoystickFlatTerrain'
env_cfg_ref = registry.get_default_config(env_name)
env_ref = registry.load(env_name)

times = [datetime.now()]


ppo_training_params = dict(ppo_params)
network_factory = ppo_networks.make_ppo_networks
if "network_factory" in ppo_params:
  del ppo_training_params["network_factory"]
  network_factory = functools.partial(
      ppo_networks.make_ppo_networks,
      **ppo_params.network_factory
  )

train_fn = functools.partial(
    ppo.train, **dict(ppo_training_params),
    network_factory=network_factory,
    randomization_fn=go1_randomizer,
    progress_fn=progress
)

make_inference_fn, params, metrics = train_fn(
    environment=go1_js_env_test,
    eval_env=registry.load(env_name, config=env_cfg),
    wrap_env_fn=wrapper.wrap_for_brax_training,
)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")